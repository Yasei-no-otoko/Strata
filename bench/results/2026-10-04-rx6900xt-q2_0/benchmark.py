#!/usr/bin/env python3
"""Run serial fresh-prompt Q2_0 long-context benchmarks against a local Strata server."""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TARGETS = "32768,128000"
ENGINE_LINE = re.compile(
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read>\d+)(?: of \d+)? read in "
    r"(?P<read_ms>[\d.]+) ms \((?P<prefill>[\d.]+) tok/s\), (?P<generated>\d+) generated in "
    r"(?P<decode_ms>[\d.]+) ms \((?P<decode>[\d.]+) tok/s\)"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_request(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_log_from(path, offset):
    if path is None:
        return ""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            end = stream.tell()
            stream.seek(min(offset, end))
            return stream.read().decode("utf-8", "replace")
    except OSError as exc:
        return f"[log read error: {exc}]"


def request_body(content, maximum=256):
    return {
        "model": "strata",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "reasoning_effort": "none",
        "max_tokens": maximum,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def make_filler_line(index):
    return (f"def transform_{index:05d}(value: int, offset: int = {index}) -> int:\n"
            f"    return (value * {(index % 97) + 1} + offset + {index}) % 100003\n\n")


def sized_prompt(target_tokens, run_id, count_tokens):
    """Build a deterministic code-review prompt at the exact rendered token count."""
    prefix = (f"Benchmark nonce: {run_id}. This is a fresh independent request.\n"
              "Review this synthetic Python module and explain its behavior, edge cases, tests, and complexity.\n")
    suffix = ("\n\nExplain the deterministic integer transforms, modulo arithmetic, tests, complexity, "
              "and maintainability. Give concrete observations.\n")

    def build(lines):
        body = "".join(make_filler_line(index) for index in range(lines))
        return prefix + body + suffix

    # Grow an upper bound, then binary-search full rendered OpenAI prompts. The final
    # expected count is measured again from the exact request that will be sent.
    lo, hi = 0, max(8, target_tokens // 10)
    while count_tokens(build(hi)) < target_tokens:
        hi *= 2
        if hi > 1_000_000:
            raise RuntimeError(f"could not size prompt to {target_tokens} tokens")
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if count_tokens(build(middle)) <= target_tokens:
            lo = middle
        else:
            hi = middle - 1

    content = build(lo)
    actual = count_tokens(content)
    # Full functions do not hit every possible count. A short plain-text pad
    # supplies the remaining tokens, then the exact outgoing request is counted.
    for _ in range(100):
        if actual == target_tokens:
            break
        if actual > target_tokens:
            raise RuntimeError(f"prompt sizing overshot {target_tokens}: {actual}")
        content += " x"
        actual = count_tokens(content)
    if actual != target_tokens:
        raise RuntimeError(f"could not construct target {target_tokens}: local rendered count is {actual}")
    return content, actual


def history_record(metrics, started_wall):
    requests = metrics.get("requests", []) if isinstance(metrics, dict) else []
    candidates = [record for record in requests
                  if isinstance(record, dict) and record.get("time", 0) >= started_wall - 0.1]
    return candidates[-1] if candidates else None


def summarize(values):
    numbers = [value for value in values if isinstance(value, (int, float))]
    if not numbers:
        return {"median": None, "min": None, "max": None}
    return {"median": statistics.median(numbers), "min": min(numbers), "max": max(numbers)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/Strata"),
                        help="Strata source tree containing the local tokenizer and frontend")
    parser.add_argument("--pack", type=Path, default=Path("/workspace/Strata-data/packs/q2_0"),
                        help="prepared Q2_0 pack containing tokenizer/ files")
    parser.add_argument("--url", default="http://127.0.0.1:18080", help="local Strata base URL")
    parser.add_argument("--out", type=Path, default=Path("/workspace/results"),
                        help="directory for requests, raw SSE, outputs, records, and summaries")
    parser.add_argument("--log", type=Path, help="optional engine log to capture per-run appended lines")
    parser.add_argument("--targets", default=DEFAULT_TARGETS,
                        help="comma-separated rendered prompt-token targets (default: %(default)s)")
    parser.add_argument("--runs", type=int, default=3, help="measured runs per target (default: %(default)s)")
    args = parser.parse_args()

    try:
        if args.runs < 1:
            raise ValueError("--runs must be at least 1")
        targets = [int(value.strip()) for value in args.targets.split(",") if value.strip()]
        if not targets or any(target < 1 for target in targets):
            raise ValueError("--targets must be a non-empty comma-separated list of positive token counts")
        if len(set(targets)) != len(targets):
            raise ValueError("--targets must not contain duplicates")
    except ValueError as exc:
        parser.error(str(exc))

    args.out.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    sys.path[:0] = [str(args.root), str(args.root / "tools")]
    status_path = args.out / "status.json"
    results_path = args.out / "results.jsonl"
    fatal_path = args.out / "fatal.json"
    current_label = None

    def persist_status(state, **fields):
        status = {"state": state, "updated_at": utc_now(), "current_run": current_label, **fields}
        write_json(status_path, status)

    try:
        from strata_tokenizer import Tokenizer
        from serve.frontend import ChatTemplate, openai_to_messages

        tokenizer_dir = args.pack / "tokenizer"
        vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for token, number in vocab.items():
            tokens[number] = token
        tokenizer = Tokenizer(tokens, (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").splitlines(),
                              json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8")))
        template = ChatTemplate(tokenizer_dir / "chat_template.jinja")

        def count_tokens(content):
            req = request_body(content)
            messages, tools, kwargs = openai_to_messages(req)
            rendered = template.render(messages, tools, **kwargs)
            return len(tokenizer.encode(rendered, parse_special=True))

        initial_status = json_request(base + "/v1/status")
        health = json_request(base + "/health")
        if not health.get("loaded") or max(targets) + 256 + 8 > health.get("max_context", 0):
            raise RuntimeError("Loaded model context is too small for the requested input and output")
        initial_metrics = json_request(base + "/metrics")
        write_json(args.out / "initial-status.json", initial_status)
        write_json(args.out / "initial-metrics.json", initial_metrics)
        results_path.write_text("", encoding="utf-8")
        records = []
        persist_status("running", targets=targets, runs_per_target=args.runs,
                       started_at=utc_now(), completed_runs=0)

        def perform(label, content, expected_prompt_tokens=None, warmup=False):
            nonlocal current_label
            current_label = label
            payload = request_body(content, maximum=256)
            wire = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
            request_path = args.out / f"{safe_label}-request.json"
            sse_path = args.out / f"{safe_label}.sse"
            output_path = args.out / f"{safe_label}-output.txt"
            record_path = args.out / f"{safe_label}-record.json"
            request_path.write_bytes(wire + b"\n")
            try:
                metrics_before = json_request(base + "/metrics")
            except Exception as exc:  # Record a telemetry failure without discarding the request result.
                metrics_before = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                status_before = json_request(base + "/v1/status")
            except Exception as exc:
                status_before = {"error": f"{type(exc).__name__}: {exc}"}
            started_wall = time.time()
            log_offset = 0
            if args.log:
                try:
                    log_offset = args.log.stat().st_size
                except OSError:
                    pass

            started_perf = time.perf_counter()
            first_text_at = None
            chunks, text_parts, reasoning_parts = [], [], []
            finish_reason, usage, stream_error = None, None, None
            response_status = None
            last_progress_at = started_perf
            progress = {"label": label, "phase": "started", "started_at": utc_now(),
                        "expected_prompt_tokens": expected_prompt_tokens, "request_sha256": hashlib.sha256(wire).hexdigest()}
            write_json(args.out / f"{safe_label}-status.json", progress)
            persist_status("running", targets=targets, runs_per_target=args.runs,
                           completed_runs=len(records), current_phase="request_started")
            print(json.dumps({"event": "start", "run": label, "target_tokens": expected_prompt_tokens,
                              "time": utc_now()}, ensure_ascii=False), flush=True)
            wire_request = urllib.request.Request(base + "/v1/chat/completions", data=wire,
                                                  headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(wire_request, timeout=1800) as response, sse_path.open("wb") as raw:
                    response_status = response.status
                    for line in response:
                        raw.write(line)
                        raw.flush()
                        if not line.startswith(b"data: "):
                            now = time.perf_counter()
                            if now - last_progress_at >= 10:
                                progress.update({"phase": "streaming", "updated_at": utc_now(),
                                                 "elapsed_s": now - started_perf, "sse_bytes": raw.tell(),
                                                 "chunks": len(chunks), "text_characters": sum(map(len, text_parts))})
                                write_json(args.out / f"{safe_label}-status.json", progress)
                                print(json.dumps({"event": "progress", **progress}, ensure_ascii=False), flush=True)
                                last_progress_at = now
                            continue
                        data = line[6:].strip()
                        if data == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(data.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            stream_error = {"type": "invalid_sse_json", "message": str(exc)}
                            continue
                        chunks.append(chunk)
                        if chunk.get("error"):
                            stream_error = chunk["error"]
                        if "usage" in chunk:
                            usage = chunk["usage"]
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            text = delta.get("content") or ""
                            reasoning = delta.get("reasoning_content") or ""
                            if text or reasoning:
                                if first_text_at is None:
                                    first_text_at = time.perf_counter() - started_perf
                                text_parts.append(text)
                                reasoning_parts.append(reasoning)
                            finish_reason = choice.get("finish_reason") or finish_reason
                        now = time.perf_counter()
                        if now - last_progress_at >= 10:
                            progress.update({"phase": "streaming", "updated_at": utc_now(),
                                             "elapsed_s": now - started_perf, "sse_bytes": raw.tell(),
                                             "chunks": len(chunks), "text_characters": sum(map(len, text_parts))})
                            write_json(args.out / f"{safe_label}-status.json", progress)
                            print(json.dumps({"event": "progress", **progress}, ensure_ascii=False), flush=True)
                            last_progress_at = now
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                stream_error = {"type": "client_error", "message": f"{type(exc).__name__}: {exc}"}

            elapsed = time.perf_counter() - started_perf
            output_path.write_text("".join(text_parts), encoding="utf-8")
            (args.out / f"{safe_label}-reasoning.txt").write_text("".join(reasoning_parts), encoding="utf-8")
            try:
                metrics_after = json_request(base + "/metrics")
                metrics_error = None
            except Exception as exc:
                metrics_after = {}
                metrics_error = f"{type(exc).__name__}: {exc}"
            try:
                status_after = json_request(base + "/v1/status")
                status_error = None
            except Exception as exc:
                status_after = {}
                status_error = f"{type(exc).__name__}: {exc}"

            engine = history_record(metrics_after, started_wall)
            log_text = read_log_from(args.log, log_offset)
            log_lines = [line.rstrip() for line in log_text.splitlines() if line.strip()]
            parsed_log = []
            for line in log_lines:
                match = ENGINE_LINE.search(line)
                if match:
                    parsed_log.append({key: (float(value) if "." in value else int(value))
                                       for key, value in match.groupdict().items()})
            if args.log:
                (args.out / f"{safe_label}-engine-log.txt").write_text(
                    "\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")

            errors = []
            if stream_error:
                errors.append({"type": "stream", "detail": stream_error})
            if not text_parts:
                errors.append({"type": "empty_output", "detail": "no nonempty text delta received"})
            if engine is None:
                errors.append({"type": "missing_engine_metrics", "detail": "no request record appeared in /metrics"})
            if expected_prompt_tokens is not None and engine:
                actual_prompt = engine.get("prompt_tokens")
                if actual_prompt != expected_prompt_tokens:
                    errors.append({"type": "prompt_count_mismatch", "expected": expected_prompt_tokens,
                                   "engine": actual_prompt})
            if engine is not None and not warmup and engine.get("reused") != 0:
                errors.append({"type": "prefix_reuse", "reused": engine.get("reused")})
            api_cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            if not warmup and api_cached not in (None, 0):
                errors.append({"type": "api_prefix_reuse", "cached_tokens": api_cached})
            if expected_prompt_tokens is not None and usage and usage.get("prompt_tokens") != expected_prompt_tokens:
                errors.append({"type": "api_prompt_count_mismatch", "expected": expected_prompt_tokens,
                               "api": usage.get("prompt_tokens")})

            engine_generated = (engine or {}).get("engine_generated")
            if engine_generated == 256 and finish_reason == "length":
                completion_status = "reached_256_token_limit"
            else:
                completion_status = "early_stop_or_incomplete"
            if engine and usage and usage.get("completion_tokens") is not None and engine_generated is not None:
                if usage["completion_tokens"] != engine_generated:
                    errors.append({"type": "output_count_mismatch", "engine": engine_generated,
                                   "api": usage["completion_tokens"]})

            engine_timings = None
            if engine:
                prompt_count = engine.get("prompt_tokens") or 0
                reused = engine.get("reused") or 0
                prompt_ms = engine.get("prompt_ms")
                decode_ms = engine.get("decode_ms")
                engine_timings = {
                    "prompt_tokens": prompt_count, "fresh_prompt_tokens": max(0, prompt_count - reused),
                    "reused_tokens": reused, "prompt_ms": prompt_ms,
                    "prefill_tok_s": ((max(0, prompt_count - reused) / (prompt_ms / 1000))
                                      if prompt_ms else None),
                    "generated_tokens": engine_generated, "decode_ms": decode_ms,
                    "decode_tok_s": ((engine_generated / (decode_ms / 1000))
                                     if engine_generated is not None and decode_ms else None),
                }
            stream_timings = next((chunk["timings"] for chunk in reversed(chunks) if chunk.get("timings")), None)
            row = {
                "label": label, "warmup": warmup, "started_at": progress["started_at"], "finished_at": utc_now(),
                "target_prompt_tokens": expected_prompt_tokens,
                "request_sha256": hashlib.sha256(wire).hexdigest(),
                "request_file": request_path.name, "raw_sse_file": sse_path.name,
                "output_file": output_path.name, "reasoning_file": f"{safe_label}-reasoning.txt",
                "client_ttft_s": first_text_at, "client_total_latency_s": elapsed,
                "http_status": response_status, "finish_reason": finish_reason,
                "completion_status": completion_status, "usage": usage, "engine": engine,
                "metrics_before": metrics_before, "metrics_after": metrics_after,
                "status_before": status_before, "status_after": status_after,
                "metrics_error": metrics_error, "status_error": status_error,
                "engine_timings": engine_timings, "stream_timings": stream_timings,
                "prefix_reused_tokens": (engine or {}).get("reused"), "api_cached_tokens": api_cached,
                "engine_log_lines": log_lines[-30:], "parsed_engine_log_timings": parsed_log,
                "output_characters": sum(map(len, text_parts)),
                "reasoning_characters": sum(map(len, reasoning_parts)),
                "raw_chunk_count": len(chunks), "errors": errors,
            }
            write_json(record_path, row)
            records.append(row)
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.update({"phase": "completed_with_errors" if errors else "completed",
                             "updated_at": utc_now(), "finished_at": row["finished_at"],
                             "client_ttft_s": first_text_at, "client_total_latency_s": elapsed,
                             "engine_prompt_tokens": (engine or {}).get("prompt_tokens"),
                             "prefix_reused_tokens": (engine or {}).get("reused"),
                             "generated_tokens": engine_generated, "completion_status": completion_status,
                             "errors": errors})
            write_json(args.out / f"{safe_label}-status.json", progress)
            persist_status("running", targets=targets, runs_per_target=args.runs,
                           completed_runs=len(records), current_phase="request_finished")
            print(json.dumps({"event": "done", "run": label, "ttft_s": first_text_at,
                              "total_s": elapsed, "prompt_tokens": (engine or {}).get("prompt_tokens"),
                              "reused": (engine or {}).get("reused"), "generated": engine_generated,
                              "completion_status": completion_status, "errors": errors}, ensure_ascii=False),
                  flush=True)
            if errors:
                raise RuntimeError(f"{label} failed validation; see {record_path.name}")
            return row

        perform("warmup", "Benchmark warm-up: explain in detail how a Python heap maintains priority order.",
                expected_prompt_tokens=None, warmup=True)
        for target in targets:
            for run in range(1, args.runs + 1):
                label = f"tokens-{target}-run-{run}"
                nonce = f"rx6900xt-q2_0-{target}-trial-{run}"
                content, expected = sized_prompt(target, nonce, count_tokens)
                # Keep the prompt-construction result durable before network I/O starts.
                request_path = args.out / f"{label}-request.json"
                write_json(args.out / f"{label}-prompt-count.json", {
                    "target_prompt_tokens": target, "local_rendered_prompt_tokens": expected,
                    "content_characters": len(content), "nonce": nonce,
                })
                perform(label, content, expected_prompt_tokens=expected)

        groups = {}
        for target in targets:
            subset = [row for row in records if not row["warmup"] and row["label"].startswith(f"tokens-{target}-")]

            def engine_value(row, key):
                engine = row.get("engine") or {}
                return engine.get(key)

            groups[str(target)] = {
                "runs": len(subset),
                "target_prompt_tokens": target,
                "engine_prompt_tokens": summarize([engine_value(row, "prompt_tokens") for row in subset]),
                "prefix_reused_tokens": summarize([row.get("prefix_reused_tokens") for row in subset]),
                "generated_tokens": summarize([engine_value(row, "engine_generated") for row in subset]),
                "client_ttft_s": summarize([row.get("client_ttft_s") for row in subset]),
                "client_total_latency_s": summarize([row.get("client_total_latency_s") for row in subset]),
                "prefill_tok_s": summarize([(row.get("engine_timings") or {}).get("prefill_tok_s") for row in subset]),
                "decode_tok_s": summarize([(row.get("engine_timings") or {}).get("decode_tok_s") for row in subset]),
                "completion_statuses": [row.get("completion_status") for row in subset],
                "finish_reasons": [row.get("finish_reason") for row in subset],
            }
        summary = {
            "url": base, "targets": targets, "runs_per_target": args.runs,
            "request_settings": {"temperature": 0, "reasoning_effort": "none", "max_tokens": 256,
                                 "warmup_max_tokens": 256, "stream": True, "warmup_excluded": True},
            "prompt_count_method": "Strata local Tokenizer + OpenAI openai_to_messages + ChatTemplate renderer",
            "validation": {"engine_prompt_count_must_match_local": True,
                           "prefix_reused_tokens_must_be_zero_for_measured_runs": True,
                           "early_stops_are_recorded_without_fabricating_256_tokens": True},
            "case_results": groups, "results_jsonl": str(results_path),
        }
        write_json(args.out / "summary.json", summary)
        final_status = json_request(base + "/v1/status")
        final_metrics = json_request(base + "/metrics")
        write_json(args.out / "final-status.json", final_status)
        write_json(args.out / "final-metrics.json", final_metrics)
        persist_status("completed", targets=targets, runs_per_target=args.runs,
                       started_at=initial_status.get("time"), finished_at=utc_now(),
                       completed_runs=len(records), summary_file="summary.json")
        print(json.dumps({"event": "summary", **summary}, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        fatal = {"state": "failed", "time": utc_now(), "current_run": current_label,
                 "error_type": type(exc).__name__, "error": str(exc)}
        try:
            write_json(fatal_path, fatal)
            persist_status("failed", error_type=type(exc).__name__, error=str(exc), fatal_file=fatal_path.name)
        except Exception:
            pass
        print(json.dumps({"event": "fatal", **fatal}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
