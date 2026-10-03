#!/usr/bin/env python3
"""Offline integrity check for the six completed long-context benchmark records."""

import hashlib
import json
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "measured"
OUTPUT = DATA / "verification.json"
TARGETS = (32768, 128000)
RUNS = 3
MAX_CONTEXT = 131072
MAX_OUTPUT = 256


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(errors, label, condition, detail=""):
    if not condition:
        errors.append(f"{label}: {detail or 'check failed'}")


def same(errors, label, actual, expected):
    check(errors, label, actual == expected, f"expected {expected!r}, got {actual!r}")


def median_range(values):
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return {"median": None, "min": None, "max": None}
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    errors, records = [], {}
    for target in TARGETS:
        for run in range(1, RUNS + 1):
            label = f"tokens-{target}-run-{run}"
            path = DATA / f"{label}-record.json"
            if not path.is_file():
                errors.append(f"{label}: record is missing; all six runs must finish first")
                continue
            try:
                row = load(path)
                records[label] = row
                engine = row.get("engine") or {}
                usage = row.get("usage") or {}
                api_cache = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")

                same(errors, f"{label}.label", row.get("label"), label)
                same(errors, f"{label}.warmup", row.get("warmup"), False)
                same(errors, f"{label}.target", row.get("target_prompt_tokens"), target)
                same(errors, f"{label}.http_status", row.get("http_status"), 200)
                same(errors, f"{label}.finish_reason", row.get("finish_reason"), "length")
                same(errors, f"{label}.errors", row.get("errors"), [])
                same(errors, f"{label}.status_before.in_flight",
                     (row.get("status_before") or {}).get("activity", {}).get("in_flight"), 0)
                same(errors, f"{label}.engine_prompt_tokens", engine.get("prompt_tokens"), target)
                same(errors, f"{label}.engine_prompt_total", engine.get("prompt_total"), target)
                same(errors, f"{label}.engine_prompt_read", engine.get("prompt_read"), target)
                same(errors, f"{label}.api_prompt_tokens", usage.get("prompt_tokens"), target)
                same(errors, f"{label}.engine_reused", engine.get("reused"), 0)
                same(errors, f"{label}.record_reused", row.get("prefix_reused_tokens"), 0)
                check(errors, f"{label}.api_cached_tokens", api_cache in (None, 0), repr(api_cache))
                check(errors, f"{label}.record_api_cached_tokens", row.get("api_cached_tokens") in (None, 0),
                      repr(row.get("api_cached_tokens")))
                same(errors, f"{label}.engine_generated", engine.get("engine_generated"), MAX_OUTPUT)
                same(errors, f"{label}.engine_output_tokens", engine.get("output_tokens"), MAX_OUTPUT)
                same(errors, f"{label}.api_completion_tokens", usage.get("completion_tokens"), MAX_OUTPUT)
                same(errors, f"{label}.engine_finish", engine.get("finish"), "length")

                before = (row.get("metrics_before") or {}).get("requests") or []
                after = (row.get("metrics_after") or {}).get("requests") or []
                before_times = {entry.get("time") for entry in before}
                additions = [entry for entry in after if entry.get("time") not in before_times]
                same(errors, f"{label}.new_history_entries", len(additions), 1)
                engine_time = engine.get("time")
                same(errors, f"{label}.selected_history_match",
                     len([entry for entry in additions if entry.get("time") == engine_time]), 1)
                same(errors, f"{label}.selected_time_unique",
                     len([entry for entry in after if entry.get("time") == engine_time]), 1)

                request_path = DATA / row.get("request_file", "")
                if not request_path.is_file():
                    errors.append(f"{label}.request_file: missing {request_path.name}")
                else:
                    digest = hashlib.sha256(request_path.read_bytes().rstrip(b"\r\n")).hexdigest()
                    same(errors, f"{label}.request_sha256", digest, row.get("request_sha256"))
                same(errors, f"{label}.context_fit", target + MAX_OUTPUT <= MAX_CONTEXT, True)
                before_context = (row.get("status_before") or {}).get("context", {}).get("max_positions")
                check(errors, f"{label}.reported_context", isinstance(before_context, int) and
                      target + MAX_OUTPUT <= before_context, repr(before_context))

                sse_path = DATA / row.get("raw_sse_file", "")
                if not sse_path.is_file():
                    errors.append(f"{label}.raw_sse_file: missing {sse_path.name}")
                else:
                    sse_timings = []
                    for line in sse_path.read_bytes().splitlines():
                        if not line.startswith(b"data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == b"[DONE]":
                            continue
                        chunk = json.loads(payload)
                        if isinstance(chunk, dict) and isinstance(chunk.get("timings"), dict):
                            sse_timings.append(chunk["timings"])
                    check(errors, f"{label}.sse_timings", bool(sse_timings), "no timing chunk found")
                    if sse_timings:
                        timings = sse_timings[-1]
                        recorded = row.get("stream_timings") or {}
                        for key in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms"):
                            same(errors, f"{label}.sse_recorded.{key}", timings.get(key), recorded.get(key))
                        for key, value in (("cache_n", engine.get("reused")),
                                           ("prompt_n", engine.get("prompt_tokens")),
                                           ("prompt_ms", engine.get("prompt_ms")),
                                           ("predicted_n", engine.get("engine_generated")),
                                           ("predicted_ms", engine.get("decode_ms"))):
                            same(errors, f"{label}.sse_engine.{key}", timings.get(key), value)
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

    summary_path = DATA / "summary.json"
    summary = None
    try:
        if not summary_path.is_file():
            errors.append("summary.json: missing")
        else:
            summary = load(summary_path)
            same(errors, "summary.targets", summary.get("targets"), list(TARGETS))
            same(errors, "summary.runs_per_target", summary.get("runs_per_target"), RUNS)
            cases = summary.get("case_results") or {}
            metrics = {
                "engine_prompt_tokens": lambda row: (row.get("engine") or {}).get("prompt_tokens"),
                "prefix_reused_tokens": lambda row: row.get("prefix_reused_tokens"),
                "generated_tokens": lambda row: (row.get("engine") or {}).get("engine_generated"),
                "client_ttft_s": lambda row: row.get("client_ttft_s"),
                "client_total_latency_s": lambda row: row.get("client_total_latency_s"),
                "prefill_tok_s": lambda row: (row.get("engine_timings") or {}).get("prefill_tok_s"),
                "decode_tok_s": lambda row: (row.get("engine_timings") or {}).get("decode_tok_s"),
            }
            for target in TARGETS:
                label = str(target)
                subset = [records[f"tokens-{target}-run-{run}"] for run in range(1, RUNS + 1)
                          if f"tokens-{target}-run-{run}" in records]
                case = cases.get(label) or {}
                same(errors, f"summary.{label}.runs", case.get("runs"), RUNS)
                same(errors, f"summary.{label}.target_prompt_tokens", case.get("target_prompt_tokens"), target)
                for key, value_of in metrics.items():
                    same(errors, f"summary.{label}.{key}", case.get(key),
                         median_range([value_of(row) for row in subset]))
    except Exception as exc:
        errors.append(f"summary validation: {type(exc).__name__}: {exc}")

    result = {"status": "failed" if errors else "passed", "expected_targets": list(TARGETS),
              "expected_runs_per_target": RUNS, "measured_records_found": len(records), "errors": errors}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
