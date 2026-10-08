#!/usr/bin/env python3
"""Bounded, sequential API smoke checks for a live local DeepSeek-V4 server."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "dsv4_ext" / "build-hip-win" / "serve-real" / "api-checks.json"
BASE_URL = "http://127.0.0.1:8095"
TIMEOUT_SECONDS = 300
MODEL = "deepseek-v4-flash"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _save(results: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"base_url": BASE_URL, "timeout_seconds": TIMEOUT_SECONDS,
                               "results": results}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(OUTPUT)


def _http_request(path: str, payload: dict, streaming: bool = False) -> dict:
    url = BASE_URL + path
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if streaming else "application/json"}
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    result = {"url": url, "method": "POST", "headers": headers, "request": payload,
              "started_at_utc": _now(), "streaming": streaming}
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            result["http_status"] = response.status
            result["response_headers"] = dict(response.headers.items())
            if streaming:
                raw_lines: list[str] = []
                events: list[dict] = []
                text_parts: list[str] = []
                result["raw_response"] = ""
                result["events"] = events
                result["content"] = ""
                done = False
                first_content_latency = None
                while True:
                    line = response.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    raw_lines.append(decoded)
                    result["raw_response"] = "".join(raw_lines)
                    if not decoded.startswith("data:"):
                        continue
                    data = decoded[5:].strip()
                    if data == "[DONE]":
                        done = True
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    events.append(event)
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        chunk = delta.get("content")
                        if isinstance(chunk, str) and chunk:
                            if first_content_latency is None:
                                first_content_latency = time.perf_counter() - started
                            text_parts.append(chunk)
                            result["content"] = "".join(text_parts)
                result["done_marker"] = done
                result["first_content_latency_seconds"] = first_content_latency
            else:
                response_bytes = response.read()
                raw = response_bytes.decode("utf-8", errors="replace")
                result["raw_response"] = raw
                try:
                    result["response_json"] = json.loads(raw)
                except json.JSONDecodeError:
                    result["response_json"] = None
                result["content"] = _response_text(result["response_json"])
                # A non-streaming endpoint reveals content only when its complete response arrives.
                result["first_content_latency_seconds"] = time.perf_counter() - started if result["content"] else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        result.update({"http_status": exc.code, "response_headers": dict(exc.headers.items()),
                       "raw_response": raw, "response_json": _maybe_json(raw), "error": "HTTPError"})
    except Exception as exc:  # Persist transport failures as evidence and continue with the next case.
        result.update({"raw_response": "", "response_json": None,
                       "error": "%s: %s" % (type(exc).__name__, exc)})
    result["wall_time_seconds"] = time.perf_counter() - started
    result["finished_at_utc"] = _now()
    return result


def _maybe_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _response_text(data) -> str:
    if not isinstance(data, dict):
        return ""
    texts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if (isinstance(part, dict) and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)):
                texts.append(part["text"])
    return "".join(texts)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def main() -> int:
    checks = [
        ("chat-arithmetic-default", "/v1/chat/completions",
         {"model": MODEL, "messages": [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
          "max_tokens": 32, "temperature": 0}, False),
        ("chat-japanese-stream", "/v1/chat/completions",
         {"model": MODEL, "messages": [{"role": "user", "content": "日本の首都はどこですか？地名だけを日本語で答えてください。"}],
          "max_tokens": 32, "temperature": 0, "stream": True}, True),
        ("responses-moon", "/v1/responses",
         {"model": MODEL, "input": "In one short English sentence, what is the Moon?",
          "max_output_tokens": 48, "temperature": 0}, False),
    ]
    results: list[dict] = []
    passed = True
    for name, path, payload, streaming in checks:
        print("Starting %s..." % name, flush=True)
        result = _http_request(path, payload, streaming)
        result["name"] = name
        content = result.get("content", "")
        if name == "chat-arithmetic-default":
            data = result.get("response_json")
            chat_content = ""
            if isinstance(data, dict):
                choices = data.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    chat_content = _content_text(choices[0].get("message", {}).get("content"))
            result["content"] = chat_content
            result["pass"] = bool(chat_content.strip()) and re.search(r"\b4\b", chat_content) is not None
            result["assertions"] = {"nonempty_content": bool(chat_content.strip()), "contains_arithmetic_result_4": re.search(r"\b4\b", chat_content) is not None}
        elif name == "chat-japanese-stream":
            result["pass"] = bool(content.strip()) and "東京" in content and result.get("done_marker") is True
            result["assertions"] = {"nonempty_content": bool(content.strip()), "contains_tokyo": "東京" in content,
                                    "sse_done_marker": result.get("done_marker") is True}
        else:
            result["pass"] = bool(content.strip())
            result["assertions"] = {"nonempty_response_text": bool(content.strip())}
        status_ok = result.get("http_status") == 200
        result["assertions"]["http_status_200"] = status_ok
        result["pass"] = bool(result["pass"] and status_ok)
        if result.get("error"):
            result["pass"] = False
        results.append(result)
        _save(results)
        passed = passed and bool(result["pass"])
        print("%s: %s (%.3fs wall, first content %s)" %
              (name, "PASS" if result["pass"] else "FAIL", result["wall_time_seconds"],
               ("%.3fs" % result["first_content_latency_seconds"]
                if result.get("first_content_latency_seconds") is not None else "unobserved")), flush=True)
        if not result["pass"]:
            detail = result.get("raw_response", "")
            print("  response/error: %s" % (detail[:1200] or result.get("error", "assertion failed")), flush=True)
    print("Evidence: %s" % OUTPUT, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
