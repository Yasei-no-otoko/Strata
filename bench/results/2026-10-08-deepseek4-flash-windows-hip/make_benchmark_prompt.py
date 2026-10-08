#!/usr/bin/env python3
"""Build a deterministic, exact-token DeepSeek-V4-Flash API benchmark prompt."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path(
    r"C:\Dev\Strata-data\models\deepseek-v4-flash-UD-IQ1_M"
    r"\DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf"
)
MODEL_REVISION = "e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0"
MODEL_SHA256 = "279f2abec23194a3d293a78e74daccba4e2b96b04eacfc44c4a2439a7b06bfac"
DEFAULT_OUTPUT = ROOT / "dsv4_ext" / "build-hip-win" / "short-128-input"

OBSERVATIONS = [
    "At 06:10, the north rain gauge read 12 millimeters, and a loose cable was secured.",
    "At 06:40, the east footbridge remained clear; creek water stayed below the marked line.",
    "At 07:05, a radio check confirmed three volunteers at the shelter, and two batteries were replaced.",
    "At 07:30, clouds thinned; the west path reopened after a fallen branch was moved.",
]
REQUEST = (
    "Summarize these fictional observations in about 150 words. Separate confirmed observations, "
    "completed actions, and remaining uncertainty. Do not invent details."
)
TAG_WORDS = [
    "north", "east", "west", "creek", "bridge", "shelter", "radio", "gauge", "rain", "clouds",
    "path", "cable", "branch", "volunteers", "batteries", "water", "clear", "steady", "morning",
    "check", "record", "field", "safe", "marked", "level", "wind", "weather", "equipment",
    "route", "signal", "supply", "team", "dawn", "update", "status", "log", "confirmed",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="first GGUF shard containing the tokenizer and template")
    ap.add_argument("--tokens", type=int, default=128, help="exact rendered prompt token count (default: 128)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="fixture output directory")
    args = ap.parse_args()
    if args.tokens < 1:
        ap.error("--tokens must be positive")
    if not args.model.is_file():
        ap.error(f"model shard not found: {args.model}")
    actual_model_sha256 = sha256(args.model.read_bytes())
    if actual_model_sha256 != MODEL_SHA256:
        ap.error(f"first GGUF shard SHA-256 is not the pinned value: {actual_model_sha256}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT / "dsv4_ext" / "tools"))
    import strata_tokenizer as ST
    from gguf_reader import GGUFFile
    from serve.frontend import ChatTemplate
    from tokenizer_compat import apply_joyai_pretokenizer

    metadata = GGUFFile(args.model).metadata
    template_source = metadata.get("tokenizer.chat_template")
    if not template_source:
        raise SystemExit("the pinned GGUF does not contain tokenizer.chat_template")
    tokenizer = apply_joyai_pretokenizer(ST.Tokenizer.from_gguf(args.model))
    if tokenizer.pre != "joyai-llm":
        raise SystemExit(f"expected JoyAI tokenizer metadata, got {tokenizer.pre!r}")
    bos_id = int(metadata["tokenizer.ggml.bos_token_id"])
    bos_token = tokenizer.tokens[bos_id]
    eos_id = int(metadata.get("tokenizer.ggml.eos_token_id", 1))
    eos_token = tokenizer.tokens[eos_id]

    messages = [{"role": "user", "content": "\n\n".join(OBSERVATIONS + [REQUEST])}]
    with tempfile.TemporaryDirectory(prefix="dsv4-benchmark-template-") as td:
        template_path = Path(td) / "gguf-chat-template.jinja"
        template_path.write_text(template_source, encoding="utf-8", newline="\n")
        template = ChatTemplate(template_path)

        def render_and_encode(content: str) -> tuple[str, list[int]]:
            current_messages = [{"role": "user", "content": content}]
            rendered = template.render(
                current_messages,
                add_generation_prompt=True,
                enable_thinking=False,
                bos_token=bos_token,
                eos_token=eos_token,
            )
            return rendered, tokenizer.encode(rendered, parse_special=True)

        content = messages[0]["content"]
        rendered, ids = render_and_encode(content)
        if len(ids) > args.tokens:
            raise SystemExit(
                f"base fictional notes and 150-word-summary request already render to {len(ids)} tokens; "
                f"cannot fit exactly {args.tokens}"
            )

        # Add only ordinary English words inside a labeled synthetic log-tag field. Re-tokenize the full
        # rendered template after every candidate so BPE boundary effects and template overhead are counted.
        accepted: list[str] = []
        tag_prefix = "\n\nSupplementary log tags:"
        while len(ids) < args.tokens:
            accepted_one = False
            for word in [w for w in TAG_WORDS if w not in accepted] + accepted:
                tags = accepted + [word]
                candidate_content = "\n\n".join(OBSERVATIONS) + tag_prefix + " " + ", ".join(tags) + "\n\n" + REQUEST
                candidate_rendered, candidate_ids = render_and_encode(candidate_content)
                if len(ids) < len(candidate_ids) <= args.tokens:
                    accepted.append(word)
                    content, rendered, ids = candidate_content, candidate_rendered, candidate_ids
                    accepted_one = True
                    break
            if not accepted_one:
                # A different punctuation boundary can make a one-token remainder fit cleanly.
                base_tags = ", ".join(accepted)
                for word in [w for w in TAG_WORDS if w not in accepted] + accepted:
                    candidate_content = "\n\n".join(OBSERVATIONS) + tag_prefix + " " + base_tags + " " + word + "\n\n" + REQUEST
                    candidate_rendered, candidate_ids = render_and_encode(candidate_content)
                    if len(ids) < len(candidate_ids) <= args.tokens:
                        accepted.append(word)
                        content, rendered, ids = candidate_content, candidate_rendered, candidate_ids
                        accepted_one = True
                        break
            if not accepted_one:
                raise SystemExit(
                    f"could not reach exactly {args.tokens} tokens using complete ordinary-word log tags; "
                    f"current count is {len(ids)}"
                )

    if len(ids) != args.tokens:
        raise SystemExit(f"internal error: rendered prompt has {len(ids)} tokens, requested {args.tokens}")
    if not rendered.startswith(bos_token):
        raise SystemExit("embedded chat template did not render its BOS token")

    # Keep the serialized source messages exactly aligned with what was rendered.
    messages = [{"role": "user", "content": content}]
    args.output.mkdir(parents=True, exist_ok=True)
    messages_bytes = (json.dumps(messages, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    prompt_bytes = rendered.encode("utf-8")
    ids_text = ",".join(str(token_id) for token_id in ids)
    ids_bytes = (ids_text + "\n").encode("ascii")
    (args.output / "messages.json").write_bytes(messages_bytes)
    (args.output / "rendered-prompt.txt").write_bytes(prompt_bytes)
    (args.output / "prompt-ids.txt").write_bytes(ids_bytes)

    manifest = {
        "model": {
            "repository": "unsloth/DeepSeek-V4-Flash-GGUF",
            "revision": MODEL_REVISION,
            "quantization": "UD-IQ1_M",
            "first_shard": args.model.name,
            "first_shard_sha256": actual_model_sha256,
        },
        "tokenizer": {"gguf_model": metadata.get("tokenizer.ggml.model"),
                      "pre_tokenizer": tokenizer.pre, "special_token_parsing": True},
        "chat_template": {
            "source": "embedded GGUF tokenizer.chat_template",
            "sha256": sha256(template_source.encode("utf-8")),
            "add_generation_prompt": True,
            "enable_thinking": False,
            "bos_token_id": bos_id,
            "bos_token": bos_token,
            "bos_included_by_template": True,
            "template_overhead_in_token_count": True,
        },
        "prompt_tokens": len(ids),
        "requested_tokens": args.tokens,
        "token_ids_format": "comma-separated decimal IDs, with a trailing newline in prompt-ids.txt",
        "sha256": {
            "messages.json": sha256(messages_bytes),
            "rendered-prompt.txt": sha256(prompt_bytes),
            "prompt-ids.txt": sha256(ids_bytes),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(f"tokens={len(ids)} bos={bos_id} joyai={tokenizer.pre} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
