"""Optional DSV4 JoyAI BPE parity check using the GGUF vocabulary and Rust tokenizers.

Run with a tokenizer-bearing GGUF shard:
    python dsv4_ext/tests/test_tokenizer_joyai_oracle.py --model path/to/model-00001-of-00003.gguf

This compares Strata's BPE output with the independent Rust `tokenizers` BPE implementation,
constructed from the model's exact GGUF vocab and merges. JoyAI pre-tokenization is supplied by
`tokenizer_compat.py`, transcribed from the pinned upstream llama.cpp source
[`src/llama-vocab.cpp`](https://github.com/ggml-org/llama.cpp/blob/4f92965a7bfa9e8eb6519908ff962e23c2cb7b93/src/llama-vocab.cpp)
(JoyAI regex list at lines 297-305; `joyai-llm` mapping at lines 2342-2344); this script does not
independently verify that regex. It also checks that the GGUF chat template emits its declared BOS
token when supplied.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "dsv4_ext" / "tools"))


def _dependency_skip(exc: ImportError) -> int:
    print("SKIP: optional JoyAI oracle dependencies are unavailable: %s" % exc)
    print("Install the Python packages `regex` and `tokenizers` in this environment to run this check.")
    return 0


def _rust_bpe_ids(rust_tokenizer, pretokenizer, text: str, byte_to_unicode) -> list[int]:
    ids: list[int] = []
    for piece in pretokenizer.findall(text):
        mapped = "".join(byte_to_unicode[b] for b in piece.encode("utf-8"))
        # No Rust pre-tokenizer is installed: this invokes only the independent Rust BPE model
        # over each JoyAI piece, preventing a second regex implementation from obscuring the check.
        ids.extend(rust_tokenizer.encode(mapped, add_special_tokens=False).ids)
    return ids


def _reference_ids(tokenizer, rust_tokenizer, text: str, byte_to_unicode) -> list[int]:
    """Use adapter special-token boundaries but independent Rust BPE for all ordinary text."""
    out: list[int] = []
    pos = 0
    specials = tokenizer._special_re
    for match in specials.finditer(text) if specials is not None else ():
        if match.start() > pos:
            out.extend(_rust_bpe_ids(rust_tokenizer, tokenizer._re, text[pos:match.start()], byte_to_unicode))
        out.append(tokenizer.special_tokens[match.group(0)])
        pos = match.end()
    if pos < len(text):
        out.extend(_rust_bpe_ids(rust_tokenizer, tokenizer._re, text[pos:], byte_to_unicode))
    return out


def _check_bos_template(tokenizer, metadata) -> bool:
    template_source = metadata.get("tokenizer.chat_template")
    bos_id = metadata.get("tokenizer.ggml.bos_token_id")
    if not template_source or bos_id is None:
        print("SKIP: GGUF has no chat template or BOS token ID")
        return False
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:
        print("SKIP: optional Jinja2 BOS template check unavailable: %s" % exc)
        return False

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    rendered = env.from_string(template_source).render(
        messages=[{"role": "user", "content": "BOS check"}],
        add_generation_prompt=False,
        bos_token=tokenizer.tokens[int(bos_id)],
        eos_token=tokenizer.tokens[int(metadata.get("tokenizer.ggml.eos_token_id", 1))],
        thinking=False,
        enable_thinking=False,
        reasoning_effort=None,
        tools=None,
    )
    ids = tokenizer.encode(rendered, parse_special=True)
    if not ids or ids[0] != int(bos_id):
        raise AssertionError("rendered chat template did not tokenize to declared BOS id %s" % bos_id)
    print("PASS: chat template BOS tokenizes as ID %s" % bos_id)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="tokenizer-bearing GGUF file, normally shard 1")
    args = ap.parse_args()

    try:
        import regex  # noqa: F401 - dependency used by the project tokenizer and adapter
        from tokenizers import Tokenizer as RustTokenizer
        from tokenizers.models import BPE
    except ImportError as exc:
        return _dependency_skip(exc)

    from gguf_reader import GGUFFile
    import strata_tokenizer as ST
    from tokenizer_compat import apply_joyai_pretokenizer

    model_path = pathlib.Path(args.model)
    if not model_path.is_file():
        ap.error("model file does not exist: %s" % model_path)
    metadata = GGUFFile(model_path).metadata
    if metadata.get("tokenizer.ggml.pre") != "joyai-llm":
        ap.error("expected tokenizer.ggml.pre='joyai-llm', got %r" % metadata.get("tokenizer.ggml.pre"))
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        ap.error("expected tokenizer.ggml.model='gpt2', got %r" % metadata.get("tokenizer.ggml.model"))

    tokenizer = apply_joyai_pretokenizer(ST.Tokenizer.from_gguf(model_path))
    merges = [tuple(merge.split(" ")) for merge in metadata["tokenizer.ggml.merges"]]
    rust = RustTokenizer(BPE(vocab={token: i for i, token in enumerate(tokenizer.tokens)},
                             merges=merges, unk_token=None, fuse_unk=False))

    cases = (
        ("english-numbers-punctuation", "Version 2026, GPT-4.1."),
        ("japanese-numbers", "今日は2026年の予定です。"),
        ("whitespace", " leading  spaces\tand\nnewlines "),
        ("unicode-punctuation", "“AI” / v2.0 — test!"),
        ("chat-special-tokens", "<｜begin▁of▁sentence｜><｜User｜>今日は2026年？<｜Assistant｜><｜end▁of▁sentence｜>"),
        ("mixed-language", "GPT-4.1 / 2026年: OK? yes!\n"),
    )
    for name, text in cases:
        actual = tokenizer.encode(text, parse_special=True)
        expected = _reference_ids(tokenizer, rust, text, ST.BYTE_TO_UNICODE)
        if actual != expected:
            limit = min(len(actual), len(expected))
            mismatch = next((i for i in range(limit) if actual[i] != expected[i]), limit)
            raise AssertionError("%s: token IDs differ at %d (Strata=%s, Rust=%s)" %
                                 (name, mismatch, actual[mismatch:mismatch + 8], expected[mismatch:mismatch + 8]))
        print("PASS: %-28s %d token IDs match" % (name, len(actual)))

    if metadata.get("tokenizer.ggml.add_bos_token", False):
        print("NOTE: GGUF add_bos_token is true; template check verifies BOS content but not caller policy")
    _check_bos_template(tokenizer, metadata)
    print("BPE oracle: Rust tokenizers over the GGUF vocab/merges; JoyAI regex: llama.cpp 4f92965, not independently checked here.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as exc:
        raise SystemExit(_dependency_skip(exc))
