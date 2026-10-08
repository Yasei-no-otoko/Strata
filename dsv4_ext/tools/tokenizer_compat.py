"""DSV4-only tokenizer compatibility for the GGUF joyai-llm pre-tokenizer.

The shared Strata tokenizer implements Qwen3.5's pre-tokenization. llama.cpp assigns
JoyAI's joyai-llm metadata three ordered regex passes (numbers, CJK runs, then the
general pattern), so this adapter replaces only the tokenizer instance's regex
splitter when the DSV4 GGUF declares that pre-tokenizer.
"""
from __future__ import annotations

import regex


JOYAI_PRE = "joyai-llm"

# From ggml-org/llama.cpp src/llama-vocab.cpp, LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM,
# pinned at https://github.com/ggml-org/llama.cpp/blob/4f92965a7bfa9e8eb6519908ff962e23c2cb7b93/src/llama-vocab.cpp
# (regex list at lines 297-305; joyai-llm mapping at lines 2342-2344).
# Preserve the ordered passes: a single alternation can incorrectly absorb the
# optional whitespace prefix into a following CJK match.
JOYAI_PATTERNS = (
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)


class JoyAIPreTokenizer:
    """findall-compatible splitter matching llama.cpp's ordered regex passes."""

    def __init__(self):
        self._patterns = tuple(regex.compile(pattern) for pattern in JOYAI_PATTERNS)

    def findall(self, text: str) -> list[str]:
        pieces = [text] if text else []
        for pattern in self._patterns:
            split: list[str] = []
            for piece in pieces:
                start = 0
                for match in pattern.finditer(piece):
                    if match.start() > start:
                        split.append(piece[start:match.start()])
                    if match.end() > match.start():
                        split.append(match.group(0))
                    start = match.end()
                if start < len(piece):
                    split.append(piece[start:])
            pieces = split
        return pieces


def apply_joyai_pretokenizer(tokenizer):
    """Apply JoyAI's splitter only when tokenizer.pre identifies that GGUF pre-type."""
    if getattr(tokenizer, "pre", None) == JOYAI_PRE:
        tokenizer._re = JoyAIPreTokenizer()
    return tokenizer


def patch_tokenizer_class_for_joyai(tokenizer_class):
    """Force JoyAI splitting for instances built by Strata's DSV4 server.

    The upstream server constructs Tokenizer(tokens, merges, types) from the
    extracted tokenizer directory and does not pass its pre metadata. This
    class-local patch is intended only inside the dedicated DSV4 serving process.
    """
    if getattr(tokenizer_class, "_dsv4_joyai_patched", False):
        return
    original_init = tokenizer_class.__init__

    def init_joyai(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.pre = JOYAI_PRE
        self._re = JoyAIPreTokenizer()

    tokenizer_class.__init__ = init_joyai
    tokenizer_class._dsv4_joyai_patched = True
