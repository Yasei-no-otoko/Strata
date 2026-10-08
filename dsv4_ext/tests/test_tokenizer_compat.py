"""DSV4 JoyAI pre-tokenization stays local to the DSV4 adapter."""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "dsv4_ext" / "tools"))

import regex
import strata_tokenizer as ST
from tokenizer_compat import (
    JOYAI_PRE,
    JoyAIPreTokenizer,
    apply_joyai_pretokenizer,
    patch_tokenizer_class_for_joyai,
)


class JoyAITokenizerCompatTests(unittest.TestCase):
    def test_ordered_passes_split_numbers_and_keep_japanese_runs(self):
        text = "\u4eca\u65e5\u306f2026\u5e74\u306e\u4e88\u5b9a\u3067\u3059\u3002"
        joyai = JoyAIPreTokenizer().findall(text)
        qwen = regex.compile(ST.QWEN35_PATTERN).findall(text)
        self.assertEqual(joyai, [
            "\u4eca\u65e5\u306f", "202", "6", "\u5e74\u306e\u4e88\u5b9a\u3067\u3059", "\u3002",
        ])
        self.assertEqual(qwen, [
            "\u4eca\u65e5\u306f", "2", "0", "2", "6", "\u5e74\u306e\u4e88\u5b9a\u3067\u3059", "\u3002",
        ])

    def test_numeric_pretoken_boundary_changes_bpe_ids(self):
        tokens = list(ST.BYTE_TO_UNICODE.values()) + ["20", "202"]
        merges = ["2 0", "20 2"]
        joyai = ST.Tokenizer(tokens, merges, pre=JOYAI_PRE)
        qwen = ST.Tokenizer(tokens, merges, pre="qwen35")

        self.assertEqual(apply_joyai_pretokenizer(joyai), joyai)
        self.assertEqual(joyai.encode("202"), [joyai.ids["202"]])
        self.assertEqual(qwen.encode("202"), [qwen.ids["2"], qwen.ids["0"], qwen.ids["2"]])

    def test_non_joyai_instance_is_unchanged(self):
        tokenizer = ST.Tokenizer(list(ST.BYTE_TO_UNICODE.values()), [], pre="qwen35")
        original = tokenizer._re
        self.assertIs(apply_joyai_pretokenizer(tokenizer), tokenizer)
        self.assertIs(tokenizer._re, original)

    def test_server_class_patch_handles_constructor_without_pre_argument(self):
        class ServerTokenizer:
            def __init__(self, tokens, merges, token_types=None):
                self.tokens = tokens
                self._re = regex.compile(ST.QWEN35_PATTERN)

        patch_tokenizer_class_for_joyai(ServerTokenizer)
        patch_tokenizer_class_for_joyai(ServerTokenizer)
        tokenizer = ServerTokenizer([], [])
        self.assertEqual(tokenizer.pre, JOYAI_PRE)
        self.assertEqual(tokenizer._re.findall("2026"), ["202", "6"])


if __name__ == "__main__":
    unittest.main()
