"""
Filter untuk mencegah model mengeluarkan aksara China/Jepang (CJK).

Model Qwen aslinya banyak dilatih dengan bahasa Mandarin, sehingga versi kecil
(0.5B) yang baru di-fine-tune dengan data kecil kadang "bocor" mengeluarkan
karakter seperti 报价. Filter ini mematikan (set -inf) semua token CJK saat
proses generate, sehingga model dipaksa hanya memakai karakter non-CJK
(huruf latin / Bahasa Indonesia).
"""

import re
import torch
from transformers import LogitsProcessor, LogitsProcessorList

# Rentang Unicode: Hiragana/Katakana, CJK Ext-A, CJK Unified, Compatibility, Halfwidth
_CJK = re.compile(r'[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]')


class _SuppressTokens(LogitsProcessor):
    """Set skor token tertentu ke -inf agar tidak pernah terpilih."""

    def __init__(self, token_ids):
        self.token_ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)

    def __call__(self, input_ids, scores):
        scores[:, self.token_ids] = float("-inf")
        return scores


def build_cjk_suppressor(tokenizer):
    """Bangun LogitsProcessorList yang memblokir semua token mengandung aksara CJK.

    Dipanggil sekali saat memuat model (perlu beberapa ratus milidetik).
    Mengembalikan None jika tidak ada token CJK yang ditemukan.
    """
    bad_ids = [i for i in range(len(tokenizer)) if _CJK.search(tokenizer.decode([i]))]
    if not bad_ids:
        return None
    return LogitsProcessorList([_SuppressTokens(bad_ids)])
