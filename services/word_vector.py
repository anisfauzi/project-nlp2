"""Service word vector: muat model Word2Vec dan sediakan lookup & pencarian kata mirip."""

import os
import threading

from config import W2V_PATH
from training_word_vector import tokenize  # tokenisasi sama seperti saat training

_w2v = None
_w2v_lock = threading.Lock()


def get_word_vectors():
    """Muat model Word2Vec sekali. Mengembalikan objek KeyedVectors (.wv)."""
    global _w2v
    if _w2v is not None:
        return _w2v
    with _w2v_lock:
        if _w2v is not None:
            return _w2v
        if not os.path.exists(W2V_PATH):
            raise FileNotFoundError(
                "Model word vector belum ada. Jalankan training_word_vector.py dulu."
            )
        from gensim.models import Word2Vec
        print(f"[INFO] Memuat model Word2Vec dari: {W2V_PATH}")
        _w2v = Word2Vec.load(W2V_PATH).wv
        print(f"[INFO] Word2Vec siap ({len(_w2v)} kata).")
        return _w2v


def first_token(text):
    """Ambil kata pertama (sudah dinormalisasi). None jika kosong."""
    tokens = tokenize(text or "")
    return tokens[0] if tokens else None


def get_vector(word):
    """Kembalikan vektor sebuah kata sebagai list float, atau None jika tak ada."""
    wv = get_word_vectors()
    if word not in wv:
        return None
    vector = wv[word]
    return [round(float(v), 4) for v in vector]


def most_similar(word, topn=10):
    """Kembalikan daftar kata paling mirip, atau None jika kata tak ada."""
    wv = get_word_vectors()
    if word not in wv:
        return None
    return [
        {"word": w, "score": round(float(score), 4)}
        for w, score in wv.most_similar(word, topn=topn)
    ]
