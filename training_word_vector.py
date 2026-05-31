"""
Training Word Vector (Word2Vec) dengan gensim.

Melatih representasi vektor kata dari korpus berbahasa Indonesia, lalu
menyimpannya ke folder model/. Model ini dipakai oleh halaman web:
    - /word_vector         -> menampilkan vektor sebuah kata
    - /word_vector_search  -> mencari kata-kata yang paling mirip

Cara pakai:
    python training_word_vector.py

Struktur:
    dataset/corpus_indonesia.txt  -> korpus (satu kalimat per baris)
    model/word2vec.model          -> model hasil training
"""

import os
import re

from gensim.models import Word2Vec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "dataset", "corpus_indonesia.txt")
MODEL_PATH = os.path.join(BASE_DIR, "model", "word2vec", "word2vec.model")

# Pola tokenisasi: ambil hanya kata (huruf), buang tanda baca & angka.
_TOKEN = re.compile(r"[a-zA-ZÀ-ÿ]+")


def tokenize(text):
    """Ubah satu kalimat menjadi daftar kata (lowercase). Dipakai juga di app.py."""
    return _TOKEN.findall(text.lower())


def load_corpus(path):
    """Baca korpus, kembalikan daftar kalimat (tiap kalimat = daftar kata)."""
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens = tokenize(line)
            if tokens:
                sentences.append(tokens)
    return sentences


def main():
    print(f"[INFO] Membaca korpus : {CORPUS_PATH}")
    sentences = load_corpus(CORPUS_PATH)
    total_kata = sum(len(s) for s in sentences)
    print(f"[INFO] Jumlah kalimat : {len(sentences)}")
    print(f"[INFO] Jumlah kata    : {total_kata}")

    print("[INFO] Melatih model Word2Vec...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,    # dimensi vektor tiap kata
        window=5,           # jumlah kata tetangga yang dilihat
        min_count=2,        # kata harus muncul minimal 2x agar dipelajari
        sg=1,               # 1 = skip-gram (bagus untuk korpus kecil)
        negative=10,        # negative sampling
        epochs=100,         # banyak epoch karena korpus kecil
        workers=4,
        seed=42,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    print(f"[INFO] Model tersimpan: {MODEL_PATH}")
    print(f"[INFO] Ukuran kosakata: {len(model.wv)} kata")

    # Contoh hasil agar langsung kelihatan bekerja
    print("\n[CONTOH] Kata yang mirip dengan 'kucing':")
    try:
        for kata, skor in model.wv.most_similar("kucing", topn=5):
            print(f"   {kata:15s} {skor:.3f}")
    except KeyError:
        print("   (kata 'kucing' tidak ada di kosakata)")


if __name__ == "__main__":
    main()
