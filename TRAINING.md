# Dokumentasi Training Model Chatbot Customer Service

Panduan training model chatbot **customer service Bahasa Indonesia** menggunakan
**Qwen 2.5** dengan teknik **LoRA (Low-Rank Adaptation)**.

---

## 1. Gambaran Umum

Kita melakukan *fine-tuning* (melatih ulang) model bahasa Qwen 2.5 agar mampu
menjawab pertanyaan pelanggan layaknya seorang customer service yang ramah dan
sopan dalam Bahasa Indonesia.

Daripada melatih seluruh bobot model (sangat berat), kita memakai **LoRA**: hanya
sebagian kecil parameter tambahan yang dilatih. Hasilnya jauh lebih ringan, cepat,
dan bisa dijalankan bahkan tanpa GPU.

### Struktur folder

```
project-2/
├─ dataset/
│  └─ customer_service.jsonl     # data latih (tanya-jawab CS)
├─ model/                        # hasil training disimpan di sini
│  ├─ lora_adapter/              # adapter LoRA (ringan, butuh base model)
│  ├─ merged/                    # model gabungan siap pakai (jika pakai --merge)
│  └─ checkpoints/               # checkpoint selama training
├─ training_model.py             # script training utama
├─ chat_test.py                  # script untuk mencoba model hasil training
└─ requirements-training.txt     # daftar dependensi
```

---

## 2. Persyaratan

- **Python** 3.9 – 3.12
- **RAM** minimal 8 GB (disarankan 16 GB)
- **GPU NVIDIA** opsional. Tanpa GPU, training tetap jalan di **CPU**.

> **Catatan soal Intel Iris Xe:** GPU terintegrasi seperti Iris Xe **tidak**
> didukung oleh PyTorch standar di Windows. Training otomatis memakai CPU.
> Itu normal — untuk model 0.5B + dataset kecil masih wajar walau agak lambat.

---

## 3. Instalasi

Dari dalam folder proyek (gunakan virtual environment yang sudah ada, `venv`):

```powershell
# 1. Install PyTorch versi CPU
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install dependensi lainnya
venv\Scripts\pip install -r requirements-training.txt
```

> Jika punya GPU NVIDIA, ganti langkah 1 dengan versi CUDA dari
> <https://pytorch.org>, contoh CUDA 12.1:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## 4. Format Dataset

File: `dataset/customer_service.jsonl` — format **JSON Lines** (satu objek JSON per baris).

Setiap baris punya dua kolom:

| Kolom         | Keterangan                          |
|---------------|-------------------------------------|
| `instruction` | pertanyaan / pesan dari pelanggan   |
| `response`    | jawaban ideal dari customer service |

Contoh:

```json
{"instruction": "Berapa lama waktu pengiriman barang?", "response": "Estimasi pengiriman 1-2 hari kerja untuk Jabodetabek..."}
{"instruction": "Bagaimana cara melacak pesanan saya?", "response": "Anda bisa melacak pesanan di menu 'Pesanan Saya'..."}
```

### Menambah data

Cukup tambahkan baris baru ke file `.jsonl`. **Semakin banyak dan beragam data,
semakin baik hasil model.** Untuk hasil yang bagus, idealnya ratusan – ribuan baris.
Pastikan tetap satu objek JSON per baris (tanpa koma di akhir baris).

---

## 5. Menjalankan Training

Jalankan dengan pengaturan default (model 0.5B, cocok untuk CPU):

```powershell
venv\Scripts\python training_model.py
```

Saat dijalankan, script akan:
1. Memuat base model **Qwen2.5-0.5B-Instruct** (otomatis diunduh dari HuggingFace).
2. Mengubah dataset ke format chat Qwen (system + user + assistant).
3. Melatih adapter LoRA.
4. Menyimpan hasil ke folder `model/`.

### Opsi (argumen) yang tersedia

| Argumen            | Default                       | Keterangan                                              |
|--------------------|-------------------------------|---------------------------------------------------------|
| `--base_model`     | `Qwen/Qwen2.5-0.5B-Instruct`  | Base model dari HuggingFace                              |
| `--dataset_path`   | `dataset/customer_service.jsonl` | Lokasi file dataset                                  |
| `--output_dir`     | `model`                       | Folder output hasil training                            |
| `--epochs`         | `5`                           | Jumlah putaran training                                 |
| `--batch_size`     | `2`                           | Ukuran batch per langkah                                |
| `--grad_accum`     | `4`                           | Gradient accumulation (batch efektif = batch × grad_accum) |
| `--learning_rate`  | `2e-4`                        | Kecepatan belajar                                       |
| `--max_seq_length` | `1024`                        | Panjang maksimum token per contoh                       |
| `--lora_r`         | `16`                          | Rank LoRA (makin besar = makin banyak parameter dilatih) |
| `--lora_alpha`     | `32`                          | Skala LoRA                                              |
| `--lora_dropout`   | `0.05`                        | Dropout LoRA                                            |
| `--use_4bit`       | (off)                         | QLoRA 4-bit — **hanya untuk GPU NVIDIA**                |
| `--merge`          | (off)                         | Gabungkan adapter ke base model jadi model mandiri      |

### Contoh penggunaan

```powershell
# Training lebih lama dengan epoch lebih banyak
venv\Scripts\python training_model.py --epochs 10

# Langsung hasilkan model gabungan siap pakai
venv\Scripts\python training_model.py --merge

# Pakai model lebih besar (butuh GPU NVIDIA + RAM/VRAM besar)
venv\Scripts\python training_model.py --base_model Qwen/Qwen2.5-1.5B-Instruct --use_4bit
```

---

## 6. Hasil Training

Setelah selesai, folder `model/` berisi:

- **`model/lora_adapter/`** — adapter LoRA. Ukurannya kecil (beberapa MB), tetapi
  saat dipakai masih membutuhkan base model Qwen.
- **`model/merged/`** — *(hanya jika pakai `--merge`)* model penuh hasil
  penggabungan adapter + base model. Siap dipakai mandiri tanpa adapter terpisah.
- **`model/checkpoints/`** — checkpoint per epoch selama training.

**Adapter vs Merged — pilih yang mana?**
- `lora_adapter` → hemat ruang, cocok kalau base model sudah ada.
- `merged` → praktis untuk deployment, tapi ukuran file lebih besar (sebesar model penuh).

---

## 7. Mencoba Model

Gunakan `chat_test.py` untuk mengobrol langsung dengan model di terminal:

```powershell
venv\Scripts\python chat_test.py
```

Script otomatis memakai `model/merged` jika ada, atau `model/lora_adapter` di atas
base model bila belum di-merge. Ketik `keluar` untuk berhenti.

Contoh sesi:

```
Anda: Bagaimana cara melacak pesanan saya?
Bot : Anda bisa melacak pesanan dengan masuk ke menu 'Pesanan Saya'...
```

---

## 8. Tips & Troubleshooting

| Masalah                                   | Solusi                                                                 |
|-------------------------------------------|------------------------------------------------------------------------|
| Training sangat lambat                    | Wajar di CPU. Kurangi `--epochs`, atau tetap pakai model 0.5B.         |
| `OutOfMemory` / RAM penuh                 | Kecilkan `--batch_size` ke `1`, kecilkan `--max_seq_length`.           |
| Unduhan model gagal / lambat              | Cek koneksi internet; model diunduh sekali lalu di-cache.             |
| Jawaban model kurang relevan              | Tambah jumlah & variasi data, naikkan `--epochs`.                      |
| Jawaban terlalu mengulang / aneh          | Turunkan `--epochs` (overfitting) atau tambah data.                    |
| `bitsandbytes` error di CPU               | Abaikan `--use_4bit`; flag itu hanya untuk GPU NVIDIA.                 |
| `UnicodeDecodeError: 'charmap' codec...` saat impor `trl` | Bug encoding Windows. Script sudah otomatis pakai mode UTF-8. Jika masih muncul, jalankan: `$env:PYTHONUTF8=1; python training_model.py`. |

### Bagaimana cara kerja LoRA secara singkat?

LoRA menyisipkan matriks kecil tambahan di lapisan-lapisan model (attention & MLP).
Saat training, hanya matriks kecil ini yang diperbarui, sedangkan bobot asli model
dibekukan. Akibatnya jumlah parameter yang dilatih turun drastis (sering < 1%),
sehingga hemat memori dan cepat, tanpa banyak mengorbankan kualitas.

---

## 9. Langkah Selanjutnya

Setelah model jadi, model bisa dihubungkan ke aplikasi Flask (`app.py`) pada
endpoint `/chat_bot/reply` agar chatbot berjalan di antarmuka web.
