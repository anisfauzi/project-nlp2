"""
Training script: Fine-tuning Qwen 2.5 untuk Chatbot Customer Service Bahasa Indonesia
menggunakan LoRA (Low-Rank Adaptation).

Struktur folder:
    dataset/customer_service.jsonl   -> data training (instruction + response)
    model/                           -> hasil model (adapter LoRA + model gabungan)

Cara pakai (contoh):
    python training_model.py
    python training_model.py --base_model Qwen/Qwen2.5-1.5B-Instruct --epochs 5 --use_4bit

Catatan:
    - Default memakai Qwen2.5-0.5B-Instruct supaya bisa training di CPU (tanpa GPU NVIDIA).
      GPU terintegrasi seperti Intel Iris Xe TIDAK didukung PyTorch standar -> otomatis pakai CPU.
    - Training di CPU lambat. Untuk dataset kecil (~30 baris) model 0.5B masih wajar
      (perkiraan beberapa menit s/d puluhan menit tergantung CPU & jumlah epoch).
    - Jika nanti punya GPU NVIDIA, bisa naik ke Qwen/Qwen2.5-1.5B / 3B / 7B-Instruct,
      dan aktifkan --use_4bit (QLoRA, butuh paket bitsandbytes + CUDA) untuk hemat VRAM.
"""

import os
import sys
import argparse

# Windows memakai encoding cp1252 secara default, yang membuat impor trl gagal
# (UnicodeDecodeError). Paksa Python berjalan dalam mode UTF-8 dengan me-restart
# interpreter satu kali sebelum library lain di-impor.
if sys.platform == "win32" and not sys.flags.utf8_mode:
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

# Direktori dasar = lokasi file script ini, supaya path selalu konsisten
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# System prompt yang menentukan "kepribadian" chatbot
SYSTEM_PROMPT = (
    "Kamu adalah asisten customer service Kurawa, aplikasi HRIS untuk pengelolaan "
    "karyawan (absensi, payroll, cuti, dan data karyawan) dengan website kurawa.id. "
    "Jawab pertanyaan calon pelanggan dalam Bahasa Indonesia yang ramah, sopan, dan jelas, "
    "bantu mereka memahami produk, serta tawarkan bantuan lebih lanjut bila perlu."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Training LoRA Qwen2.5 - Customer Service Indonesia")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Nama / path base model dari HuggingFace (default 0.5B agar muat di CPU)")
    parser.add_argument("--dataset_path", type=str,
                        default=os.path.join(BASE_DIR, "dataset", "customer_service.jsonl"),
                        help="Path file dataset (.jsonl)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "model"),
                        help="Folder untuk menyimpan hasil model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (efektif batch = batch_size * grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true",
                        help="Gunakan kuantisasi 4-bit (QLoRA) untuk hemat VRAM")
    parser.add_argument("--merge", action="store_true",
                        help="Gabungkan adapter LoRA ke base model setelah training (model siap pakai mandiri)")
    return parser.parse_args()


def build_dataset(dataset_path, tokenizer):
    """Muat dataset .jsonl dan ubah ke format teks dengan chat template Qwen."""
    raw = load_dataset("json", data_files=dataset_path, split="train")

    def format_example(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return raw.map(format_example, remove_columns=raw.column_names)


def main():
    args = parse_args()

    use_gpu = torch.cuda.is_available()
    print(f"[INFO] Base model    : {args.base_model}")
    print(f"[INFO] Dataset       : {args.dataset_path}")
    print(f"[INFO] Output dir    : {args.output_dir}")
    print(f"[INFO] Device        : {'GPU (CUDA)' if use_gpu else 'CPU'}")
    if not use_gpu and args.use_4bit:
        print("[WARN] --use_4bit hanya untuk GPU NVIDIA. Diabaikan, training tetap di CPU.")
        args.use_4bit = False
    if not use_gpu:
        print("[INFO] Training di CPU: gunakan model kecil & sabar, prosesnya lambat.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Konfigurasi kuantisasi (opsional QLoRA) ----------
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ---------- Muat base model ----------
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_gpu else torch.float32,
        device_map="auto" if use_gpu else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # wajib False saat training dengan gradient checkpointing

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ---------- Konfigurasi LoRA ----------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Modul attention + MLP pada arsitektur Qwen2.5
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- Dataset ----------
    train_dataset = build_dataset(args.dataset_path, tokenizer)
    print(f"[INFO] Jumlah data training: {len(train_dataset)}")

    # ---------- Argumen training (SFTConfig = TrainingArguments + opsi khusus SFT) ----------
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_gpu,
        # gradient checkpointing menghemat VRAM di GPU, tapi memperlambat di CPU -> matikan di CPU
        gradient_checkpointing=use_gpu,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        # opsi khusus SFT (di trl versi baru pindah ke sini)
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
    )

    # ---------- Trainer ----------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("[INFO] Mulai training...")
    trainer.train()

    # ---------- Simpan adapter LoRA ----------
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[INFO] Adapter LoRA tersimpan di: {adapter_dir}")

    # ---------- (Opsional) Gabungkan adapter ke base model ----------
    if args.merge:
        print("[INFO] Menggabungkan adapter LoRA ke base model...")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        merged = PeftModel.from_pretrained(base, adapter_dir)
        merged = merged.merge_and_unload()

        merged_dir = os.path.join(args.output_dir, "merged")
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[INFO] Model gabungan (siap pakai) tersimpan di: {merged_dir}")

    print("[INFO] Selesai!")


if __name__ == "__main__":
    main()
    # Paksa proses berhenti & kembali ke prompt. Tanpa ini, thread latar dari
    # PyTorch/tokenizers kadang membuat terminal seolah "menggantung" di Windows
    # padahal training sudah selesai.
    sys.stdout.flush()
    os._exit(0)
