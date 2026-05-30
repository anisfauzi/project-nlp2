import os
import sys

# Paksa mode UTF-8 di Windows (hindari UnicodeDecodeError saat impor library ML)
if sys.platform == "win32" and not sys.flags.utf8_mode:
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import threading

import torch
from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from cjk_filter import build_cjk_suppressor

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi model chatbot
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MERGED_DIR = os.path.join(BASE_DIR, "model", "merged")
ADAPTER_DIR = os.path.join(BASE_DIR, "model", "lora_adapter")

SYSTEM_PROMPT = (
    "Kamu adalah asisten customer service Kurawa, aplikasi HRIS untuk pengelolaan "
    "karyawan (absensi, payroll, cuti, dan data karyawan) dengan website kurawa.id. "
    "Jawab pertanyaan calon pelanggan dalam Bahasa Indonesia yang ramah, sopan, dan jelas, "
    "bantu mereka memahami produk, serta tawarkan bantuan lebih lanjut bila perlu."
)

# Model dimuat sekali saja (lazy: saat permintaan chat pertama datang).
# Lock memastikan generate dijalankan satu per satu (CPU tidak sanggup paralel).
_model = None
_tokenizer = None
_cjk_blocker = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_model():
    """Muat model & tokenizer sekali, lalu cache di memori."""
    global _model, _tokenizer, _cjk_blocker
    if _model is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None:  # cek ulang setelah dapat lock
            return _model, _tokenizer

        if os.path.isdir(MERGED_DIR):
            print(f"[INFO] Memuat model gabungan dari: {MERGED_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_DIR, torch_dtype=torch.float32, trust_remote_code=True
            )
        elif os.path.isdir(ADAPTER_DIR):
            print(f"[INFO] Memuat base model + adapter LoRA dari: {ADAPTER_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        else:
            raise FileNotFoundError(
                "Model belum ada. Jalankan training_model.py terlebih dahulu."
            )

        model.eval()
        _model, _tokenizer = model, tokenizer
        _cjk_blocker = build_cjk_suppressor(tokenizer)  # blokir aksara China/Jepang
        print("[INFO] Model siap.")
        return _model, _tokenizer


def generate_reply(message):
    """Hasilkan balasan chatbot untuk satu pesan pengguna."""
    model, tokenizer = get_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    # Satu generate dalam satu waktu (CPU); permintaan lain menunggu giliran.
    with _infer_lock:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=_cjk_blocker,
            )

    reply = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return reply.strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/word_vector')
def word_vector():
    return render_template("word_vector.html")


@app.route('/chat_bot')
def chat_bot():
    return render_template("chat_bot.html")


@app.route('/chat_bot/reply', methods=['POST'])
def chat_bot_reply():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'reply': 'Silakan ketik pesan terlebih dahulu.'})

    try:
        reply = generate_reply(message)
        if not reply:
            reply = 'Maaf, saya belum punya jawaban untuk itu. Boleh dijelaskan lagi?'
    except FileNotFoundError:
        reply = 'Model belum tersedia. Jalankan training_model.py dulu, ya.'
    except Exception as e:
        print(f"[ERROR] {e}")
        reply = 'Maaf, terjadi kesalahan saat memproses pesan Anda.'

    return jsonify({'reply': reply})


if __name__ == '__main__':
    # use_reloader=False agar model tidak dimuat dua kali (boros RAM) di mode debug.
    app.run(debug=True, use_reloader=False)
