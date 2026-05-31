"""Konfigurasi terpusat untuk aplikasi (path, nama model, system prompt, dll)."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Folder model dipisah per jenis: model/chatbot, model/lstm, model/word2vec
CHATBOT_DIR = os.path.join(MODEL_DIR, "chatbot")
LSTM_DIR = os.path.join(MODEL_DIR, "lstm")
W2V_DIR = os.path.join(MODEL_DIR, "word2vec")

# ---- Chatbot (Qwen 2.5 + LoRA) ----
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MERGED_DIR = os.path.join(CHATBOT_DIR, "merged")
ADAPTER_DIR = os.path.join(CHATBOT_DIR, "lora_adapter")
SYSTEM_PROMPT = (
    "Kamu adalah asisten customer service Kurawa, aplikasi HRIS untuk pengelolaan "
    "karyawan (absensi, payroll, cuti, dan data karyawan) dengan website kurawa.id. "
    "Jawab pertanyaan calon pelanggan dalam Bahasa Indonesia yang ramah, sopan, dan jelas, "
    "bantu mereka memahami produk, serta tawarkan bantuan lebih lanjut bila perlu."
)

# ---- Word Vector (Word2Vec) ----
W2V_PATH = os.path.join(W2V_DIR, "word2vec.model")

# ---- LSTM cuaca ----
WEATHER_CSV = os.path.join(DATASET_DIR, "minute_weather.csv")
WEATHER_TARGETS = [
    {"key": "air_temp", "label": "Suhu Udara"},
    {"key": "relative_humidity", "label": "Kelembaban Relatif"},
    {"key": "air_pressure", "label": "Tekanan Udara"},
    {"key": "avg_wind_speed", "label": "Kecepatan Angin (rata-rata)"},
    {"key": "max_wind_speed", "label": "Kecepatan Angin (maks)"},
]
