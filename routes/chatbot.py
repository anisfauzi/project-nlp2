"""Route chatbot customer service."""

from flask import Blueprint, request, render_template, jsonify

from services.chatbot import generate_reply

chatbot_bp = Blueprint("chatbot", __name__)


@chatbot_bp.route("/chat_bot")
def chat_bot():
    return render_template("chat_bot.html")


@chatbot_bp.route("/chat_bot/reply", methods=["POST"])
def chat_bot_reply():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Silakan ketik pesan terlebih dahulu."})

    try:
        reply = generate_reply(message)
        if not reply:
            reply = "Maaf, saya belum punya jawaban untuk itu. Boleh dijelaskan lagi?"
    except FileNotFoundError:
        reply = "Model belum tersedia. Jalankan training_model.py dulu, ya."
    except Exception as e:
        print(f"[ERROR] {e}")
        reply = "Maaf, terjadi kesalahan saat memproses pesan Anda."

    return jsonify({"reply": reply})
