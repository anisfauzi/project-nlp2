"""Route prediksi cuaca dengan LSTM (mendukung pilihan variabel)."""

from flask import Blueprint, request, render_template, jsonify

from config import WEATHER_TARGETS
from services.weather import predict_weather

weather_bp = Blueprint("weather", __name__)


@weather_bp.route("/weather_lstm")
def weather_lstm():
    return render_template("weather_lstm.html", targets=WEATHER_TARGETS)


@weather_bp.route("/weather_lstm/predict", methods=["POST"])
def weather_lstm_predict():
    data = request.get_json(silent=True) or {}
    target = data.get("target") or "air_temp"
    try:
        result = predict_weather(target)
        result["found"] = True
        return jsonify(result)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"found": False, "message": str(e)})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"found": False, "message": "Terjadi kesalahan saat memprediksi."})
