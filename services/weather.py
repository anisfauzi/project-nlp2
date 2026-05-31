"""Service LSTM cuaca: muat model per-variabel (di-cache), data dibaca sekali."""

import os
import threading

import numpy as np
import pandas as pd
import torch
import joblib

from config import LSTM_DIR, WEATHER_CSV, WEATHER_TARGETS
from predict_lstm import WeatherLSTM

_VALID_TARGETS = {t["key"] for t in WEATHER_TARGETS}

_weather_df = None          # dataframe terresample (sama untuk semua target)
_lstm_models = {}           # cache: target -> (model, ckpt, scalers)
_lstm_lock = threading.Lock()


def _get_weather_df(ckpt):
    """Baca & ringkas data cuaca sekali saja (lambat ~beberapa detik), lalu cache."""
    global _weather_df
    if _weather_df is None:
        print("[INFO] Menyiapkan data cuaca (resample)... mohon tunggu.")
        df = pd.read_csv(WEATHER_CSV, usecols=["hpwren_timestamp"] + ckpt["features"],
                         parse_dates=["hpwren_timestamp"])
        df = df.set_index("hpwren_timestamp").sort_index()
        _weather_df = df.resample(ckpt["resample"]).mean().interpolate(method="time").dropna()
        print(f"[INFO] Data cuaca siap ({len(_weather_df):,} baris).")
    return _weather_df


def get_lstm(target):
    """Muat model LSTM untuk satu variabel (di-cache). Kembalikan (model, ckpt, scalers, df)."""
    if target not in _VALID_TARGETS:
        raise ValueError(f"Variabel '{target}' tidak dikenal.")
    with _lstm_lock:
        if target not in _lstm_models:
            ckpt_path = os.path.join(LSTM_DIR, f"lstm_{target}.pt")
            scaler_path = os.path.join(LSTM_DIR, f"lstm_scalers_{target}.joblib")
            if not (os.path.exists(ckpt_path) and os.path.exists(scaler_path)):
                raise FileNotFoundError(
                    f"Model untuk '{target}' belum ada. Jalankan training_lstm.py dulu."
                )
            print(f"[INFO] Memuat model LSTM: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = WeatherLSTM(len(ckpt["features"]), ckpt["hidden_size"], ckpt["num_layers"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            _lstm_models[target] = (model, ckpt, joblib.load(scaler_path))
        model, ckpt, scalers = _lstm_models[target]
        df = _get_weather_df(ckpt)
        return model, ckpt, scalers, df


def predict_weather(target="air_temp", history_len=72):
    """Prediksi nilai target 1 langkah ke depan + ambil riwayat untuk grafik."""
    model, ckpt, scalers, df = get_lstm(target)
    features, window = ckpt["features"], ckpt["window"]

    last_window = df[features].values[-window:]
    x = scalers["feature_scaler"].transform(last_window).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(x).numpy()
    pred = float(scalers["target_scaler"].inverse_transform(pred_scaled).ravel()[0])

    tail = df[target].tail(history_len)
    history = [
        {"time": t.strftime("%Y-%m-%d %H:%M"), "value": round(float(v), 2)}
        for t, v in tail.items()
    ]
    step = df.index[-1] - df.index[-2]
    next_time = (df.index[-1] + step).strftime("%Y-%m-%d %H:%M")

    return {
        "target": target,
        "resample": ckpt["resample"],
        "last_time": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        "last_value": round(float(df[target].values[-1]), 2),
        "next_time": next_time,
        "prediction": round(pred, 2),
        "history": history,
    }
