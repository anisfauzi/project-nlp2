"""
Memakai model LSTM hasil training untuk memprediksi nilai cuaca berikutnya.

Mengambil data terakhir dari dataset/minute_weather.csv (sebanyak ukuran window),
lalu memprediksi satu langkah ke depan.

Cara pakai:
    python predict_lstm.py                       # default: air_temp
    python predict_lstm.py --target relative_humidity
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "minute_weather.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model", "lstm")


class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def main():
    parser = argparse.ArgumentParser(description="Prediksi cuaca dengan LSTM")
    parser.add_argument("--target", default="air_temp",
                        help="Variabel yang diprediksi (mis. air_temp, relative_humidity)")
    args = parser.parse_args()

    ckpt_path = os.path.join(MODEL_DIR, f"lstm_{args.target}.pt")
    scaler_path = os.path.join(MODEL_DIR, f"lstm_scalers_{args.target}.joblib")
    if not (os.path.exists(ckpt_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(
            f"Model untuk '{args.target}' belum ada. Jalankan training_lstm.py dulu."
        )

    # ---- Muat model & konfigurasi ----
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    features = ckpt["features"]
    target = ckpt["target"]
    window = ckpt["window"]

    model = WeatherLSTM(len(features), ckpt["hidden_size"], ckpt["num_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scalers = joblib.load(scaler_path)
    feature_scaler = scalers["feature_scaler"]
    target_scaler = scalers["target_scaler"]

    # ---- Ambil data terakhir sebesar window (resample sama seperti saat training) ----
    df = pd.read_csv(CSV_PATH, usecols=["hpwren_timestamp"] + features,
                     parse_dates=["hpwren_timestamp"])
    df = df.set_index("hpwren_timestamp").sort_index()
    df = df.resample(ckpt["resample"]).mean().interpolate(method="time").dropna()

    last_window = df[features].values[-window:]
    waktu_terakhir = df.index[-1]
    nilai_terakhir = df[target].values[-1]

    # ---- Prediksi satu langkah ke depan ----
    x = feature_scaler.transform(last_window).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)  # (1, window, n_features)
    with torch.no_grad():
        pred_scaled = model(x).numpy()
    pred = target_scaler.inverse_transform(pred_scaled).ravel()[0]

    print(f"Variabel diprediksi : {target}")
    print(f"Data terakhir       : {waktu_terakhir}  ->  {nilai_terakhir:.2f}")
    print(f"Prediksi periode berikutnya ({ckpt['resample']}): {pred:.2f}")


if __name__ == "__main__":
    main()
