"""
Training model LSTM untuk prediksi cuaca (time-series forecasting).

Dataset : dataset/minute_weather.csv  (data sensor cuaca per menit, ~1,58 juta baris)
Tugas   : memprediksi nilai berikutnya dari sebuah variabel cuaca berdasarkan
          jendela data masa lalu. Bisa melatih satu variabel atau semua sekaligus.

Karena data per-menit sangat besar dan kita berjalan di CPU, data diringkas
(resample) menjadi rata-rata per jam terlebih dahulu agar training tetap cepat.

Cara pakai:
    python training_lstm.py                 # latih SEMUA variabel (default)
    python training_lstm.py --target air_temp
    python training_lstm.py --target relative_humidity --window 48 --epochs 30

Hasil disimpan ke folder model/lstm/ (satu set file per variabel):
    model/lstm/lstm_<target>.pt          -> bobot model + konfigurasi
    model/lstm/lstm_scalers_<target>.joblib  -> scaler fitur & target
    model/lstm/lstm_pred_<target>.png    -> grafik prediksi vs aktual (data uji)
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

import matplotlib
matplotlib.use("Agg")  # backend tanpa GUI (simpan ke file)
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "minute_weather.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model", "lstm")  # hasil LSTM disimpan di sini

# Kolom fitur yang dipakai sebagai input model (sekaligus daftar variabel yang bisa diprediksi)
FEATURES = [
    "air_pressure",
    "air_temp",
    "avg_wind_speed",
    "max_wind_speed",
    "relative_humidity",
]


def parse_args():
    p = argparse.ArgumentParser(description="Training LSTM prediksi cuaca")
    p.add_argument("--target", default="all", choices=["all"] + FEATURES,
                   help="Variabel yang diprediksi ('all' = semua variabel)")
    p.add_argument("--resample", default="1h",
                   help="Aturan resample pandas (mis. 1h = per jam, 30min = per 30 menit)")
    p.add_argument("--window", type=int, default=24,
                   help="Jumlah langkah waktu masa lalu sebagai input (mis. 24 jam)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--test_ratio", type=float, default=0.2)
    return p.parse_args()


def load_data(resample_rule):
    """Muat CSV, ringkas (resample) jadi rata-rata per periode, isi nilai kosong."""
    print(f"[INFO] Membaca data: {CSV_PATH}")
    df = pd.read_csv(
        CSV_PATH,
        usecols=["hpwren_timestamp"] + FEATURES,
        parse_dates=["hpwren_timestamp"],
    )
    print(f"[INFO] Baris mentah  : {len(df):,}")
    df = df.set_index("hpwren_timestamp").sort_index()
    df = df.resample(resample_rule).mean().interpolate(method="time").dropna()
    print(f"[INFO] Baris setelah resample '{resample_rule}': {len(df):,}")
    return df


def make_sequences(features_scaled, target_scaled, window):
    """Sliding window: 'window' langkah masa lalu -> 1 nilai berikutnya."""
    X, y = [], []
    for i in range(window, len(features_scaled)):
        X.append(features_scaled[i - window:i])
        y.append(target_scaled[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_one_target(target, df, feature_scaled, feature_scaler, X, split, args):
    """Latih & simpan model untuk satu variabel target. X dipakai bersama semua target."""
    print(f"\n========== TARGET: {target} ==========")
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(df[[target]].values).ravel()

    # y mengikuti urutan X (mulai dari indeks 'window')
    y = target_scaled[args.window:].astype(np.float32)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeatherLSTM(len(FEATURES), args.hidden_size, args.num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"   Epoch {epoch:3d}/{args.epochs}  loss(MSE)={total / len(X_train):.5f}")

    # Evaluasi
    model.eval()
    with torch.no_grad():
        pred_test = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
    pred_real = target_scaler.inverse_transform(pred_test).ravel()
    y_real = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(np.mean((pred_real - y_real) ** 2)))
    mae = float(np.mean(np.abs(pred_real - y_real)))
    print(f"   [HASIL] RMSE={rmse:.3f} | MAE={mae:.3f}")

    # Simpan model + scaler
    torch.save({
        "state_dict": model.state_dict(),
        "features": FEATURES, "target": target, "window": args.window,
        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        "resample": args.resample,
    }, os.path.join(MODEL_DIR, f"lstm_{target}.pt"))
    joblib.dump(
        {"feature_scaler": feature_scaler, "target_scaler": target_scaler},
        os.path.join(MODEL_DIR, f"lstm_scalers_{target}.joblib"),
    )

    # Grafik
    n_show = min(300, len(y_real))
    plt.figure(figsize=(12, 5))
    plt.plot(y_real[:n_show], label="Aktual", linewidth=1.5)
    plt.plot(pred_real[:n_show], label="Prediksi", linewidth=1.5)
    plt.title(f"Prediksi LSTM '{target}' (RMSE={rmse:.2f}, MAE={mae:.2f})")
    plt.xlabel("Langkah waktu (data uji)"); plt.ylabel(target)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"lstm_pred_{target}.png"), dpi=110)
    plt.close()
    return {"target": target, "rmse": rmse, "mae": mae}


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    targets = FEATURES if args.target == "all" else [args.target]
    print(f"[INFO] Variabel yang dilatih: {targets}")

    df = load_data(args.resample)

    # Scaler fitur & urutan input (X) sama untuk semua target -> dibuat sekali
    feature_scaler = MinMaxScaler()
    feature_scaled = feature_scaler.fit_transform(df[FEATURES].values)
    X, _ = make_sequences(feature_scaled, feature_scaled[:, 0], args.window)
    split = int(len(X) * (1 - args.test_ratio))
    print(f"[INFO] Sampel: {len(X):,} | latih: {split:,} | uji: {len(X) - split:,}")

    hasil = [train_one_target(t, df, feature_scaled, feature_scaler, X, split, args)
             for t in targets]

    print("\n================ RINGKASAN ================")
    for h in hasil:
        print(f"   {h['target']:20s} RMSE={h['rmse']:.3f}  MAE={h['mae']:.3f}")
    print(f"[INFO] Model tersimpan di folder: {MODEL_DIR}")
    print("[INFO] Selesai!")


if __name__ == "__main__":
    main()
