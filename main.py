"""
main.py
=======
Pattern Recognition for Financial Time Series Forecasting
Assignment 2 - Complete Implementation (PyTorch)

Folder structure created automatically on first run:
    data/       <- downloaded stock data saved as CSV files
    outputs/    <- all figures saved as PNG files

Usage:
    python main.py

To change tickers or settings, edit the CONFIG dict below.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram as scipy_spectrogram
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ─────────────────────────────────────────────
# CONFIG  <-- edit here, nothing else needed
# ─────────────────────────────────────────────

CONFIG = {
    # Yahoo Finance ticker symbols
    # NSE stocks  -> SYMBOL.NS  e.g. WIPRO.NS, INFY.NS
    # BSE stocks  -> SYMBOL.BO
    # Sensex      -> ^BSESN
    # Nifty 50    -> ^NSEI
    # USD-INR     -> USDINR=X
    "tickers"      : ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    "start"        : "2020-01-01",
    "end"          : "2024-01-01",

    # Signal processing
    "window_len"   : 64,    # STFT window length (samples)
    "hop_size"     : 16,    # STFT stride between windows
    "future_steps" : 5,     # how many days ahead to predict

    # Training
    "epochs"       : 60,
    "batch_size"   : 32,
    "lr"           : 1e-3,
    "patience"     : 10,    # early stopping patience

    # Folders
    "data_dir"     : "data",
    "output_dir"   : "outputs",
}


# ─────────────────────────────────────────────
# TASK 1 - Data Collection & Preparation
# ─────────────────────────────────────────────

def collect_data(tickers: list, start: str, end: str, data_dir: str) -> pd.DataFrame:
    """
    Download daily closing prices for each ticker from Yahoo Finance
    and save each ticker's raw data as a CSV in data_dir.
    All tickers are then aligned to a common date index.
    """
    all_close = {}

    for ticker in tickers:
        csv_path = os.path.join(data_dir, f"{ticker.replace('.', '_')}_raw.csv")

        # Use cached CSV if it already exists
        if os.path.exists(csv_path):
            print(f"  Loading from cache: {csv_path}")
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            all_close[ticker] = df["Close"].squeeze()
        else:
            print(f"  Downloading {ticker}...")
            df = yf.download(ticker, start=start, end=end, progress=False)
            df.to_csv(csv_path)
            print(f"  Saved raw data -> {csv_path}")
            all_close[ticker] = df["Close"].squeeze()

    combined = pd.DataFrame(all_close).dropna()
    print(f"\n  Rows  : {len(combined)}")
    print(f"  Range : {combined.index[0].date()} -> {combined.index[-1].date()}")
    return combined


def normalize_data(df: pd.DataFrame, data_dir: str):
    """
    Min-Max scale each ticker to [0, 1].
    Saves the scaled combined DataFrame as data/scaled_prices.csv.

    Returns
    -------
    scaled_df : pd.DataFrame
    scalers   : dict {ticker: MinMaxScaler}
    """
    scalers = {}
    scaled = pd.DataFrame(index=df.index)
    for col in df.columns:
        sc = MinMaxScaler()
        scaled[col] = sc.fit_transform(df[[col]]).flatten()
        scalers[col] = sc

    csv_path = os.path.join(data_dir, "scaled_prices.csv")
    scaled.to_csv(csv_path)
    print(f"  Saved scaled data -> {csv_path}")
    return scaled, scalers


def plot_time_series(df: pd.DataFrame, output_dir: str):
    """Figure 1 - normalised stock prices over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col, linewidth=1)
    ax.set_title("Figure 1 - Normalised stock prices (time series)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "figure_1_time_series.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────
# TASK 2 - Signal Processing
# ─────────────────────────────────────────────

def compute_fft(signal: np.ndarray, fs: float = 1.0):
    """One-sided FFT amplitude spectrum."""
    N = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal)) * 2 / N
    freqs    = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, fft_vals


def compute_spectrogram(signal: np.ndarray, window_len: int = 64, hop_size: int = 16):
    """
    STFT spectrogram via scipy.

    Parameters
    ----------
    signal     : 1-D normalised price array
    window_len : L - samples per window (larger -> better freq resolution)
    hop_size   : H - stride between windows (smaller -> better time resolution)
                 overlap = window_len - hop_size

    Returns
    -------
    freqs  : frequency axis
    times  : time axis (window indices)
    Sxx_dB : 2D spectrogram in dB  shape (freq_bins, time_steps)
    """
    overlap = window_len - hop_size
    f, t, Sxx = scipy_spectrogram(
        signal, fs=1.0,
        nperseg=window_len,
        noverlap=overlap,
        scaling="spectrum",
    )
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_dB


def plot_signal_analysis(signal: np.ndarray, ticker: str, output_dir: str):
    """Figures 2 & 3 - frequency spectrum and STFT spectrogram."""
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(signal, linewidth=0.8, color="steelblue")
    ax1.set_title(f"{ticker} - time domain (normalised)")
    ax1.set_xlabel("Trading days")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    freqs, fft_vals = compute_fft(signal)
    ax2.plot(freqs, fft_vals, color="darkorange", linewidth=0.8)
    ax2.set_title(f"{ticker} - Figure 2: frequency spectrum")
    ax2.set_xlabel("Frequency (cycles / day)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    f, t, Sxx_dB = compute_spectrogram(signal, window_len=64, hop_size=16)
    im = ax3.pcolormesh(t, f, Sxx_dB, shading="gouraud", cmap="viridis")
    fig.colorbar(im, ax=ax3, label="Power (dB)")
    ax3.set_title(f"{ticker} - Figure 3: STFT spectrogram")
    ax3.set_xlabel("Window index")
    ax3.set_ylabel("Frequency (cycles / day)")

    plt.suptitle(f"Signal analysis - {ticker}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f"figure_2_signal_{ticker.replace('.', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved -> {path}")


def build_spectrogram_dataset(
    signal: np.ndarray,
    ticker: str,
    data_dir: str,
    window_len: int   = 64,
    hop_size: int     = 16,
    future_steps: int = 5,
):
    """
    Slide a window across the signal. For each window:
      - compute a spectrogram patch  -> CNN input  (1, F, T)
      - record price future_steps days later       -> target

    Saves the targets as data/<TICKER>_targets.csv for reference.

    Returns
    -------
    X : np.ndarray  (N, 1, freq_bins, time_steps)   channel-first for PyTorch
    y : np.ndarray  (N,)
    """
    X_list, y_list = [], []
    n = len(signal)

    i = 0
    while i + window_len + future_steps <= n:
        seg = signal[i : i + window_len]
        _, _, Sxx_dB = compute_spectrogram(seg, window_len=32, hop_size=8)
        X_list.append(Sxx_dB[np.newaxis, ...])
        y_list.append(signal[i + window_len + future_steps - 1])
        i += hop_size

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.float32)

    # Save targets to CSV for reference
    csv_path = os.path.join(data_dir, f"{ticker.replace('.', '_')}_targets.csv")
    pd.DataFrame({"target_scaled": y}).to_csv(csv_path, index_label="sample")
    print(f"  Saved targets -> {csv_path}")

    return X, y


# ─────────────────────────────────────────────
# TASK 3 - CNN Model & Training
# ─────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """PyTorch Dataset wrapper for (spectrogram, target) pairs."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockPriceCNN(nn.Module):
    """
    CNN regression model: spectrogram image -> future price.

    Architecture
    ------------
    Conv2d(32, 3x3) -> ReLU -> MaxPool(2x2)
    Conv2d(64, 3x3) -> ReLU -> MaxPool(2x2)
    Conv2d(128,3x3) -> ReLU -> AdaptiveAvgPool(4x4)
    Flatten -> Linear(2048->128) -> ReLU -> Dropout(0.3) -> Linear(128->1)
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,          64,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,          128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def train_model(model, train_loader, val_loader, epochs, lr, patience):
    """
    Train with Adam + MSE loss.
    Includes ReduceLROnPlateau scheduler and early stopping.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val, no_improve, best_state = float("inf"), 0, None

    print(f"  Device : {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        val_batch = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_batch.append(criterion(model(xb), yb).item())

        t_loss = float(np.mean(batch_losses))
        v_loss = float(np.mean(val_batch))
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  train={t_loss:.5f}  val={v_loss:.5f}")

        if v_loss < best_val:
            best_val, no_improve = v_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}  (best val={best_val:.5f})")
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def plot_training(train_losses, val_losses, ticker, output_dir):
    """Training & validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train MSE")
    ax.plot(val_losses,   label="val MSE")
    ax.set_title(f"{ticker} - training curves")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"figure_3_training_{ticker.replace('.', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved -> {path}")


# ─────────────────────────────────────────────
# TASK 4 - Evaluation & Analysis
# ─────────────────────────────────────────────

def evaluate(model, test_loader, scaler, ticker, data_dir, output_dir):
    """
    Run inference on the test set, compute metrics, and save:
      - outputs/figure_4_predictions_<TICKER>.png
      - data/<TICKER>_predictions.csv
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.extend(model(xb.to(device)).cpu().numpy().flatten())
            actuals.extend(yb.numpy().flatten())

    preds   = np.array(preds,   dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)

    y_pred   = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    mse  = float(mean_squared_error(y_actual, y_pred))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100)

    print(f"\n  -- Results: {ticker} --")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    # Save predictions to CSV
    csv_path = os.path.join(data_dir, f"{ticker.replace('.', '_')}_predictions.csv")
    pd.DataFrame({"actual": y_actual, "predicted": y_pred}).to_csv(csv_path, index_label="sample")
    print(f"  Saved predictions -> {csv_path}")

    # Figure 4 - actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_actual, label="Actual",    linewidth=1)
    ax.plot(y_pred,   label="Predicted", linewidth=1, linestyle="--")
    ax.set_title(f"{ticker} - Figure 4: Actual vs Predicted  (RMSE={rmse:.4f})")
    ax.set_xlabel("Test sample"); ax.set_ylabel("Price")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"figure_4_predictions_{ticker.replace('.', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved -> {path}")

    return {"ticker": ticker, "mse": mse, "rmse": rmse, "mape": mape}


def compare_results(results: list, output_dir: str):
    """Figure 5 - RMSE bar chart across all tickers."""
    labels    = [r["ticker"] for r in results]
    rmse_vals = [r["rmse"]   for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, rmse_vals, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_title("Figure 5 - RMSE comparison across tickers")
    ax.set_ylabel("RMSE (original price scale)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "figure_5_comparison.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"  Saved -> {path}")

    print("\n  Summary")
    print(f"  {'Ticker':<20} {'MSE':>10} {'RMSE':>10} {'MAPE':>10}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['ticker']:<20} {r['mse']:>10.4f} {r['rmse']:>10.4f} {r['mape']:>9.2f}%")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(cfg: dict):
    # Create folders
    os.makedirs(cfg["data_dir"],   exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── Task 1 ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("TASK 1: Data preparation")
    print("="*55)
    raw_df             = collect_data(cfg["tickers"], cfg["start"], cfg["end"], cfg["data_dir"])
    scaled_df, scalers = normalize_data(raw_df, cfg["data_dir"])
    plot_time_series(scaled_df, cfg["output_dir"])

    results = []

    for ticker in cfg["tickers"]:
        print(f"\n{'='*55}")
        print(f"  Ticker : {ticker}")
        print("="*55)

        signal = scaled_df[ticker].values

        # ── Task 2 ───────────────────────────────────────────────────
        print("\nTASK 2: Signal processing")
        plot_signal_analysis(signal, ticker, cfg["output_dir"])
        X, y = build_spectrogram_dataset(
            signal, ticker, cfg["data_dir"],
            cfg["window_len"], cfg["hop_size"], cfg["future_steps"],
        )
        print(f"  Dataset : X={X.shape}  y={y.shape}")

        # Time-ordered 80/20 split
        split    = int(len(X) * 0.8)
        train_ds = SpectrogramDataset(X[:split], y[:split])
        test_ds  = SpectrogramDataset(X[split:], y[split:])
        val_size = max(1, int(len(train_ds) * 0.15))
        train_ds, val_ds = random_split(
            train_ds, [len(train_ds) - val_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"])
        test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"])

        # ── Task 3 ───────────────────────────────────────────────────
        print("\nTASK 3: Model development")
        model = StockPriceCNN(in_channels=1)
        print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            cfg["epochs"], cfg["lr"], cfg["patience"],
        )
        plot_training(train_losses, val_losses, ticker, cfg["output_dir"])

        # ── Task 4 ───────────────────────────────────────────────────
        print("\nTASK 4: Evaluation")
        metrics = evaluate(model, test_loader, scalers[ticker],
                           ticker, cfg["data_dir"], cfg["output_dir"])
        results.append(metrics)

    compare_results(results, cfg["output_dir"])

    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Data   -> {cfg['data_dir']}/")
    print(f"  Figures -> {cfg['output_dir']}/")
    print("="*55)


if __name__ == "__main__":
    run_pipeline(CONFIG)
