# Student Details

Name: Madhav Krishna T

Register Number: TCR24CS043

# Pattern Recognition for Financial Time Series Forecasting

A deep learning pipeline that transforms stock price data into time-frequency spectrograms using STFT and trains a CNN regression model to predict future prices.

> **Assignment 2** — Signal Processing & Deep Learning

---

## Overview

Financial time series are non-stationary signals — their statistical properties change over time. This project treats stock prices as signals, converts them to 2D spectrograms using the Short-Time Fourier Transform (STFT), and feeds those spectrograms into a CNN that learns to predict future prices.

```
Raw prices → Normalise → STFT → Spectrogram (image) → CNN → Predicted price
```

---

## Project Structure

```
stock_forecast/
├── main.py              # Everything — run this
├── requirements.txt     # Python dependencies
├── data/                # Auto-created on first run
│   ├── RELIANCE_NS_raw.csv          <- raw downloaded OHLCV data
│   ├── scaled_prices.csv            <- normalised prices (all tickers)
│   ├── RELIANCE_NS_targets.csv      <- prediction targets (scaled)
│   └── RELIANCE_NS_predictions.csv  <- actual vs predicted (original scale)
└── outputs/             # Auto-created on first run
    ├── figure_1_time_series.png
    ├── figure_2_signal_<TICKER>.png
    ├── figure_3_training_<TICKER>.png
    ├── figure_4_predictions_<TICKER>.png
    └── figure_5_comparison.png
```

---

## Requirements

- Python 3.9 – 3.14
- Internet connection (for first-time data download)

> **Note:** TensorFlow does not support Python 3.14+. This project uses **PyTorch** which works on all recent Python versions.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/stock-forecast.git
cd stock-forecast

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

- On the **first run**, data is downloaded from Yahoo Finance and saved to `data/` as CSV files.
- On **subsequent runs**, the cached CSV files are used — no internet needed.
- All figures are saved to `outputs/` automatically.

---

## Configuration

Edit the `CONFIG` dictionary at the top of `main.py`:

```python
CONFIG = {
    "tickers"      : ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    "start"        : "2020-01-01",
    "end"          : "2024-01-01",
    "window_len"   : 64,     # STFT window length (samples)
    "hop_size"     : 16,     # STFT stride
    "future_steps" : 5,      # days ahead to predict
    "epochs"       : 60,
    "batch_size"   : 32,
    "lr"           : 1e-3,
    "patience"     : 10,
    "data_dir"     : "data",
    "output_dir"   : "outputs",
}
```

### Ticker Symbols (Yahoo Finance)

| Market       | Format      | Example        |
|--------------|-------------|----------------|
| NSE stocks   | `SYMBOL.NS` | `WIPRO.NS`     |
| BSE stocks   | `SYMBOL.BO` | `WIPRO.BO`     |
| BSE Sensex   | `^BSESN`    |                |
| NSE Nifty 50 | `^NSEI`     |                |
| USD–INR      | `USDINR=X`  |                |
| US stocks    | `SYMBOL`    | `AAPL`, `MSFT` |

---

## Output Files

### `data/` folder (CSVs)

| File | Description |
|------|-------------|
| `<TICKER>_raw.csv` | Raw OHLCV data from Yahoo Finance |
| `scaled_prices.csv` | Min-Max normalised closing prices |
| `<TICKER>_targets.csv` | Prediction target values (scaled) |
| `<TICKER>_predictions.csv` | Actual vs predicted prices (original scale) |

### `outputs/` folder (PNGs)

| Figure | Description |
|--------|-------------|
| `figure_1_time_series.png` | Normalised stock prices over time |
| `figure_2_signal_<TICKER>.png` | Time domain + frequency spectrum + STFT spectrogram |
| `figure_3_training_<TICKER>.png` | Training & validation MSE curves |
| `figure_4_predictions_<TICKER>.png` | Actual vs predicted prices |
| `figure_5_comparison.png` | RMSE comparison across all tickers |

---

## Model Architecture

```
Input: Spectrogram (1 × F × T)
  │
  ├─ Conv2D(32,  3×3) → ReLU → MaxPool(2×2)
  ├─ Conv2D(64,  3×3) → ReLU → MaxPool(2×2)
  ├─ Conv2D(128, 3×3) → ReLU → AdaptiveAvgPool(4×4)
  │
  ├─ Flatten → Linear(2048 → 128) → ReLU → Dropout(0.3)
  └─ Linear(128 → 1)   →   p̂(t + Δt)
```

Training uses the Adam optimiser with MSE loss, a ReduceLROnPlateau learning rate scheduler, and early stopping.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MSE  | Mean Squared Error |
| RMSE | Root Mean Squared Error (same unit as price) |
| MAPE | Mean Absolute Percentage Error |

---

## References

1. Y. Zhang and C. Aggarwal, "Stock Market Prediction Using Deep Learning," *IEEE Access*
2. A. Tsantekidis et al., "Deep Learning for Financial Time Series Forecasting"
3. S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997
4. A. Borovykh et al., "Conditional Time Series Forecasting with CNNs"
