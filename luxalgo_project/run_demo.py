"""
End-to-end demo. Run from the project root:

    python run_demo.py                # tries to download GOOGL weekly
    python run_demo.py --csv mydata.csv

It will:
  1. load OHLC data (yfinance if available, else a synthetic series so the
     script always runs),
  2. compute TRAMA + Trendlines with Breaks,
  3. backtest the combined strategy,
  4. print metrics and save a chart that mirrors the TradingView layout.

CSV format expected (if you pass --csv): a Date column + columns named
open/high/low/close (case-insensitive).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from luxalgo.backtest import print_metrics, run_backtest
from luxalgo.strategy import build_signals


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_yfinance(ticker: str, interval: str = "1wk") -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        raw = yf.download(ticker, period="max", interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        return raw[["open", "high", "low", "close"]].dropna()
    except Exception as exc:  # network blocked, etc.
        print(f"[yfinance failed: {exc}]", file=sys.stderr)
        return None


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    return df[["open", "high", "low", "close"]].dropna()


def synthetic(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Fallback data: a trending random walk so the demo always runs."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 1.2, n)
    noise = rng.normal(0, 0.03, n).cumsum()
    close = 100 * np.exp(drift + noise)
    high = close * (1 + np.abs(rng.normal(0, 0.015, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.015, n)))
    open_ = (high + low) / 2
    idx = pd.date_range("2018-01-01", periods=n, freq="W")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=idx)


# ---------------------------------------------------------------------------
# Plot (mimics the TradingView layout: price + TRAMA + trendlines + breaks)
# ---------------------------------------------------------------------------
def plot(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(df.index, df["close"], color="#7d8aa0", lw=1, label="Close")
    ax1.plot(df.index, df["trama"], color="#ff1100", lw=2, label="TRAMA 99")
    ax1.plot(df.index, df["upper_line"], color="#26a69a", lw=0.8, alpha=0.7)
    ax1.plot(df.index, df["lower_line"], color="#ef5350", lw=0.8, alpha=0.7)

    up = df[df["upper_break"]]
    dn = df[df["lower_break"]]
    ax1.scatter(up.index, up["close"], marker="^", color="#26a69a", s=70, label="Bull break", zorder=5)
    ax1.scatter(dn.index, dn["close"], marker="v", color="#ef5350", s=70, label="Bear break", zorder=5)
    ax1.set_title("LuxAlgo replication — TRAMA + Trendlines with Breaks")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.15)

    ax2.plot(df.index, df["equity"], color="#2962ff", lw=1.5, label="Strategy")
    ax2.plot(df.index, df["buy_hold"], color="#999999", lw=1, ls="--", label="Buy & hold")
    ax2.set_ylabel("Growth of $1")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nChart saved to {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="GOOGL")
    ap.add_argument("--interval", default="1wk")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--out", default="luxalgo_chart.png")
    args = ap.parse_args()

    if args.csv:
        df = load_csv(args.csv)
        label = args.csv
    else:
        df = load_yfinance(args.ticker, args.interval)
        label = f"{args.ticker} {args.interval}"
        if df is None:
            print("[no network/yfinance — using synthetic data so the demo runs]")
            df = synthetic()
            label = "SYNTHETIC"

    print(f"Loaded {len(df)} bars ({label})\n")

    signals = build_signals(df)
    bt = run_backtest(signals, signals["position"],
                      fee_bps=args.fee_bps, periods_per_year=52)

    merged = signals.join(bt["results"][["equity", "buy_hold", "strat_ret"]])
    print("Backtest metrics:")
    print_metrics(bt["metrics"])

    plot(merged, args.out)


if __name__ == "__main__":
    main()
