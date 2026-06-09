"""
Faithful Python port of two LuxAlgo (open-source) TradingView indicators:

  1. TRAMA  - Trend Regularity Adaptive Moving Average
  2. Trendlines with Breaks

The goal is *replication*, not reinterpretation. Each function below mirrors
the original Pine Script line-for-line where it matters, with the Pine code
quoted in comments so you can audit the translation yourself.

Everything is built on pandas/numpy so the outputs slot straight into a
backtest or a live data feed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pine helpers (re-created so the math matches TradingView exactly)
# ---------------------------------------------------------------------------
def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's moving average (Pine `ta.rma`). Used inside ATR.

    Pine seeds RMA with an SMA of the first `length` values, then applies
    alpha = 1/length. ewm with alpha replicates the recursive part; the small
    warmup difference is irrelevant for a backtest over hundreds of bars.
    """
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Average True Range (Pine `ta.atr`) = RMA of True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return rma(true_range, length)


# ---------------------------------------------------------------------------
# 1. TRAMA - Trend Regularity Adaptive Moving Average
# ---------------------------------------------------------------------------
def trama(df: pd.DataFrame, length: int = 99, src_col: str = "close") -> pd.Series:
    """Trend Regularity Adaptive Moving Average (LuxAlgo).

    Original Pine (v4):
        length = input(99)
        src    = input(close)
        ama = 0.
        hh = max(sign(change(highest(length))), 0)
        ll = max(sign(change(lowest(length)) * -1), 0)
        tc = pow(sma(hh or ll ? 1 : 0, length), 2)
        ama := nz(ama[1] + tc * (src - ama[1]), src)

    Idea: `tc` (the "trend coefficient") is the squared fraction of recent
    bars that printed a fresh rolling high or low. In a strong trend that
    fraction is high so the average tracks price closely; in chop it's low so
    the average flattens out. Squaring penalises low values -> extra smoothing.
    """
    src = df[src_col]
    high, low = df["high"], df["low"]

    # highest(high, length) / lowest(low, length)
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()

    # hh / ll : did we make a NEW rolling high / low this bar?
    hh = np.maximum(np.sign(highest.diff()), 0)
    ll = np.maximum(np.sign(-lowest.diff()), 0)

    # (hh or ll) ? 1 : 0   -> treat NaN warmup as 0
    new_extreme = (((hh > 0) | (ll > 0)).astype(float)).fillna(0.0)

    # tc = pow(sma(new_extreme, length), 2)
    tc = new_extreme.rolling(length).mean() ** 2

    # ama := nz(ama[1] + tc*(src - ama[1]), src)  -- recursive, must loop
    src_v = src.to_numpy(dtype=float)
    tc_v = tc.to_numpy(dtype=float)
    ama = np.empty(len(src_v))
    prev = np.nan
    for i in range(len(src_v)):
        t = tc_v[i]
        if np.isnan(t) or np.isnan(prev):
            prev = src_v[i]              # nz(..., src): fall back to price
        else:
            prev = prev + t * (src_v[i] - prev)
        ama[i] = prev
    return pd.Series(ama, index=df.index, name=f"TRAMA_{length}")


# ---------------------------------------------------------------------------
# 2. Trendlines with Breaks
# ---------------------------------------------------------------------------
def trendlines_with_breaks(
    df: pd.DataFrame,
    length: int = 14,
    k: float = 1.0,
    method: str = "atr",
) -> pd.DataFrame:
    """Pivot-based trendlines with breakout detection (LuxAlgo).

    Original Pine (v5):
        ph = ta.pivothigh(length, length)
        pl = ta.pivotlow(length, length)
        slope = switch method
            'Atr'    => ta.atr(length)/length*k
            'Stdev'  => ta.stdev(src,length)/length*k
            'Linreg' => abs(sma(src*n,length)-sma(src,length)*sma(n,length))
                        /variance(n,length)/2*k
        slope_ph := ph ? slope : slope_ph[1]
        slope_pl := pl ? slope : slope_pl[1]
        upper := ph ? ph : upper[1] - slope_ph
        lower := pl ? pl : lower[1] + slope_pl
        upos := ph ? 0 : close > upper - slope_ph*length ? 1 : upos[1]
        dnos := pl ? 0 : close < lower + slope_pl*length ? 1 : dnos[1]
        upper_break = upos > upos[1]   (fires the bar price clears the line)
        lower_break = dnos > dnos[1]

    Returns a DataFrame with the realtime trendline levels and break flags.

    Note on timing: a pivot is only *confirmed* `length` bars after it forms
    (you need to see the bars on the right). That confirmation lag is built in
    here, so break signals are causal -- no lookahead, safe to backtest.
    """
    method = method.lower()
    high, low, close = df["high"], df["low"], df["close"]
    n = len(df)
    idx = np.arange(n)

    # --- slope series (one value per bar) ---------------------------------
    if method == "atr":
        slope_series = (atr(df, length) / length * k).to_numpy()
    elif method == "stdev":
        slope_series = (close.rolling(length).std(ddof=0) / length * k).to_numpy()
    elif method == "linreg":
        sma_sx = (close * idx).rolling(length).mean()
        sma_s = close.rolling(length).mean()
        sma_x = pd.Series(idx, index=df.index).rolling(length).mean()
        var_x = pd.Series(idx, index=df.index).rolling(length).var(ddof=0)
        slope_series = (
            (sma_sx - sma_s * sma_x).abs() / var_x / 2 * k
        ).to_numpy()
    else:
        raise ValueError("method must be 'atr', 'stdev', or 'linreg'")

    h = high.to_numpy(float)
    l = low.to_numpy(float)
    c = close.to_numpy(float)

    # --- detect pivots, with the length-bar confirmation lag --------------
    # A pivot high sits at bar (i-length); it is confirmed at bar i once the
    # `length` bars to its right are known. ph_at[i] holds that pivot's price.
    ph_at = np.full(n, np.nan)
    pl_at = np.full(n, np.nan)
    for i in range(2 * length, n):
        p = i - length                      # candidate pivot bar
        left, right = slice(p - length, p), slice(p + 1, i + 1)
        if h[p] > h[left].max() and h[p] > h[right].max():
            ph_at[i] = h[p]
        if l[p] < l[left].min() and l[p] < l[right].min():
            pl_at[i] = l[p]

    # --- recursive trendline + breakout state -----------------------------
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    slope_ph = np.zeros(n)
    slope_pl = np.zeros(n)
    upos = np.zeros(n)
    dnos = np.zeros(n)
    upper_break = np.zeros(n, dtype=bool)
    lower_break = np.zeros(n, dtype=bool)

    for i in range(n):
        s = slope_series[i] if not np.isnan(slope_series[i]) else 0.0
        ph = not np.isnan(ph_at[i])
        pl = not np.isnan(pl_at[i])

        slope_ph[i] = s if ph else (slope_ph[i - 1] if i else s)
        slope_pl[i] = s if pl else (slope_pl[i - 1] if i else s)

        prev_up = upper[i - 1] if i and not np.isnan(upper[i - 1]) else np.nan
        prev_lo = lower[i - 1] if i and not np.isnan(lower[i - 1]) else np.nan
        upper[i] = ph_at[i] if ph else (prev_up - slope_ph[i])
        lower[i] = pl_at[i] if pl else (prev_lo + slope_pl[i])

        # realtime line level at this bar = anchor projected forward `length`
        up_line = upper[i] - slope_ph[i] * length
        lo_line = lower[i] + slope_pl[i] * length

        prev_upos = upos[i - 1] if i else 0.0
        prev_dnos = dnos[i - 1] if i else 0.0
        if ph:
            upos[i] = 0.0
        elif not np.isnan(up_line) and c[i] > up_line:
            upos[i] = 1.0
        else:
            upos[i] = prev_upos
        if pl:
            dnos[i] = 0.0
        elif not np.isnan(lo_line) and c[i] < lo_line:
            dnos[i] = 1.0
        else:
            dnos[i] = prev_dnos

        upper_break[i] = upos[i] > prev_upos   # bullish break of resistance
        lower_break[i] = dnos[i] > prev_dnos   # bearish break of support

    return pd.DataFrame(
        {
            "upper_line": upper - slope_ph * length,
            "lower_line": lower + slope_pl * length,
            "upper_break": upper_break,
            "lower_break": lower_break,
        },
        index=df.index,
    )
