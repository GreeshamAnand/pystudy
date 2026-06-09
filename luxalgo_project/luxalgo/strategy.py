"""
A sample strategy that *combines* the two indicators into a position.

This is deliberately simple and easy to tweak -- it's a starting point you can
backtest, not a recommendation. The rules:

  LONG when:
      - price is above TRAMA (trend filter), AND
      - an upper trendline break has fired recently (momentum trigger)
  FLAT (exit) when:
      - price closes back below TRAMA, OR
      - a lower trendline break fires

You can flip `allow_short=True` to mirror the logic on the short side.
"""

from __future__ import annotations

import pandas as pd

from .indicators import trama, trendlines_with_breaks


def build_signals(
    df: pd.DataFrame,
    trama_length: int = 99,
    tl_length: int = 14,
    tl_slope: float = 1.0,
    tl_method: str = "atr",
    break_lookback: int = 3,
    allow_short: bool = False,
) -> pd.DataFrame:
    """Attach indicators + a position column to the price frame."""
    out = df.copy()
    out["trama"] = trama(df, length=trama_length)

    tl = trendlines_with_breaks(df, length=tl_length, k=tl_slope, method=tl_method)
    out = out.join(tl)

    above_trama = out["close"] > out["trama"]
    # "recent" break = any bullish/bearish break in the last `break_lookback` bars
    recent_up = out["upper_break"].rolling(break_lookback, min_periods=1).max().astype(bool)
    recent_dn = out["lower_break"].rolling(break_lookback, min_periods=1).max().astype(bool)

    # Build position with simple state: enter long on trigger, hold while the
    # trend filter holds, exit on filter loss or bearish break.
    pos = []
    state = 0
    for i in range(len(out)):
        long_entry = above_trama.iat[i] and recent_up.iat[i]
        long_exit = (not above_trama.iat[i]) or recent_dn.iat[i]
        short_entry = (not above_trama.iat[i]) and recent_dn.iat[i]
        short_exit = above_trama.iat[i] or recent_up.iat[i]

        if state <= 0 and long_entry:
            state = 1
        elif state == 1 and long_exit:
            state = 0
        if allow_short:
            if state >= 0 and short_entry:
                state = -1
            elif state == -1 and short_exit:
                state = 0
        pos.append(state)

    out["position"] = pd.Series(pos, index=out.index)
    return out
