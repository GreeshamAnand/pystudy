"""
A small, transparent backtester. No hidden magic: it takes a price series and
a target-position series (1 = long, 0 = flat, -1 = short), shifts the position
by one bar to avoid lookahead, and compounds the returns.

Keep it boring and auditable -- that matters more than features when real money
is involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps: float = 0.0,
    periods_per_year: int = 52,   # weekly bars -> 52
) -> dict:
    """Backtest a position series against close-to-close returns.

    position : desired exposure decided using info available *at* each bar.
               It is shifted forward one bar inside this function, so the
               trade is taken on the next bar's open-to-close move. No peeking.
    fee_bps  : round-trip-ish cost charged on every change in position, in
               basis points (10 bps = 0.10%).
    """
    close = df["close"]
    ret = close.pct_change().fillna(0.0)

    # Trade on the bar AFTER the signal -> shift(1). This is the single most
    # important line for honest backtesting.
    pos = position.shift(1).fillna(0.0)

    # Transaction costs whenever exposure changes.
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (fee_bps / 10_000.0)

    strat_ret = pos * ret - cost
    equity = (1 + strat_ret).cumprod()
    buy_hold = (1 + ret).cumprod()

    metrics = _metrics(strat_ret, equity, periods_per_year)
    metrics["n_trades"] = int((pos.diff().abs() > 0).sum())

    results = pd.DataFrame(
        {
            "close": close,
            "ret": ret,
            "position": pos,
            "strat_ret": strat_ret,
            "equity": equity,
            "buy_hold": buy_hold,
        }
    )
    return {"results": results, "metrics": metrics}


def _metrics(strat_ret: pd.Series, equity: pd.Series, ppy: int) -> dict:
    total_return = equity.iloc[-1] - 1
    years = len(strat_ret) / ppy
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    vol = strat_ret.std() * np.sqrt(ppy)
    sharpe = (strat_ret.mean() * ppy) / vol if vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    wins = strat_ret[strat_ret > 0]
    losses = strat_ret[strat_ret < 0]
    win_rate = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) else np.nan

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
    }


def print_metrics(metrics: dict) -> None:
    pct = lambda x: f"{x * 100:6.2f}%"
    print("  Total return : ", pct(metrics["total_return"]))
    print("  CAGR         : ", pct(metrics["cagr"]))
    print("  Ann. vol     : ", pct(metrics["ann_vol"]))
    print("  Sharpe       : ", f"{metrics['sharpe']:6.2f}")
    print("  Max drawdown : ", pct(metrics["max_drawdown"]))
    print("  Win rate     : ", pct(metrics["win_rate"]))
    print("  # trades     : ", metrics["n_trades"])
