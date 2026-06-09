# CLAUDE.md — project context for Claude Code

## What this is
A faithful Python port of two open-source LuxAlgo TradingView indicators, plus a
backtester. Built and verified in a previous session. The end goal is an
automated trading agent (target broker: Robinhood — see caveats below).

- **TRAMA** (Trend Regularity Adaptive Moving Average): length 99, source close
- **Trendlines with Breaks**: length 14, slope 1, method ATR

## Layout
```
luxalgo/indicators.py   # TRAMA + Trendlines, ported line-for-line from Pine (audited)
luxalgo/backtest.py     # vectorized backtester + metrics (Sharpe, CAGR, DD, etc.)
luxalgo/strategy.py     # sample combo rule -> position series
run_demo.py             # load data -> indicators -> backtest -> chart
```

## Commands
```bash
pip install -r requirements.txt
python run_demo.py                    # GOOGL weekly via yfinance
python run_demo.py --ticker AAPL
python run_demo.py --csv data.csv     # Date + open/high/low/close
```

## Invariants — do not break these
- **No lookahead.** Pivots are confirmed `length` bars after they form; the
  backtester shifts positions by one bar. Any new signal must stay causal.
- **TRAMA recurrence** `ama = ama[1] + tc*(src - ama[1])` is verified to 1e-9
  against the Pine source. Don't "simplify" it.
- **Keep strategy logic pure** (data in -> position out). Broker/order code must
  live in a separate execution layer so it's swappable and paper-testable.

## Roadmap / likely next tasks
1. Wire real GOOGL weekly data and reproduce the chart from the screenshot.
2. Parameter sweep + walk-forward validation (avoid overfitting).
3. Paper-trading execution layer (no real orders) on a live data feed.
4. Only then: broker connection with hard risk limits + kill switch.

## Robinhood caveat
No official public API for **stock** trading. Unofficial libs (e.g.
`robin_stocks`) are reverse-engineered and can break / conflict with terms.
Official crypto API exists. Treat any broker integration as untrusted and
sandbox it. This is not financial advice.
