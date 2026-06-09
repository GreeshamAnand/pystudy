# LuxAlgo → Python: TRAMA + Trendlines with Breaks

A faithful, auditable Python port of the two open-source LuxAlgo indicators from
your GOOGL weekly chart, wired into a small backtester.

- **TRAMA** — Trend Regularity Adaptive Moving Average (length **99**, source **close**)
- **Trendlines with Breaks** (length **14**, slope **1**, method **ATR**)

Both match the parameters shown in your screenshot.

## What's here

```
luxalgo/
  indicators.py   # the two indicators, ported line-for-line from Pine (commented)
  backtest.py     # transparent vectorized backtester + metrics
  strategy.py     # sample rules combining TRAMA + breaks into a position
run_demo.py       # load data → indicators → backtest → chart
requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python run_demo.py                      # downloads GOOGL weekly (yfinance)
python run_demo.py --ticker AAPL        # any ticker
python run_demo.py --csv my_ohlc.csv    # your own data (Date + open/high/low/close)
```

If there's no internet (or no yfinance), it falls back to synthetic data so the
script always runs. Output: a metrics table + `luxalgo_chart.png` laid out like
the TradingView view (price, red TRAMA, trendlines, break markers, equity curve).

## How faithful is it?

The Pine source for each indicator is quoted in the docstrings of
`indicators.py`, right next to the translation. Key points:

- **TRAMA** reproduces the exact recurrence `ama = ama[1] + tc·(src − ama[1])`,
  where `tc` is the squared rolling fraction of bars making new highs/lows.
  Verified numerically against a manual step (matches to 1e-9).
- **Trendlines** reproduces the pivot detection, the ATR/Stdev/Linreg slope
  options, the recursive `upper/lower` lines, and the `upos/dnos` break logic.
- **No lookahead.** A pivot is only confirmed `length` bars after it forms (you
  need the bars to its right), and that lag is built in. Break signals are
  causal, so the backtest is honest. Positions are also shifted one bar before
  returns are applied.

Minor warmup differences vs TradingView are possible in the first ~`length`
bars (RMA seeding, NaN handling) — irrelevant over a long backtest, but don't
expect bit-identical values on bar 5.

## Should you move to Claude Code?

**Yes, for the agent part.** This chat is a good place to build and verify the
core logic (done). But your end goal — an autonomous trading agent — means many
files, real credentials, scheduled runs, logging, and iterative testing. That's
exactly what Claude Code is for: it works in your actual repo, runs the scripts,
and edits across files. Drop this folder in a repo and continue there.

## The Robinhood "agent" — read this before you build it

A few honest caveats so you don't hit walls:

1. **Robinhood has no official public API for stock trading.** People use
   unofficial libraries like `robin_stocks` (reverse-engineered, can break
   without notice and may bump against Robinhood's terms). Robinhood *does* have
   an official **Crypto** API with API keys, but not for equities.
2. **Separate the brain from the hands.** Keep this signal/strategy code pure
   (in → out), and put any broker calls in a thin, swappable execution layer.
   That way you can paper-trade first and switch brokers later.
3. **Suggested path:** backtest here → forward/paper-test on live data with no
   real orders → only then connect a broker, with hard risk limits (position
   size caps, max daily loss, a kill switch).

I'm not a financial advisor and this isn't financial advice — the sample
strategy is a starting point for testing, not a recommendation to trade.
```
