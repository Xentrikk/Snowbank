# BTC/USD Autonomous Trading Bot

A modular, risk-first algorithmic trading bot for BTC against any major USD
stablecoin (`BTC/USDT`, `BTC/USDC`, `BTC/USD`, ...), built on `ccxt`,
`pandas`/`numpy`, and `pydantic`.

## Project layout

```
trading-bot/
├── main.py             # autonomous execution loop
├── config.py           # pydantic-validated settings, loaded from .env
├── exchange_client.py   # ccxt wrapper: market data + order execution, with retries
├── data_feed.py         # order book imbalance, whale-trade flow, liquidity heatmap proxy
├── strategy.py          # hybrid trend + order-flow confirmation signal generator
├── risk_manager.py      # position sizing, circuit breakers, trailing stops
├── backtester.py        # 2yr+ historical walk-forward validation
├── tests/                # pytest suite for risk math + strategy + backtest metrics
├── requirements.txt
└── .env.example
```

## Strategy design

The strategy deliberately avoids the classic curve-fitted "80% win rate"
mean-reversion trap that blows up on the first strong trend day. Instead:

1. **Trend/momentum filter (higher timeframe, default 4h):** an EMA(20/50/200)
   stack alignment combined with MACD histogram direction and slope
   establishes a directional bias, or "stand aside" if the market is choppy.
2. **Order-flow / liquidity confirmation (lower timeframe, default 1h):**
   a trade only fires when the higher-timeframe bias is corroborated by
   *live* order book imbalance, recent whale trade flow, and proximity to a
   liquidity heatmap cluster (see below). Disagreement between the two
   layers means the bot stands aside — most cycles produce no trade, by
   design.
3. **Asymmetric risk/reward:** every trade uses an ATR-based stop and a fixed
   reward:risk target (default 1:2.5). At the target win rate of 35-45%,
   expectancy is solidly positive:

   ```
   expectancy = win_rate * RR - (1 - win_rate)
   0.40 * 2.5 - 0.60 = +0.40   (per R risked)
   ```

### On "heatmap" / liquidation data

Public exchange APIs do not expose historical liquidation clusters, and
free/legal liquidation-heatmap feeds are effectively nonexistent. Two
different approximations are used, and this split is intentional rather than
hidden:

- **Live trading (`data_feed.py`):** builds real-time liquidity clusters from
  the live order book (volume-weighted price bands) plus real order-flow
  signals — order book imbalance, large ("whale") trade detection, and
  funding rate — to approximate where liquidity/interest is concentrated.
- **Backtesting (`backtester.py`):** since historical order book depth isn't
  available either, the confirmation layer is replaced with a volume-based
  proxy (relative volume + proximity to rolling swing high/low levels) so
  the *trend/momentum core* of the strategy can be honestly validated
  against 2+ years of real OHLCV history. Backtest results should be read as
  validating the trend-following core; live performance also benefits from
  the order-flow layer that isn't backtestable with public data.

## Risk management

All enforced in `risk_manager.py`, independent of signal generation:

- **Position sizing:** fractional risk — position size is derived from
  `risk_per_trade_pct` (default 1%) of equity divided by the ATR-based stop
  distance, additionally hard-capped by `max_position_pct_of_equity`.
- **Stops on every trade:** a hard ATR-based stop-loss and RR-based
  take-profit are computed at signal time; in live trading these are placed
  as real exchange orders immediately after entry so they protect the
  position even if the bot process restarts.
- **Trailing take-profit:** once a position reaches `trailing_activation_rr`
  (default 1R) in profit, the stop trails by `trailing_atr_multiplier * ATR`,
  and only ever moves in the trade's favor.
- **Daily circuit breaker:** if realized equity drawdown for the day exceeds
  `max_daily_drawdown_pct` (default 5%), trading halts until the next UTC day.
- **Execution safety:** pre-trade checks for balance sufficiency, max open
  positions, and a consecutive-error counter that halts the loop rather than
  retrying into a runaway failure spiral. All exchange calls retry with
  exponential backoff only on transient network/rate-limit errors.

## Setup

```bash
cd trading-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
- Set `EXCHANGE_ID` to a ccxt-supported exchange (`binance`, `coinbase`,
  `kraken`, ...).
- For testnet/sandbox trading, set `USE_TESTNET=true` and use API keys
  generated on that exchange's testnet/sandbox portal (not your live keys).
  Note: not every exchange's ccxt integration exposes a sandbox endpoint —
  the bot logs a warning and falls back to `DRY_RUN` behavior if so.
- Keep `DRY_RUN=true` while validating end-to-end behavior: the bot computes
  real signals and logs the orders it *would* place, without sending them.
- Set `SYMBOL` to any major USD-stablecoin pair, e.g. `BTC/USDT`, `BTC/USDC`.

**Never commit `.env`** — it's already covered by `.gitignore`.

## Running the backtest

```bash
python backtester.py --exchange binance --symbol BTC/USDT --years 2 --folds 6
```

Fetches 2 years of historical OHLCV (both the entry and trend timeframes),
runs a full-period backtest with realistic maker/taker fees and slippage,
then splits the data into `--folds` contiguous chronological windows and
backtests each independently (walk-forward), so a strategy that only works
in one regime (e.g. only in a strong 2021 bull run) is visible as such
rather than averaged away. Reports for each window and a summary:

- Total return, Sharpe ratio, max drawdown, profit factor, win rate, avg R.

## Running the autonomous loop

```bash
python main.py                 # continuous loop, Ctrl+C to stop gracefully
python main.py --once          # single iteration, useful for testing/cron
python main.py --dry-run       # force paper-trading regardless of .env
```

Every iteration: fetches fresh OHLCV + order book + trades, computes the
signal and sentiment snapshot, checks the daily circuit breaker, sizes and
validates any resulting order against current balance, and (unless
`DRY_RUN`) executes the entry plus stop-loss/take-profit orders. All signals,
orders, fills, and errors are written to `logs/trading_bot.log` (rotating,
console + file) and `logs/trade_journal.jsonl` (structured, one event per
line).

## Tests

```bash
pytest tests/ -v
```

Covers: position-sizing math (risk %, notional caps, zero-stop-distance
edge case), circuit-breaker trip/reset behavior, trailing-stop monotonicity,
EMA/MACD/ATR indicator correctness against reference implementations, trend
bias classification on synthetic trending/choppy data, signal generation
(agreement/disagreement between trend and order-flow layers, RR math), and
backtest metric calculations (Sharpe, max drawdown, profit factor) against
known synthetic equity curves.

## Going live safely

1. Run the backtest and walk-forward validation; do not trade a strategy
   whose walk-forward folds are inconsistent (e.g. profitable in only one
   fold).
2. Run `main.py --dry-run` for at least several days and review the trade
   journal — confirm signal frequency and sizing look sane before risking
   capital.
3. Start on exchange testnet/sandbox with `USE_TESTNET=true`.
4. Move to live trading with small size: start `risk_per_trade_pct` well
   below the 1-2% ceiling, and watch the circuit breaker actually halts
   trading if you simulate a bad day.

## Disclaimer

This is a technical reference implementation, not financial advice. Trading
cryptocurrency is highly risky; past backtest performance does not guarantee
future results. Test thoroughly on testnet before using real funds, and only
risk capital you can afford to lose.
