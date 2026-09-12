"""
Historical backtesting with walk-forward validation.

Honest limitation: public exchange APIs do not expose historical order
book depth or liquidation data, so the order-flow/heatmap confirmation
layer cannot be replayed tick-for-tick against history. The backtester
therefore validates the trend/momentum core of the strategy (the part
that is replayable) on real OHLCV history, and approximates the
liquidity-confirmation layer with a volume-based proxy (recent relative
volume + proximity to rolling swing high/low levels standing in for
heatmap clusters). Live trading (main.py) uses the real order book /
whale-trade / funding-rate data via data_feed.py. This split is called
out explicitly rather than papered over.

Usage:
    python backtester.py --symbol BTC/USDT --exchange binance --years 2
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import get_settings
from strategy import compute_indicators, trend_bias

log = logging.getLogger("trading_bot.backtester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


# --------------------------------------------------------------------------
# Historical data acquisition
# --------------------------------------------------------------------------

def fetch_historical_ohlcv(exchange_id: str, symbol: str, timeframe: str, years: float) -> pd.DataFrame:
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()

    ms_per_candle = exchange.parse_timeframe(timeframe) * 1000
    total_candles_needed = int((years * 365 * 24 * 60 * 60 * 1000) / ms_per_candle)
    since = exchange.milliseconds() - total_candles_needed * ms_per_candle

    all_rows: list[list[float]] = []
    limit = 1000
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + ms_per_candle
        if next_since <= since:
            break
        since = next_since
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000.0)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


# --------------------------------------------------------------------------
# Liquidity-confirmation proxy (see module docstring)
# --------------------------------------------------------------------------

def add_liquidity_proxy(df: pd.DataFrame, lookback: int = 48) -> pd.DataFrame:
    out = df.copy()
    out["swing_high"] = out["high"].rolling(lookback).max()
    out["swing_low"] = out["low"].rolling(lookback).min()
    out["rel_volume"] = out["volume"] / out["volume"].rolling(lookback).mean()
    return out


def confirms_long(row: pd.Series, settings) -> bool:
    near_low = row["swing_low"] > 0 and abs(row["close"] - row["swing_low"]) / row["close"] <= settings.heatmap_proximity_pct * 3
    breakout = row["close"] >= row["swing_high"] * (1 - settings.heatmap_proximity_pct)
    high_participation = row["rel_volume"] >= 1.0
    return bool(high_participation and (near_low or breakout))


def confirms_short(row: pd.Series, settings) -> bool:
    near_high = row["swing_high"] > 0 and abs(row["swing_high"] - row["close"]) / row["close"] <= settings.heatmap_proximity_pct * 3
    breakdown = row["close"] <= row["swing_low"] * (1 + settings.heatmap_proximity_pct)
    high_participation = row["rel_volume"] >= 1.0
    return bool(high_participation and (near_high or breakdown))


# --------------------------------------------------------------------------
# Trade simulation
# --------------------------------------------------------------------------

@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    size: float = 0.0
    pnl: float = 0.0
    r_multiple: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_r_multiple: float = 0.0
    num_trades: int = 0


def run_backtest(df: pd.DataFrame, settings, initial_equity: float, periods_per_year: int) -> BacktestResult:
    """Event-driven simulation over a single (already indicator-enriched,
    already trend-biased) dataframe. `df` must contain the higher-timeframe
    trend bias resampled/forward-filled onto this timeframe's index in a
    column called `trend_bias`."""
    equity = initial_equity
    equity_curve = []
    trades: list[Trade] = []
    open_trade: Trade | None = None
    stop_price = take_profit_price = None
    initial_risk = 0.0

    slippage = settings.slippage_bps / 10_000.0

    for _, row in df.iterrows():
        price = row["close"]

        if open_trade is not None:
            hit_stop = (row["low"] <= stop_price) if open_trade.side == "long" else (row["high"] >= stop_price)
            hit_tp = (row["high"] >= take_profit_price) if open_trade.side == "long" else (row["low"] <= take_profit_price)

            exit_price = None
            exit_reason = ""
            if hit_stop:
                exit_price = stop_price
                exit_reason = "stop"
            elif hit_tp:
                exit_price = take_profit_price
                exit_reason = "take_profit"

            if exit_price is not None:
                fill_price = exit_price * (1 - slippage) if open_trade.side == "long" else exit_price * (1 + slippage)
                direction = 1 if open_trade.side == "long" else -1
                gross_pnl = direction * (fill_price - open_trade.entry_price) * open_trade.size
                fee = (open_trade.entry_price + fill_price) * open_trade.size * settings.taker_fee_pct
                net_pnl = gross_pnl - fee

                equity += net_pnl
                open_trade.exit_time = row.name
                open_trade.exit_price = fill_price
                open_trade.pnl = net_pnl
                open_trade.r_multiple = net_pnl / initial_risk if initial_risk > 0 else 0.0
                open_trade.exit_reason = exit_reason
                trades.append(open_trade)
                open_trade = None

        if open_trade is None:
            bias = row.get("trend_bias", "flat")
            atr_val = row.get("atr", np.nan)
            if bias in ("long", "short") and not np.isnan(atr_val) and atr_val > 0:
                go_long = bias == "long" and confirms_long(row, settings)
                go_short = bias == "short" and confirms_short(row, settings)
                if go_long or go_short:
                    side = "long" if go_long else "short"
                    entry_fill = price * (1 + slippage) if side == "long" else price * (1 - slippage)
                    stop = entry_fill - settings.atr_stop_multiplier * atr_val if side == "long" else entry_fill + settings.atr_stop_multiplier * atr_val
                    stop_distance = abs(entry_fill - stop)
                    tp = entry_fill + settings.risk_reward_ratio * stop_distance if side == "long" else entry_fill - settings.risk_reward_ratio * stop_distance

                    risk_capital = equity * settings.risk_per_trade_pct
                    size = risk_capital / stop_distance if stop_distance > 0 else 0.0
                    max_notional = equity * settings.max_position_pct_of_equity
                    size = min(size, max_notional / entry_fill) if entry_fill > 0 else 0.0

                    if size > 0:
                        entry_fee = entry_fill * size * settings.taker_fee_pct
                        equity -= entry_fee
                        open_trade = Trade(side=side, entry_time=row.name, entry_price=entry_fill, size=size)
                        stop_price, take_profit_price = stop, tp
                        initial_risk = stop_distance * size

        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, index=df.index)
    return _compute_metrics(trades, equity_series, initial_equity, periods_per_year)


def _compute_metrics(trades: list[Trade], equity_curve: pd.Series, initial_equity: float, periods_per_year: int) -> BacktestResult:
    result = BacktestResult(trades=trades, equity_curve=equity_curve, num_trades=len(trades))
    if equity_curve.empty:
        return result

    final_equity = equity_curve.iloc[-1]
    result.total_return_pct = (final_equity / initial_equity - 1) * 100

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        result.sharpe_ratio = float(np.sqrt(periods_per_year) * returns.mean() / returns.std())

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    result.max_drawdown_pct = float(drawdown.min() * 100)

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    result.profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    result.win_rate = float(len(wins) / len(trades)) if trades else 0.0
    result.avg_r_multiple = float(np.mean([t.r_multiple for t in trades])) if trades else 0.0

    return result


# --------------------------------------------------------------------------
# Walk-forward validation
# --------------------------------------------------------------------------

def walk_forward_validate(df_entry: pd.DataFrame, df_trend: pd.DataFrame, settings, n_folds: int, initial_equity: float, periods_per_year: int) -> list[BacktestResult]:
    """Splits the entry-timeframe series chronologically into `n_folds`
    contiguous, non-overlapping out-of-sample windows and backtests each
    independently. Because this strategy has no fitted/optimized parameters
    (fixed indicator windows, not curve-fit thresholds), walk-forward here
    validates *consistency of edge across regimes* rather than train/test
    parameter selection -- a fold that loses badly indicates the strategy
    is regime-dependent, which is exactly what this process is meant to
    surface."""
    df_trend_aligned = compute_indicators(df_trend, settings)
    df_trend_aligned["bias"] = _rolling_trend_bias(df_trend_aligned)

    merged = add_liquidity_proxy(compute_indicators(df_entry, settings))
    merged["trend_bias"] = df_trend_aligned["bias"].reindex(merged.index, method="ffill")
    merged.dropna(subset=["ema_trend", "atr"], inplace=True)

    fold_size = len(merged) // n_folds
    results = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(merged)
        fold_df = merged.iloc[start:end]
        if fold_df.empty:
            continue
        results.append(run_backtest(fold_df, settings, initial_equity, periods_per_year))
    return results


def _rolling_trend_bias(df_trend: pd.DataFrame) -> pd.Series:
    """Vectorized approximation of strategy.trend_bias() applied row-by-row
    for backtesting speed (the live version recomputes on each fresh candle)."""
    bullish = (df_trend["ema_fast"] > df_trend["ema_slow"]) & (df_trend["ema_slow"] > df_trend["ema_trend"]) & (df_trend["macd_hist"] > 0) & (df_trend["macd_hist"].diff() > 0)
    bearish = (df_trend["ema_fast"] < df_trend["ema_slow"]) & (df_trend["ema_slow"] < df_trend["ema_trend"]) & (df_trend["macd_hist"] < 0) & (df_trend["macd_hist"].diff() < 0)
    bias = pd.Series("flat", index=df_trend.index)
    bias[bullish] = "long"
    bias[bearish] = "short"
    return bias


PERIODS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


def _print_report(title: str, result: BacktestResult) -> None:
    print(f"\n=== {title} ===")
    print(f"Trades:            {result.num_trades}")
    print(f"Win rate:          {result.win_rate * 100:.1f}%")
    print(f"Avg R multiple:    {result.avg_r_multiple:.2f}")
    print(f"Total return:      {result.total_return_pct:.2f}%")
    print(f"Sharpe ratio:      {result.sharpe_ratio:.2f}")
    print(f"Max drawdown:      {result.max_drawdown_pct:.2f}%")
    pf = "inf" if result.profit_factor == float("inf") else f"{result.profit_factor:.2f}"
    print(f"Profit factor:     {pf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the hybrid trend/liquidity strategy")
    parser.add_argument("--exchange", default=None, help="ccxt exchange id (default: from .env)")
    parser.add_argument("--symbol", default=None, help="trading pair (default: from .env)")
    parser.add_argument("--years", type=float, default=None, help="years of history to fetch (default: from .env)")
    parser.add_argument("--folds", type=int, default=None, help="walk-forward folds (default: from .env)")
    args = parser.parse_args()

    settings = get_settings()
    exchange_id = args.exchange or settings.exchange_id
    symbol = args.symbol or settings.symbol
    years = args.years or settings.backtest_years
    folds = args.folds or settings.walk_forward_folds

    log.info("Fetching %.1f years of %s %s and %s data from %s...", years, symbol, settings.timeframe_entry, settings.timeframe_trend, exchange_id)
    df_entry = fetch_historical_ohlcv(exchange_id, symbol, settings.timeframe_entry, years)
    df_trend = fetch_historical_ohlcv(exchange_id, symbol, settings.timeframe_trend, years)
    log.info("Fetched %d entry-timeframe candles, %d trend-timeframe candles", len(df_entry), len(df_trend))

    periods_per_year = PERIODS_PER_YEAR.get(settings.timeframe_entry, 24 * 365)

    full_df = add_liquidity_proxy(compute_indicators(df_entry, settings))
    trend_df = compute_indicators(df_trend, settings)
    trend_df["bias"] = _rolling_trend_bias(trend_df)
    full_df["trend_bias"] = trend_df["bias"].reindex(full_df.index, method="ffill")
    full_df.dropna(subset=["ema_trend", "atr"], inplace=True)

    full_result = run_backtest(full_df, settings, settings.backtest_initial_equity, periods_per_year)
    _print_report(f"Full-period backtest ({years:.1f}y)", full_result)

    fold_results = walk_forward_validate(df_entry, df_trend, settings, folds, settings.backtest_initial_equity, periods_per_year)
    for i, r in enumerate(fold_results, start=1):
        _print_report(f"Walk-forward fold {i}/{len(fold_results)}", r)

    if fold_results:
        avg_sharpe = np.mean([r.sharpe_ratio for r in fold_results])
        avg_return = np.mean([r.total_return_pct for r in fold_results])
        profitable_folds = sum(1 for r in fold_results if r.total_return_pct > 0)
        print(f"\n=== Walk-forward summary ===")
        print(f"Profitable folds:  {profitable_folds}/{len(fold_results)}")
        print(f"Avg fold Sharpe:   {avg_sharpe:.2f}")
        print(f"Avg fold return:   {avg_return:.2f}%")


if __name__ == "__main__":
    main()
