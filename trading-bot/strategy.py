"""
Hybrid trend + liquidity-confirmation strategy.

Design intent (per project spec): avoid curve-fitted, high-win-rate
mean-reversion traps. This strategy trades LESS often, filters hard on
higher-timeframe trend, and only fires when order-flow/liquidity data
agrees with the trend -- targeting a realistic 35-45% win rate with a
1:2.5+ reward:risk, which is positive-expectancy even at a sub-50% hit
rate:

    expectancy = win_rate * RR - (1 - win_rate) * 1
    e.g. 0.40 * 2.5 - 0.60 * 1 = 0.40  (positive)

Two layers:
  1. Trend/momentum filter on the higher timeframe (EMA stack alignment
     + MACD histogram direction) => a directional bias, or "stand aside".
  2. Entry timing on the lower timeframe, confirmed by order-book
     imbalance and proximity to a liquidity heatmap cluster acting as a
     breakout/reclaim level. A trade only fires when both layers agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["long", "short", "flat"]


@dataclass
class Signal:
    side: Side
    confidence: float
    entry_price: float
    stop_price: float | None
    take_profit_price: float | None
    reason: str


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def macd_histogram(series: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Adds EMA/MACD/ATR columns in place-friendly (copy) fashion. Works for
    both the trend and entry timeframe dataframes."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], settings.ema_fast_period)
    out["ema_slow"] = ema(out["close"], settings.ema_slow_period)
    out["ema_trend"] = ema(out["close"], settings.ema_trend_period)
    out["macd_hist"] = macd_histogram(out["close"], settings.macd_fast, settings.macd_slow, settings.macd_signal)
    out["atr"] = atr(out, settings.atr_period)
    return out


def trend_bias(df_trend: pd.DataFrame) -> tuple[Side, float]:
    """Directional bias from the higher timeframe: EMA stack alignment plus
    MACD histogram slope (rising histogram = strengthening momentum)."""
    if len(df_trend) < 3 or df_trend[["ema_fast", "ema_slow", "ema_trend"]].iloc[-1].isna().any():
        return "flat", 0.0

    last = df_trend.iloc[-1]
    hist_now = df_trend["macd_hist"].iloc[-1]
    hist_prev = df_trend["macd_hist"].iloc[-2]
    hist_rising = hist_now > hist_prev

    bullish_stack = last["ema_fast"] > last["ema_slow"] > last["ema_trend"]
    bearish_stack = last["ema_fast"] < last["ema_slow"] < last["ema_trend"]

    if bullish_stack and hist_now > 0 and hist_rising:
        confidence = min(1.0, abs(hist_now) / (last["close"] * 0.001 + 1e-9))
        return "long", float(np.clip(confidence, 0.2, 1.0))
    if bearish_stack and hist_now < 0 and not hist_rising:
        confidence = min(1.0, abs(hist_now) / (last["close"] * 0.001 + 1e-9))
        return "short", float(np.clip(confidence, 0.2, 1.0))
    return "flat", 0.0


def generate_signal(df_trend: pd.DataFrame, df_entry: pd.DataFrame, sentiment, settings) -> Signal:
    """Combines the higher-timeframe bias with lower-timeframe order-flow /
    liquidity confirmation. Returns a `flat` signal whenever the two layers
    disagree or data is insufficient -- standing aside is a valid, and
    expected-frequent, outcome for this strategy."""
    bias, trend_confidence = trend_bias(df_trend)
    if bias == "flat" or len(df_entry) < 2:
        return Signal("flat", 0.0, float(df_entry["close"].iloc[-1]) if len(df_entry) else 0.0, None, None, "no trend bias")

    entry_row = df_entry.iloc[-1]
    price = float(entry_row["close"])
    entry_atr = float(entry_row["atr"]) if not np.isnan(entry_row["atr"]) else None
    if entry_atr is None or entry_atr <= 0:
        return Signal("flat", 0.0, price, None, None, "atr unavailable")

    imbalance = sentiment.order_book_imbalance
    composite = sentiment.composite_score
    cluster_distance = sentiment.nearest_cluster_distance_pct(price)
    near_cluster = cluster_distance is not None and cluster_distance <= settings.heatmap_proximity_pct

    confirms_long = imbalance >= settings.orderbook_imbalance_threshold and composite > 0
    confirms_short = imbalance <= -settings.orderbook_imbalance_threshold and composite < 0

    if bias == "long" and confirms_long:
        stop = price - settings.atr_stop_multiplier * entry_atr
        take_profit = price + settings.risk_reward_ratio * (price - stop)
        confidence = float(np.clip((trend_confidence + abs(composite) + (0.2 if near_cluster else 0.0)) / 2.2, 0.0, 1.0))
        reason = f"bullish trend (conf={trend_confidence:.2f}) confirmed by order-flow (imb={imbalance:.2f}, sentiment={composite:.2f})"
        return Signal("long", confidence, price, stop, take_profit, reason)

    if bias == "short" and confirms_short:
        stop = price + settings.atr_stop_multiplier * entry_atr
        take_profit = price - settings.risk_reward_ratio * (stop - price)
        confidence = float(np.clip((trend_confidence + abs(composite) + (0.2 if near_cluster else 0.0)) / 2.2, 0.0, 1.0))
        reason = f"bearish trend (conf={trend_confidence:.2f}) confirmed by order-flow (imb={imbalance:.2f}, sentiment={composite:.2f})"
        return Signal("short", confidence, price, stop, take_profit, reason)

    return Signal("flat", 0.0, price, None, None, f"trend bias={bias} not confirmed by order-flow (imb={imbalance:.2f})")
