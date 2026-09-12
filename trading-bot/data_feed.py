"""
Alternative-data / market-microstructure layer.

Builds a multi-factor "sentiment and liquidity" snapshot from:
  - order book depth imbalance (bid vs ask volume within N levels)
  - recent large ("whale") trade flow
  - funding rate (perpetual markets only, best-effort)
  - a liquidity heatmap proxy: volume-weighted price clusters built from
    the live order book, standing in for exchange liquidation-cluster
    data (which is not available through public APIs). These clusters
    behave like the "heatmap" support/resistance zones the strategy
    confirms breakouts against.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exchange_client import ExchangeClient

log = logging.getLogger("trading_bot.data_feed")


@dataclass
class LiquidityCluster:
    price: float
    volume: float


@dataclass
class SentimentSnapshot:
    order_book_imbalance: float  # -1 (all asks) .. +1 (all bids)
    whale_flow_bias: float  # -1 (whale selling) .. +1 (whale buying)
    funding_rate: float | None
    liquidity_clusters: list[LiquidityCluster] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Simple weighted blend; funding rate contributes a small contrarian nudge
        (very positive funding => crowded longs => slight bearish tilt)."""
        score = 0.6 * self.order_book_imbalance + 0.4 * self.whale_flow_bias
        if self.funding_rate is not None:
            score -= np.clip(self.funding_rate * 10, -0.2, 0.2)
        return float(np.clip(score, -1.0, 1.0))

    def nearest_cluster_distance_pct(self, price: float) -> float | None:
        if not self.liquidity_clusters:
            return None
        distances = [abs(c.price - price) / price for c in self.liquidity_clusters]
        return min(distances)


def fetch_ohlcv_df(client: ExchangeClient, timeframe: str, limit: int = 500) -> pd.DataFrame:
    raw = client.fetch_ohlcv(timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def compute_order_book_imbalance(order_book: dict, depth: int) -> float:
    bids = order_book.get("bids", [])[:depth]
    asks = order_book.get("asks", [])[:depth]
    bid_vol = sum(qty for _, qty in bids)
    ask_vol = sum(qty for _, qty in asks)
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return float((bid_vol - ask_vol) / total)


def compute_whale_flow_bias(trades: list[dict], usd_threshold: float) -> float:
    """Positive => large buy volume dominates recent tape; negative => large sells dominate."""
    buy_usd = 0.0
    sell_usd = 0.0
    for t in trades:
        notional = float(t.get("price", 0.0)) * float(t.get("amount", 0.0))
        if notional < usd_threshold:
            continue
        if t.get("side") == "buy":
            buy_usd += notional
        elif t.get("side") == "sell":
            sell_usd += notional
    total = buy_usd + sell_usd
    if total <= 0:
        return 0.0
    return float((buy_usd - sell_usd) / total)


def build_liquidity_heatmap(order_book: dict, num_clusters: int = 8) -> list[LiquidityCluster]:
    """Buckets order book levels into `num_clusters` price bands per side and
    returns the highest-volume bands as proxy support/resistance clusters."""
    clusters: list[LiquidityCluster] = []
    for side in ("bids", "asks"):
        levels = order_book.get(side, [])
        if not levels:
            continue
        prices = np.array([p for p, _ in levels])
        volumes = np.array([q for _, q in levels])
        bins = min(num_clusters, len(prices))
        if bins < 1:
            continue
        bin_edges = np.linspace(prices.min(), prices.max(), bins + 1)
        bin_idx = np.clip(np.digitize(prices, bin_edges) - 1, 0, bins - 1)
        for b in range(bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            vol = float(volumes[mask].sum())
            price = float(np.average(prices[mask], weights=volumes[mask]))
            clusters.append(LiquidityCluster(price=price, volume=vol))
    clusters.sort(key=lambda c: c.volume, reverse=True)
    return clusters[: num_clusters * 2]


def build_sentiment_snapshot(client: ExchangeClient, settings) -> SentimentSnapshot:
    order_book = client.fetch_order_book()
    trades = client.fetch_trades(limit=200)
    funding_rate = client.fetch_funding_rate()

    return SentimentSnapshot(
        order_book_imbalance=compute_order_book_imbalance(order_book, settings.order_book_depth),
        whale_flow_bias=compute_whale_flow_bias(trades, settings.whale_trade_usd_threshold),
        funding_rate=funding_rate,
        liquidity_clusters=build_liquidity_heatmap(order_book),
    )
