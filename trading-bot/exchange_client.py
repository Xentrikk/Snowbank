"""
Thin, defensive wrapper around ccxt for exchange connectivity.

Centralizes: sandbox/testnet toggling, retry-with-backoff on transient
network/rate-limit errors, and a minimal surface of methods the rest of
the bot needs (OHLCV, order book, trades, balance, orders).
"""
from __future__ import annotations

import logging
from typing import Any

import ccxt
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import Settings

log = logging.getLogger("trading_bot.exchange")

RETRYABLE_EXCEPTIONS = (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection)


def _retryable():
    return retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )


class ExchangeClient:
    """Wraps a single ccxt exchange instance for one trading pair."""

    def __init__(self, settings: Settings):
        self.settings = settings
        exchange_class = getattr(ccxt, settings.exchange_id)
        self.exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": settings.api_key or None,
                "secret": settings.api_secret or None,
                "password": settings.api_password or None,
                "enableRateLimit": True,
            }
        )
        if settings.use_testnet:
            if not self.exchange.urls.get("test"):
                log.warning(
                    "Exchange %s has no ccxt sandbox endpoint; USE_TESTNET has no effect here.",
                    settings.exchange_id,
                )
            else:
                self.exchange.set_sandbox_mode(True)
                log.info("Sandbox/testnet mode enabled for %s", settings.exchange_id)

        self.exchange.load_markets()

    # --- Market data ---------------------------------------------------------

    @_retryable()
    def fetch_ohlcv(self, timeframe: str, limit: int = 500, since: int | None = None) -> list[list[float]]:
        return self.exchange.fetch_ohlcv(self.settings.symbol, timeframe=timeframe, limit=limit, since=since)

    @_retryable()
    def fetch_order_book(self, limit: int | None = None) -> dict[str, Any]:
        return self.exchange.fetch_order_book(self.settings.symbol, limit=limit or self.settings.order_book_depth)

    @_retryable()
    def fetch_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.exchange.fetch_trades(self.settings.symbol, limit=limit)

    @_retryable()
    def fetch_ticker(self) -> dict[str, Any]:
        return self.exchange.fetch_ticker(self.settings.symbol)

    def fetch_funding_rate(self) -> float | None:
        """Only meaningful for perpetual swap markets; returns None otherwise."""
        if not self.exchange.has.get("fetchFundingRate"):
            return None
        try:
            data = self.exchange.fetch_funding_rate(self.settings.symbol)
            return data.get("fundingRate")
        except Exception as exc:  # noqa: BLE001 - funding rate is best-effort
            log.debug("fetch_funding_rate failed: %s", exc)
            return None

    # --- Account -------------------------------------------------------------

    @_retryable()
    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()

    def free_quote_balance(self) -> float:
        _, quote = self.settings.symbol.split("/")
        balance = self.fetch_balance()
        return float(balance.get(quote, {}).get("free", 0.0) or 0.0)

    def free_base_balance(self) -> float:
        base, _ = self.settings.symbol.split("/")
        balance = self.fetch_balance()
        return float(balance.get(base, {}).get("free", 0.0) or 0.0)

    # --- Orders ---------------------------------------------------------------

    @_retryable()
    def create_market_order(self, side: str, amount: float) -> dict[str, Any]:
        return self.exchange.create_order(self.settings.symbol, type="market", side=side, amount=amount)

    @_retryable()
    def create_stop_loss_order(self, side: str, amount: float, stop_price: float) -> dict[str, Any]:
        """side is the side that CLOSES the position (e.g. 'sell' to exit a long)."""
        params = {"stopPrice": stop_price, "reduceOnly": True}
        try:
            return self.exchange.create_order(self.settings.symbol, type="stop_market", side=side, amount=amount, params=params)
        except ccxt.NotSupported:
            # Fall back to a generic stop order type; exchanges vary widely here.
            return self.exchange.create_order(self.settings.symbol, type="stop", side=side, amount=amount, price=stop_price, params=params)

    @_retryable()
    def create_take_profit_order(self, side: str, amount: float, take_profit_price: float) -> dict[str, Any]:
        params = {"reduceOnly": True}
        return self.exchange.create_order(
            self.settings.symbol, type="limit", side=side, amount=amount, price=take_profit_price, params=params
        )

    @_retryable()
    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self.exchange.cancel_order(order_id, self.settings.symbol)

    @_retryable()
    def fetch_open_orders(self) -> list[dict[str, Any]]:
        return self.exchange.fetch_open_orders(self.settings.symbol)

    def amount_to_precision(self, amount: float) -> float:
        return float(self.exchange.amount_to_precision(self.settings.symbol, amount))

    def price_to_precision(self, price: float) -> float:
        return float(self.exchange.price_to_precision(self.settings.symbol, price))
