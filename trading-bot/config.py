"""
Centralized, validated configuration for the trading bot.

All operational parameters are loaded from environment variables (via a
`.env` file in development, or real environment variables in production)
and validated through pydantic so the bot fails fast on misconfiguration
instead of misbehaving at 3am with bad numbers.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Exchange connectivity -------------------------------------------------
    exchange_id: str = Field(default="binance", description="ccxt exchange id, e.g. binance/coinbase/kraken")
    api_key: str = Field(default="", description="Exchange API key")
    api_secret: str = Field(default="", description="Exchange API secret")
    api_password: str = Field(default="", description="Exchange API passphrase (Coinbase, KuCoin, etc.)")
    use_testnet: bool = Field(default=True, description="Use exchange sandbox/testnet endpoints")
    dry_run: bool = Field(default=True, description="Paper-trade: compute signals/orders but never send them")

    # --- Market ------------------------------------------------------------
    symbol: str = Field(default="BTC/USDT", description="Trading pair, any major USD stablecoin quote accepted")
    timeframe_trend: str = Field(default="4h", description="Higher timeframe used for trend filtering")
    timeframe_entry: str = Field(default="1h", description="Lower timeframe used for entry timing")
    order_book_depth: int = Field(default=50, ge=5, le=1000)
    whale_trade_usd_threshold: float = Field(default=250_000.0, gt=0)

    # --- Strategy parameters -------------------------------------------------
    ema_fast_period: int = Field(default=20, gt=0)
    ema_slow_period: int = Field(default=50, gt=0)
    ema_trend_period: int = Field(default=200, gt=0)
    macd_fast: int = Field(default=12, gt=0)
    macd_slow: int = Field(default=26, gt=0)
    macd_signal: int = Field(default=9, gt=0)
    atr_period: int = Field(default=14, gt=0)
    orderbook_imbalance_threshold: float = Field(default=0.15, ge=0, le=1)
    heatmap_proximity_pct: float = Field(default=0.004, gt=0, description="How close price must be to a liquidity cluster to count as confirmation")

    # --- Risk management -----------------------------------------------------
    risk_per_trade_pct: float = Field(default=0.01, gt=0, le=0.05, description="Fraction of equity risked per trade")
    atr_stop_multiplier: float = Field(default=1.5, gt=0)
    risk_reward_ratio: float = Field(default=2.5, gt=1.0)
    trailing_activation_rr: float = Field(default=1.0, gt=0, description="RR multiple at which trailing stop activates")
    trailing_atr_multiplier: float = Field(default=1.2, gt=0)
    max_daily_drawdown_pct: float = Field(default=0.05, gt=0, le=0.5)
    max_open_positions: int = Field(default=1, ge=1, le=10)
    max_position_pct_of_equity: float = Field(default=0.5, gt=0, le=1.0, description="Hard cap on notional exposure per position")

    # --- Execution / fees ------------------------------------------------------
    maker_fee_pct: float = Field(default=0.0005, ge=0)
    taker_fee_pct: float = Field(default=0.001, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0, description="Assumed slippage in basis points on market fills")
    loop_interval_seconds: int = Field(default=300, ge=10)
    max_consecutive_errors: int = Field(default=5, ge=1)

    # --- Backtesting -----------------------------------------------------------
    backtest_years: float = Field(default=2.0, gt=0)
    backtest_initial_equity: float = Field(default=10_000.0, gt=0)
    walk_forward_folds: int = Field(default=6, ge=2)

    # --- Logging -----------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")
    log_file: str = Field(default="trading_bot.log")
    trade_journal_file: str = Field(default="trade_journal.jsonl")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError("symbol must be in CCXT 'BASE/QUOTE' form, e.g. BTC/USDT")
        base, quote = v.split("/")
        allowed_quotes = {"USD", "USDT", "USDC", "BUSD", "DAI"}
        if quote.upper() not in allowed_quotes:
            raise ValueError(f"quote currency {quote} not in accepted USD stablecoin set {allowed_quotes}")
        return v.upper()

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level {v}")
        return v


@lru_cache
def get_settings() -> Settings:
    """Settings are cached: env is read once per process."""
    return Settings()
