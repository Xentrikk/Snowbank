"""
Portfolio- and trade-level risk controls.

Everything that decides "how much" and "should we even be trading right
now" lives here, deliberately separated from signal generation
(strategy.py) and execution (exchange_client.py / main.py) so risk rules
can be tested and audited in isolation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger("trading_bot.risk")


@dataclass
class Position:
    side: str  # "long" | "short"
    amount: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    initial_stop_distance: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def r_multiple(self, current_price: float) -> float:
        """How many multiples of the initial risk (R) the position currently is."""
        if self.initial_stop_distance <= 0:
            return 0.0
        move = current_price - self.entry_price if self.side == "long" else self.entry_price - current_price
        return move / self.initial_stop_distance


class CircuitBreakerTripped(Exception):
    """Raised when trading must halt for the remainder of the day."""


class RiskManager:
    def __init__(self, settings, journal=None):
        self.settings = settings
        self.journal = journal
        self.day_start_equity: float | None = None
        self.day_key: str | None = None
        self.halted = False
        self.consecutive_errors = 0
        self.open_positions: list[Position] = []

    # --- Daily circuit breaker -------------------------------------------------

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def start_of_day_check(self, current_equity: float) -> None:
        today = self._today_key()
        if self.day_key != today:
            self.day_key = today
            self.day_start_equity = current_equity
            self.halted = False
            log.info("New trading day; equity baseline set to %.2f", current_equity)

    def check_circuit_breaker(self, current_equity: float) -> bool:
        """Returns True (and sets self.halted) if the daily loss limit is breached."""
        self.start_of_day_check(current_equity)
        if self.day_start_equity is None or self.day_start_equity <= 0:
            return False
        drawdown = (self.day_start_equity - current_equity) / self.day_start_equity
        if drawdown >= self.settings.max_daily_drawdown_pct:
            if not self.halted:
                log.error("Circuit breaker tripped: daily drawdown %.2f%% >= limit %.2f%%", drawdown * 100, self.settings.max_daily_drawdown_pct * 100)
                if self.journal:
                    self.journal.circuit_breaker("max_daily_drawdown_exceeded", drawdown=drawdown, equity=current_equity)
            self.halted = True
        return self.halted

    def record_error(self) -> bool:
        """Returns True if consecutive errors have exceeded the safety threshold
        (caller should halt the loop to avoid a runaway failure spiral)."""
        self.consecutive_errors += 1
        return self.consecutive_errors >= self.settings.max_consecutive_errors

    def record_success(self) -> None:
        self.consecutive_errors = 0

    # --- Position sizing -------------------------------------------------------

    def calculate_position_size(self, equity: float, entry_price: float, stop_price: float) -> float:
        """Fractional-risk sizing: never risk more than `risk_per_trade_pct` of
        equity on the distance between entry and stop. Also hard-capped by
        `max_position_pct_of_equity` notional exposure."""
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0 or entry_price <= 0 or equity <= 0:
            return 0.0

        risk_capital = equity * self.settings.risk_per_trade_pct
        size_by_risk = risk_capital / stop_distance

        max_notional = equity * self.settings.max_position_pct_of_equity
        size_by_notional_cap = max_notional / entry_price

        return max(0.0, min(size_by_risk, size_by_notional_cap))

    # --- Pre-trade validation ---------------------------------------------------

    def validate_order(self, free_quote_balance: float, amount: float, price: float) -> tuple[bool, str]:
        if amount <= 0:
            return False, "computed position size is zero or negative"
        if len(self.open_positions) >= self.settings.max_open_positions:
            return False, f"max open positions ({self.settings.max_open_positions}) reached"
        required = amount * price * (1 + self.settings.taker_fee_pct)
        if required > free_quote_balance:
            return False, f"insufficient balance: need {required:.2f}, have {free_quote_balance:.2f}"
        if self.halted:
            return False, "circuit breaker is active"
        return True, "ok"

    # --- Position lifecycle -----------------------------------------------------

    def open_position(self, side: str, amount: float, entry_price: float, stop_price: float, take_profit_price: float) -> Position:
        position = Position(
            side=side,
            amount=amount,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            initial_stop_distance=abs(entry_price - stop_price),
        )
        self.open_positions.append(position)
        return position

    def close_position(self, position: Position) -> None:
        if position in self.open_positions:
            self.open_positions.remove(position)

    def update_trailing_stop(self, position: Position, current_price: float, current_atr: float) -> float:
        """Moves the stop in the trade's favor once price has moved
        `trailing_activation_rr` * R, trailing by `trailing_atr_multiplier` * ATR.
        Never loosens the stop. Returns the (possibly updated) stop price."""
        r = position.r_multiple(current_price)
        if r < self.settings.trailing_activation_rr:
            return position.stop_price

        trail_distance = self.settings.trailing_atr_multiplier * current_atr
        if position.side == "long":
            candidate = current_price - trail_distance
            position.stop_price = max(position.stop_price, candidate)
        else:
            candidate = current_price + trail_distance
            position.stop_price = min(position.stop_price, candidate)
        return position.stop_price
