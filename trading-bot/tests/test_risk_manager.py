import pytest

from risk_manager import RiskManager


def test_position_size_respects_risk_per_trade_pct(settings):
    rm = RiskManager(settings)
    equity = 10_000.0
    entry, stop = 100.0, 98.0  # $2 stop distance
    size = rm.calculate_position_size(equity, entry, stop)

    risked = size * abs(entry - stop)
    assert risked == pytest.approx(equity * settings.risk_per_trade_pct, rel=1e-6)


def test_position_size_capped_by_max_notional(settings):
    settings.max_position_pct_of_equity = 0.1
    rm = RiskManager(settings)
    equity = 10_000.0
    entry, stop = 100.0, 99.99  # tiny stop distance -> risk-based size would be huge
    size = rm.calculate_position_size(equity, entry, stop)

    assert size * entry <= equity * settings.max_position_pct_of_equity + 1e-6


def test_position_size_zero_when_stop_equals_entry(settings):
    rm = RiskManager(settings)
    assert rm.calculate_position_size(10_000.0, 100.0, 100.0) == 0.0


def test_circuit_breaker_trips_on_daily_drawdown(settings):
    settings.max_daily_drawdown_pct = 0.05
    rm = RiskManager(settings)
    rm.start_of_day_check(10_000.0)

    assert rm.check_circuit_breaker(9_600.0) is False  # 4% down, under limit
    assert rm.check_circuit_breaker(9_400.0) is True  # 6% down, breached
    assert rm.halted is True


def test_circuit_breaker_resets_on_new_day(settings):
    rm = RiskManager(settings)
    rm.day_key = "2020-01-01"
    rm.day_start_equity = 10_000.0
    rm.halted = True

    # start_of_day_check uses the real current date, which will differ from
    # the stubbed day_key above, simulating a day rollover.
    rm.start_of_day_check(9_000.0)

    assert rm.halted is False
    assert rm.day_start_equity == 9_000.0


def test_validate_order_rejects_insufficient_balance(settings):
    rm = RiskManager(settings)
    ok, reason = rm.validate_order(free_quote_balance=10.0, amount=1.0, price=100.0)
    assert ok is False
    assert "insufficient balance" in reason


def test_validate_order_rejects_when_max_positions_reached(settings):
    settings.max_open_positions = 1
    rm = RiskManager(settings)
    rm.open_position("long", 0.1, 100.0, 98.0, 106.0)

    ok, reason = rm.validate_order(free_quote_balance=100_000.0, amount=0.1, price=100.0)
    assert ok is False
    assert "max open positions" in reason


def test_validate_order_rejects_when_halted(settings):
    rm = RiskManager(settings)
    rm.halted = True
    ok, reason = rm.validate_order(free_quote_balance=100_000.0, amount=0.1, price=100.0)
    assert ok is False
    assert "circuit breaker" in reason


def test_trailing_stop_only_moves_in_favorable_direction_long(settings):
    rm = RiskManager(settings)
    settings.trailing_activation_rr = 1.0
    settings.trailing_atr_multiplier = 1.0
    position = rm.open_position("long", 1.0, entry_price=100.0, stop_price=98.0, take_profit_price=105.0)

    # Price hasn't moved enough (R < 1) yet: stop must not move.
    stop = rm.update_trailing_stop(position, current_price=101.0, current_atr=1.0)
    assert stop == 98.0

    # Price moved 1R+ in favor: trailing stop should ratchet up, never down.
    stop = rm.update_trailing_stop(position, current_price=103.0, current_atr=1.0)
    assert stop == pytest.approx(102.0)

    # Price pulls back: stop must not loosen.
    stop = rm.update_trailing_stop(position, current_price=102.0, current_atr=1.0)
    assert stop == pytest.approx(102.0)


def test_expectancy_is_positive_at_target_win_rate_and_rr(settings):
    """Sanity check on the strategy's stated edge assumption: a 35-45% win
    rate combined with >=1:2.5 RR must be net positive expectancy."""
    for win_rate in (0.35, 0.40, 0.45):
        expectancy = win_rate * settings.risk_reward_ratio - (1 - win_rate) * 1
        assert expectancy > 0
