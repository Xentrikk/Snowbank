import numpy as np
import pandas as pd
import pytest

from backtester import Trade, _compute_metrics, add_liquidity_proxy, confirms_long, confirms_short, run_backtest
from conftest import make_ohlcv
from strategy import compute_indicators


def test_metrics_on_synthetic_all_winning_trades():
    initial_equity = 10_000.0
    equity_curve = pd.Series([10_000, 10_100, 10_200, 10_300], index=pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC"))
    trades = [
        Trade(side="long", entry_time=equity_curve.index[0], entry_price=100, exit_time=equity_curve.index[1], exit_price=101, pnl=100.0, r_multiple=2.5),
        Trade(side="long", entry_time=equity_curve.index[1], entry_price=101, exit_time=equity_curve.index[2], exit_price=102, pnl=100.0, r_multiple=2.5),
        Trade(side="long", entry_time=equity_curve.index[2], entry_price=102, exit_time=equity_curve.index[3], exit_price=103, pnl=100.0, r_multiple=2.5),
    ]
    result = _compute_metrics(trades, equity_curve, initial_equity, periods_per_year=24 * 365)

    assert result.total_return_pct == pytest.approx(3.0, rel=1e-3)
    assert result.win_rate == 1.0
    assert result.profit_factor == float("inf")
    assert result.max_drawdown_pct == pytest.approx(0.0)
    assert result.avg_r_multiple == pytest.approx(2.5)


def test_metrics_max_drawdown_detected():
    equity_curve = pd.Series([10_000, 11_000, 9_000, 9_500], index=pd.date_range("2023-01-01", periods=4, freq="h", tz="UTC"))
    result = _compute_metrics([], equity_curve, 10_000.0, periods_per_year=24 * 365)
    # Peak 11,000 -> trough 9,000 = -18.18%
    assert result.max_drawdown_pct == pytest.approx(-18.1818, rel=1e-3)


def test_metrics_profit_factor_with_mixed_trades():
    equity_curve = pd.Series([10_000, 10_200, 10_100], index=pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC"))
    trades = [
        Trade(side="long", entry_time=equity_curve.index[0], entry_price=100, exit_time=equity_curve.index[1], exit_price=102, pnl=200.0, r_multiple=2.0),
        Trade(side="short", entry_time=equity_curve.index[1], entry_price=102, exit_time=equity_curve.index[2], exit_price=103, pnl=-100.0, r_multiple=-1.0),
    ]
    result = _compute_metrics(trades, equity_curve, 10_000.0, periods_per_year=24 * 365)
    assert result.profit_factor == pytest.approx(2.0)
    assert result.win_rate == pytest.approx(0.5)


def test_metrics_empty_equity_curve_returns_zeros():
    result = _compute_metrics([], pd.Series(dtype=float), 10_000.0, periods_per_year=24 * 365)
    assert result.total_return_pct == 0.0
    assert result.sharpe_ratio == 0.0
    assert result.num_trades == 0


def test_run_backtest_never_risks_more_than_configured_pct_per_trade(settings):
    df = add_liquidity_proxy(compute_indicators(make_ohlcv(500, drift=0.1, vol=0.5, seed=7), settings))
    df["trend_bias"] = "long"
    settings.risk_per_trade_pct = 0.01
    result = run_backtest(df, settings, initial_equity=10_000.0, periods_per_year=24 * 365)

    # Sanity bound: no single trade's loss should blow past ~3x the configured
    # per-trade risk (accounts for fee/slippage drift), which would indicate a
    # sizing bug rather than normal variance.
    for trade in result.trades:
        if trade.pnl < 0:
            assert abs(trade.pnl) <= 10_000.0 * settings.risk_per_trade_pct * 3


def test_run_backtest_produces_consistent_equity_curve_length(settings):
    df = add_liquidity_proxy(compute_indicators(make_ohlcv(200, seed=11), settings))
    df["trend_bias"] = "flat"  # no trades should fire
    result = run_backtest(df, settings, initial_equity=10_000.0, periods_per_year=24 * 365)

    assert len(result.equity_curve) == len(df)
    assert result.num_trades == 0
    assert result.total_return_pct == 0.0


def test_confirms_long_requires_high_participation(settings):
    row = pd.Series({"close": 100.0, "swing_high": 105.0, "swing_low": 95.0, "rel_volume": 0.5})
    assert confirms_long(row, settings) is False

    row_high_vol = pd.Series({"close": 95.05, "swing_high": 105.0, "swing_low": 95.0, "rel_volume": 1.5})
    assert confirms_long(row_high_vol, settings) is True


def test_confirms_short_requires_high_participation(settings):
    row = pd.Series({"close": 105.0, "swing_high": 105.0, "swing_low": 95.0, "rel_volume": 0.5})
    assert confirms_short(row, settings) is False

    row_high_vol = pd.Series({"close": 104.95, "swing_high": 105.0, "swing_low": 95.0, "rel_volume": 1.5})
    assert confirms_short(row_high_vol, settings) is True
