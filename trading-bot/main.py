"""
Autonomous trading loop.

Wires together: config -> exchange_client -> data_feed -> strategy ->
risk_manager -> execution, running forever (or once, with --once) at
`loop_interval_seconds` cadence. Defensive by design: any single
iteration failure is logged and swallowed (with a consecutive-error
circuit breaker) rather than crashing the process, and every meaningful
event is written to the trade journal.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from config import get_settings
from data_feed import build_sentiment_snapshot, fetch_ohlcv_df
from exchange_client import ExchangeClient
from logger import TradeJournal, setup_logger
from risk_manager import RiskManager
from strategy import compute_indicators, generate_signal

_shutdown_requested = False


def _handle_shutdown_signal(signum, frame) -> None:  # noqa: ANN001 - stdlib signal handler signature
    global _shutdown_requested
    _shutdown_requested = True


def run_iteration(client: ExchangeClient, risk_manager: RiskManager, journal: TradeJournal, settings, log: logging.Logger) -> None:
    df_trend = compute_indicators(fetch_ohlcv_df(client, settings.timeframe_trend, limit=max(300, settings.ema_trend_period * 3)), settings)
    df_entry = compute_indicators(fetch_ohlcv_df(client, settings.timeframe_entry, limit=max(300, settings.atr_period * 5)), settings)
    sentiment = build_sentiment_snapshot(client, settings)

    signal = generate_signal(df_trend, df_entry, sentiment, settings)
    journal.signal(settings.symbol, signal.side, signal.confidence, signal.reason)
    log.info("Signal: side=%s confidence=%.2f reason=%s", signal.side, signal.confidence, signal.reason)

    equity = client.free_quote_balance()
    if risk_manager.check_circuit_breaker(equity):
        log.warning("Circuit breaker active; skipping trade evaluation this cycle.")
        return

    # Manage any existing position: trailing stop + exit checks would be
    # driven by exchange-reported fills/open orders in a full implementation.
    # Here we focus the loop on entries; stop-loss/take-profit are placed as
    # real exchange orders at entry time so they protect the position even
    # if this process restarts.
    if signal.side == "flat" or signal.stop_price is None or signal.take_profit_price is None:
        return

    amount = risk_manager.calculate_position_size(equity, signal.entry_price, signal.stop_price)
    amount = client.amount_to_precision(amount)
    ok, reason = risk_manager.validate_order(equity, amount, signal.entry_price)
    if not ok:
        log.info("Order rejected by risk manager: %s", reason)
        journal.state_change("order_rejected", reason=reason)
        return

    order_side = "buy" if signal.side == "long" else "sell"
    exit_side = "sell" if signal.side == "long" else "buy"

    if settings.dry_run:
        log.info("[DRY RUN] Would %s %.6f %s @ %.2f, stop=%.2f, tp=%.2f", order_side, amount, settings.symbol, signal.entry_price, signal.stop_price, signal.take_profit_price)
        journal.order(settings.symbol, order_side, "market", amount, signal.entry_price, dry_run=True)
        return

    market_order = client.create_market_order(order_side, amount)
    journal.fill(settings.symbol, order_side, amount, float(market_order.get("average") or signal.entry_price), float(market_order.get("fee", {}).get("cost", 0.0) or 0.0))

    stop_price = client.price_to_precision(signal.stop_price)
    take_profit_price = client.price_to_precision(signal.take_profit_price)
    client.create_stop_loss_order(exit_side, amount, stop_price)
    client.create_take_profit_order(exit_side, amount, take_profit_price)

    risk_manager.open_position(signal.side, amount, signal.entry_price, stop_price, take_profit_price)
    log.info("Opened %s position: amount=%.6f entry=%.2f stop=%.2f tp=%.2f", signal.side, amount, signal.entry_price, stop_price, take_profit_price)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous BTC/USD(T) trading loop")
    parser.add_argument("--once", action="store_true", help="run a single iteration and exit (useful for testing/cron)")
    parser.add_argument("--dry-run", action="store_true", help="force dry-run mode regardless of .env setting")
    args = parser.parse_args()

    settings = get_settings()
    if args.dry_run:
        settings.dry_run = True

    log = setup_logger("trading_bot", settings.log_dir, settings.log_file, settings.log_level)
    journal = TradeJournal(settings.log_dir, settings.trade_journal_file)

    log.info("Starting trading bot | symbol=%s exchange=%s testnet=%s dry_run=%s", settings.symbol, settings.exchange_id, settings.use_testnet, settings.dry_run)

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    client = ExchangeClient(settings)
    risk_manager = RiskManager(settings, journal=journal)

    while not _shutdown_requested:
        try:
            run_iteration(client, risk_manager, journal, settings, log)
            risk_manager.record_success()
        except Exception as exc:  # noqa: BLE001 - top-level loop must never crash the process
            log.exception("Iteration failed: %s", exc)
            journal.error(str(exc))
            if risk_manager.record_error():
                log.critical("Too many consecutive errors (%d); halting bot.", risk_manager.consecutive_errors)
                journal.circuit_breaker("max_consecutive_errors_exceeded", count=risk_manager.consecutive_errors)
                break

        if args.once:
            break

        for _ in range(settings.loop_interval_seconds):
            if _shutdown_requested:
                break
            time.sleep(1)

    log.info("Trading bot shutting down.")


if __name__ == "__main__":
    sys.exit(main() or 0)
