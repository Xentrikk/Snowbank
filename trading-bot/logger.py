"""
Logging and trade-journaling.

Two sinks:
  1. A standard rotating log file + console stream for operational logs
     (signals, state changes, fills, errors).
  2. A JSON-lines trade journal, one record per meaningful event, suitable
     for later analysis/auditing.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def setup_logger(name: str, log_dir: str, log_file: str, level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        # Idempotent: don't double-attach handlers if called more than once.
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


class TradeJournal:
    """Append-only JSONL journal of signals, orders, fills, and errors."""

    def __init__(self, log_dir: str, journal_file: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.path = os.path.join(log_dir, journal_file)

    def _write(self, record: dict[str, Any]) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def signal(self, symbol: str, side: str, confidence: float, reason: str, **extra: Any) -> None:
        self._write({"event": "signal", "symbol": symbol, "side": side, "confidence": confidence, "reason": reason, **extra})

    def order(self, symbol: str, side: str, order_type: str, amount: float, price: float | None, **extra: Any) -> None:
        self._write({"event": "order", "symbol": symbol, "side": side, "order_type": order_type, "amount": amount, "price": price, **extra})

    def fill(self, symbol: str, side: str, amount: float, price: float, fee: float, **extra: Any) -> None:
        self._write({"event": "fill", "symbol": symbol, "side": side, "amount": amount, "price": price, "fee": fee, **extra})

    def state_change(self, description: str, **extra: Any) -> None:
        self._write({"event": "state_change", "description": description, **extra})

    def error(self, message: str, **extra: Any) -> None:
        self._write({"event": "error", "message": message, **extra})

    def circuit_breaker(self, reason: str, **extra: Any) -> None:
        self._write({"event": "circuit_breaker", "reason": reason, **extra})
