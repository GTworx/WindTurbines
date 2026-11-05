"""Structured JSONL logging for the BladeRunner pipeline.

Writes RFC-5424-flavored, one-record-per-line JSON to ``Logs/BladeRunner.log``
and mirrors a human-readable line to stderr so runs are observable in the CLI.
Record shape matches the spec (Section 3.1).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "agent": getattr(record, "agent", None) or "system",
            "level": record.levelname,
            "run_id": getattr(record, "run_id", None) or "-",
            "event": getattr(record, "event", None) or record.msg,
        }
        detail = getattr(record, "detail", None)
        if detail is not None:
            base["detail"] = detail
        return json.dumps(base, ensure_ascii=False, default=str)


class BladeLogger:
    """Thin wrapper that attaches per-record context (agent, run_id, event)."""

    def __init__(self, name: str = "bladerunner"):
        self.logger = logging.getLogger(name)

    @staticmethod
    def setup(log_file: Path, level: str = "INFO") -> None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("bladerunner")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(_JsonlFormatter())
        logger.addHandler(file_handler)

        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        logger.propagate = False

    def _emit(
        self,
        level: int,
        agent: str,
        run_id: Optional[str],
        event: str,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra: Dict[str, Any] = {"agent": agent, "run_id": run_id, "event": event, "detail": detail}
        self.logger.log(level, message, extra=extra)

    def info(self, agent: str, event: str, message: str, detail: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> None:
        self._emit(logging.INFO, agent, run_id, event, message, detail)

    def warning(self, agent: str, event: str, message: str, detail: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> None:
        self._emit(logging.WARNING, agent, run_id, event, message, detail)

    def error(self, agent: str, event: str, message: str, detail: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> None:
        self._emit(logging.ERROR, agent, run_id, event, message, detail)
