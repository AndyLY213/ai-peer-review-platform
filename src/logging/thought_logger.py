"""Utilities for capturing and persisting agent thought processes."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_LOG_FILENAME = "thought_logs/events.jsonl"
DEFAULT_ANALYTICS_FILENAME = "analytics/decisions.jsonl"


@dataclass
class ThoughtLogRecord:
    """Represents a single structured log entry."""

    timestamp: str
    event_type: str
    agent_name: str
    agent_role: str
    personality: Optional[str]
    specialty: Optional[str]
    context: Dict[str, Any]
    thought_process: str
    raw_response: Optional[str]


class ThoughtLogger:
    """Thread-safe JSONL logger for agent decision traces."""

    def __init__(self, workspace_path: str = "peer_review_workspace") -> None:
        self._workspace = Path(workspace_path)
        self._log_path = self._workspace / DEFAULT_LOG_FILENAME
        self._analytics_path = self._workspace / DEFAULT_ANALYTICS_FILENAME
        self._lock = threading.Lock()
        self._ensure_destination()

    def _ensure_destination(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_path.exists():
            self._log_path.touch()
        self._analytics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._analytics_path.exists():
            self._analytics_path.touch()

    def log(
        self,
        *,
        event_type: str,
        agent_name: str,
        agent_role: str,
        personality: Optional[str],
        specialty: Optional[str],
        context: Dict[str, Any],
        thought_process: str,
        raw_response: Optional[str] = None,
    ) -> None:
        record = ThoughtLogRecord(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            event_type=event_type,
            agent_name=agent_name,
            agent_role=agent_role,
            personality=personality,
            specialty=specialty,
            context=context,
            thought_process=thought_process,
            raw_response=raw_response,
        )
        payload = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
            with self._analytics_path.open("a", encoding="utf-8") as analytics_handle:
                analytics_handle.write(payload)
                analytics_handle.write("\n")


_LOGGER_SINGLETON: Optional[ThoughtLogger] = None
_LOGGER_LOCK = threading.Lock()


def get_thought_logger(workspace_path: str = "peer_review_workspace") -> ThoughtLogger:
    """Return a shared logger instance."""

    global _LOGGER_SINGLETON
    if _LOGGER_SINGLETON is None:
        with _LOGGER_LOCK:
            if _LOGGER_SINGLETON is None:
                _LOGGER_SINGLETON = ThoughtLogger(workspace_path)
    return _LOGGER_SINGLETON

