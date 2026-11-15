"""Structured logging configuration for Argus ML."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


import re

SENSITIVE_KEYS = {"api_key", "token", "secret", "password", "auth", "credentials", "private_key"}
SENSITIVE_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|auth|credentials)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?"
)


def sanitize_sensitive_data(obj: Any) -> Any:
    """Recursively redact sensitive keys and values from dictionary and list structures."""
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if any(pat in str(k).lower() for pat in SENSITIVE_KEYS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_sensitive_data(v)
        return sanitized
    elif isinstance(obj, (list, tuple)):
        return [sanitize_sensitive_data(item) for item in obj]
    elif isinstance(obj, str):
        return SENSITIVE_PATTERN.sub(r"\1=[REDACTED]", obj)
    return obj


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as structured JSON with sensitive field redaction."""

    def format(self, record: logging.LogRecord) -> str:
        raw_msg = record.getMessage()
        sanitized_msg = SENSITIVE_PATTERN.sub(r"\1=[REDACTED]", raw_msg) if isinstance(raw_msg, str) else raw_msg

        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_msg,
        }

        # Include standard extra fields if present
        for field in ("task_id", "stage", "agent", "tool_name", "duration_ms", "status", "context"):
            if hasattr(record, field):
                val = getattr(record, field)
                log_data[field] = sanitize_sensitive_data(val)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
) -> None:
    """Configure the root logger with the desired level and format."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate log records
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if format_type.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s (task=%(task_id)s stage=%(stage)s): %(message)s",
                defaults={"task_id": "-", "stage": "-"},
            )
        )

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
