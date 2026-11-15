"""Pipeline configuration and constants for Argus ML."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Orchestrator stage retry limits
MAX_RETRIES_PER_STAGE: int = 3


@dataclass(frozen=True)
class Settings:
    """Global configuration settings for Argus."""

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    default_model: str = os.getenv("DEFAULT_MODEL", "gpt-4o")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    max_retries_per_stage: int = MAX_RETRIES_PER_STAGE
    workspace_root: Path = Path(__file__).resolve().parent.parent


settings = Settings()
