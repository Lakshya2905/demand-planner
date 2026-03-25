"""Centralized configuration management."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_config_cache: dict | None = None


def get_config() -> dict:
    """Load and cache the YAML configuration."""
    global _config_cache
    if _config_cache is None:
        with open(_CONFIG_PATH, "r") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_nested(key_path: str, default: Any = None) -> Any:
    """
    Retrieve a nested config value using dot notation.

    Example:
        get_nested("forecasting.default_horizon_days")  # returns 30
    """
    cfg = get_config()
    keys = key_path.split(".")
    for key in keys:
        if isinstance(cfg, dict):
            cfg = cfg.get(key, default)
        else:
            return default
    return cfg


def get_api_key() -> str:
    """Return the OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return key


def get_model_name() -> str:
    """Return the configured model name."""
    return os.getenv("OPENAI_MODEL", get_nested("agent.model", "gpt-4o"))
