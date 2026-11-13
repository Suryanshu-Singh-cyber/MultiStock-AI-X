"""Configuration loader — pydantic-settings + YAML.

Loads configs/config.yaml, configs/symbols.yaml, configs/live.yaml (optional)
and .env secrets. Single source of truth for the whole pipeline.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


class WalkForwardConfig(BaseModel):
    train_window_days: int = 1260
    valid_window_days: int = 126
    test_window_days: int = 21


class TransactionCosts(BaseModel):
    brokerage_per_side: float = 0.0003
    statutory_per_side: float = 0.0005
    slippage_per_side: float = 0.0005

    @property
    def round_trip(self) -> float:
        return 2 * (self.brokerage_per_side + self.statutory_per_side + self.slippage_per_side)


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    project_name: str = "multistock_ai_x"
    target_symbol: str = "POWERGRID.NS"
    forecast_horizon_days: int = 1
    history_start: str = "2005-06-01"
    data_sources_historical: str = "yfinance"


@lru_cache(maxsize=1)
def load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_symbols() -> dict[str, Any]:
    return load_yaml("symbols.yaml")


def get_live_config() -> dict[str, Any]:
    """Live config with ${ENV_VAR:-default} interpolation resolved."""
    raw = load_yaml("live.yaml")
    return _interpolate(raw)


def _interpolate(obj: Any) -> Any:
    import os
    import re

    if isinstance(obj, str):
        return re.sub(
            r"\$\{([^}:]+)(?::-([^}]*))?\}",
            lambda m: os.environ.get(m.group(1), m.group(2) or ""),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v) for v in obj]
    return obj


def get_config() -> dict[str, Any]:
    return load_yaml("config.yaml")


def get_all_symbols() -> list[str]:
    """Every symbol the pipeline needs: target + peers + indices + macro."""
    s = get_symbols()
    syms = [s["target"]]
    syms += s.get("peers", [])
    syms += list(s.get("indices", {}).values())
    syms += list(s.get("macro", {}).values())
    return syms
