"""Resolve task complexity hints to configured model tiers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


Tier = Literal["T1", "T2", "T3"]
DEFAULT_CONFIG = Path(__file__).with_name("worker_tiers.yaml")
COMPLEXITY_TIERS: dict[int, Tier] = {
    1: "T1",
    2: "T1",
    3: "T2",
    4: "T2",
    5: "T3",
}


class TierProfile(BaseModel):
    model: str = Field(min_length=1)


class TierConfiguration(BaseModel):
    version: int = Field(ge=1)
    tiers: dict[Tier, TierProfile]


def resolve_task_tier(complexity: int, tier_hint: Tier | None = None) -> Tier:
    if tier_hint is not None:
        return tier_hint
    try:
        return COMPLEXITY_TIERS[complexity]
    except KeyError as error:
        raise ValueError("Task complexity must be between 1 and 5") from error


@lru_cache(maxsize=8)
def _load_configuration(path: str, modified_ns: int) -> TierConfiguration:
    del modified_ns
    with open(path, encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    return TierConfiguration.model_validate(raw)


def load_tier_configuration(path: str | os.PathLike[str] | None = None) -> TierConfiguration:
    config_path = Path(
        path or os.environ.get("AGENT_WORKER_TIERS_CONFIG") or DEFAULT_CONFIG
    ).expanduser().resolve()
    try:
        modified_ns = config_path.stat().st_mtime_ns
    except OSError as error:
        raise ValueError(f"Cannot read worker tier configuration: {config_path}") from error
    return _load_configuration(str(config_path), modified_ns)


def resolve_task_model(
    complexity: int,
    tier_hint: Tier | None = None,
    *,
    active_model: str | None = None,
) -> str:
    tier = resolve_task_tier(complexity, tier_hint)
    try:
        configured_model = load_tier_configuration().tiers[tier].model
    except KeyError as error:
        raise ValueError(f"Worker tier {tier} is missing from configuration") from error
    if configured_model == "ACTIVE_MODEL":
        if active_model:
            return active_model
        from api import get_active_model

        return get_active_model()
    return configured_model


def resolve_next_task_model(
    current_tier: Tier,
    current_model: str,
    *,
    active_model: str | None = None,
) -> tuple[Tier, str] | None:
    """Return the next stronger configured tier with a distinct model."""
    order: tuple[Tier, ...] = ("T1", "T2", "T3")
    current_index = order.index(current_tier)
    for tier in order[current_index + 1:]:
        try:
            candidate = resolve_task_model(5, tier, active_model=active_model)
        except ValueError:
            continue
        if candidate.casefold() != current_model.casefold():
            return tier, candidate
    return None


__all__ = [
    "COMPLEXITY_TIERS", "Tier", "TierConfiguration", "TierProfile",
    "load_tier_configuration", "resolve_next_task_model",
    "resolve_task_model", "resolve_task_tier",
]