"""Deterministic split gate for choosing single, wide, or exploratory work."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class SplitEstimate(BaseModel):
    estimated_steps: int = Field(
        ge=0, le=10_000,
        description="Estimated single-agent steps for the whole objective.",
    )
    sequential_dependency: bool = Field(
        description="Whether later work depends on earlier discoveries or shared state.",
    )
    independent_branches: int = Field(
        ge=0, le=8,
        description="Number of independently useful branches, capped at eight.",
    )
    read_heavy: bool = Field(
        description="Whether branches mostly inspect/analyze rather than mutate files.",
    )
    disjoint_writes: bool = Field(
        description="Whether write scopes are non-overlapping and explicitly declared.",
    )
    uncertain_steps: bool = Field(
        description="Whether results are likely to change the plan or reveal unknown work.",
    )


@dataclass(frozen=True)
class SplitDecision:
    mode: str
    reason: str


def decide_mode(estimate: SplitEstimate | None) -> SplitDecision:
    if estimate is None:
        return SplitDecision("single", "No split estimate was supplied.")
    if estimate.sequential_dependency:
        return SplitDecision("single", "The work has sequential dependencies.")
    if estimate.estimated_steps <= 8:
        return SplitDecision("single", "The estimated work is small enough for one agent.")
    if estimate.independent_branches >= 3 and (
        estimate.read_heavy or estimate.disjoint_writes
    ):
        return SplitDecision("wide", "The estimate supports independent parallel branches.")
    if estimate.uncertain_steps:
        return SplitDecision(
            "single_then_escalate",
            "Explore in the parent first, then delegate only unresolved work.",
        )
    return SplitDecision("single", "The estimate does not justify fan-out.")


__all__ = ["SplitDecision", "SplitEstimate", "decide_mode"]