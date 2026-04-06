"""Typed OpenEnv-compatible models for actions, observations, and state."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CircuitAction(BaseModel):
    """Agent action that tweaks one component parameter by a signed delta."""

    component: str = Field(..., description="Component key, e.g. R, C, or L")
    delta: float = Field(..., description="Signed change to apply")


class CircuitObservation(BaseModel):
    """Observation exposed to an agent after each transition."""

    frequency_hz: float
    gain_db: float
    cost: float
    remaining_steps: int


class CircuitState(BaseModel):
    """Internal environment state for deterministic stepping and scoring."""

    task_id: str
    components: dict[str, float]
    step_count: int
    max_steps: int
    target_frequency_hz: float
    mode: str
    budget: float
    current_cost: float
