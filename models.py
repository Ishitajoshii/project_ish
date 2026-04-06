"""Typed OpenEnv-compatible models for actions, observations, and state."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CircuitAction(BaseModel):
    """Agent action that multiplicatively adjusts one RC component."""

    component: str = Field(..., description="Component key, either R or C")
    delta: float = Field(
        ...,
        description="Signed multiplicative step size; +0.2 means x1.2, -0.2 means /1.2",
    )


class CircuitObservation(BaseModel):
    """Observation exposed to an agent after each transition."""

    task_id: str
    circuit_type: str
    target_hz: float
    current_r_ohms: float
    current_c_farads: float
    current_output_hz: float
    normalized_error: float
    current_cost: float
    remaining_steps: int
    solved: bool
    last_action_error: str | None = None


class CircuitState(BaseModel):
    """Internal environment state for deterministic stepping and scoring."""

    task_id: str
    circuit_type: str
    current_r_ohms: float
    current_c_farads: float
    current_output_hz: float
    target_hz: float
    normalized_error: float
    current_cost: float
    step_count: int
    max_steps: int
    cumulative_reward: float
    best_score: float
    done: bool
    success_tolerance_pct: float
    cost_weight: float
    step_weight: float
    min_r_ohms: float
    max_r_ohms: float
    min_c_farads: float
    max_c_farads: float
    last_action_error: str | None = None
