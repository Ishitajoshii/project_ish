"""Typed OpenEnv-compatible models for actions, observations, and state."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class CircuitActionType(str, Enum):
    """Frozen discrete action space for RC tuning."""

    R_UP = "r_up"
    R_DOWN = "r_down"
    C_UP = "c_up"
    C_DOWN = "c_down"


class CircuitAction(BaseModel):
    """Agent action chosen from the discrete RC tuning action space."""

    action: CircuitActionType = Field(
        ...,
        description="Discrete action mapped to multiplicative updates of x1.2 or /1.2",
    )


class CircuitObservation(BaseModel):
    """Observation exposed to an agent after each transition."""

    task_id: str
    circuit_type: str
    target_hz: float
    current_r_ohms: float
    current_c_farads: float
    current_hz: float
    normalized_error: float
    current_cost: float
    remaining_steps: int
    last_action_error: str | None = None


class CircuitState(BaseModel):
    """Internal environment state for deterministic stepping and scoring."""

    task_id: str
    circuit_type: str
    current_r_ohms: float
    current_c_farads: float
    current_hz: float
    target_hz: float
    normalized_error: float
    current_cost: float
    step_count: int
    max_steps: int
    cumulative_reward: float
    best_score: float
    done: bool
    min_r_ohms: float
    max_r_ohms: float
    min_c_farads: float
    max_c_farads: float
    last_action_error: str | None = None


class CircuitTaskSpec(BaseModel):
    """Typed task definition loaded from deterministic benchmark JSON."""

    task_id: str
    circuit_type: str
    target_hz: float
    initial_r_ohms: float
    initial_c_farads: float
    min_r_ohms: float
    max_r_ohms: float
    min_c_farads: float
    max_c_farads: float
    max_steps: int
    success_tolerance_pct: float
    cost_weight: float
    step_weight: float

    @model_validator(mode="after")
    def validate_ranges(self) -> "CircuitTaskSpec":
        if self.circuit_type not in {"low_pass", "high_pass"}:
            raise ValueError(f"unsupported circuit_type={self.circuit_type!r}")
        if self.min_r_ohms >= self.max_r_ohms:
            raise ValueError("min_r_ohms must be < max_r_ohms")
        if self.min_c_farads >= self.max_c_farads:
            raise ValueError("min_c_farads must be < max_c_farads")
        if not (self.min_r_ohms <= self.initial_r_ohms <= self.max_r_ohms):
            raise ValueError("initial_r_ohms must be within resistor bounds")
        if not (self.min_c_farads <= self.initial_c_farads <= self.max_c_farads):
            raise ValueError("initial_c_farads must be within capacitor bounds")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if self.success_tolerance_pct < 0.0:
            raise ValueError("success_tolerance_pct must be non-negative")
        if self.cost_weight + self.step_weight > 1.0:
            raise ValueError("cost_weight + step_weight must be <= 1.0")
        return self
