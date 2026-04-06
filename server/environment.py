"""Reset/step/state logic that wraps simulator and grader behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models import CircuitObservation, CircuitState
from server.grader import normalized_score
from server.simulator import apply_action, component_cost, cutoff_frequency_hz, gain_db


@dataclass
class CircuitEnvironment:
    """Deterministic environment with bounded horizon and static task target."""

    task: dict[str, Any]

    def __post_init__(self) -> None:
        self.state: CircuitState | None = None

    @property
    def is_done(self) -> bool:
        return self.state is not None and self.state.step_count >= self.state.max_steps

    def reset(self) -> CircuitObservation:
        components = dict(self.task["initial_components"])
        cost = component_cost(components, self.task["cost_weights"])
        self.state = CircuitState(
            task_id=self.task["task_id"],
            components=components,
            step_count=0,
            max_steps=int(self.task["max_steps"]),
            target_frequency_hz=float(self.task["target_frequency_hz"]),
            mode=self.task["mode"],
            budget=float(self.task["budget"]),
            current_cost=cost,
        )
        return self._observation()

    def step(self, action: dict[str, Any]) -> CircuitObservation:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step")
        if self.is_done:
            return self._observation()

        self.state.components = apply_action(self.state.components, action)
        self.state.current_cost = component_cost(self.state.components, self.task["cost_weights"])
        self.state.step_count += 1
        return self._observation()

    def score(self) -> float:
        if self.state is None:
            raise RuntimeError("Environment must be reset before score")

        achieved_fc = cutoff_frequency_hz(self.state.components)
        freq_error_ratio = abs(achieved_fc - self.state.target_frequency_hz) / max(self.state.target_frequency_hz, 1e-9)
        over_budget_ratio = max(self.state.current_cost - self.state.budget, 0.0) / max(self.state.budget, 1e-9)
        return normalized_score(freq_error_ratio, over_budget_ratio)

    def _observation(self) -> CircuitObservation:
        assert self.state is not None
        probe_hz = self.state.target_frequency_hz
        gdb = gain_db(self.state.mode, self.state.components, probe_hz)
        return CircuitObservation(
            frequency_hz=cutoff_frequency_hz(self.state.components),
            gain_db=gdb,
            cost=self.state.current_cost,
            remaining_steps=self.state.max_steps - self.state.step_count,
        )
