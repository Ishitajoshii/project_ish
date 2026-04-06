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
        return self.state is not None and self.state.done

    def reset(self) -> CircuitObservation:
        self.state = CircuitState(
            task_id=self.task["task_id"],
            circuit_type=self.task["circuit_type"],
            current_r_ohms=float(self.task["initial_r_ohms"]),
            current_c_farads=float(self.task["initial_c_farads"]),
            current_output_hz=0.0,
            target_hz=float(self.task["target_hz"]),
            normalized_error=0.0,
            current_cost=0.0,
            step_count=0,
            max_steps=int(self.task["max_steps"]),
            cumulative_reward=0.0,
            best_score=0.0,
            done=False,
            success_tolerance_pct=float(self.task["success_tolerance_pct"]),
            cost_weight=float(self.task["cost_weight"]),
            step_weight=float(self.task["step_weight"]),
            min_r_ohms=float(self.task["min_r_ohms"]),
            max_r_ohms=float(self.task["max_r_ohms"]),
            min_c_farads=float(self.task["min_c_farads"]),
            max_c_farads=float(self.task["max_c_farads"]),
        )
        self._refresh_metrics()
        self.state.best_score = self.score()
        return self._observation()

    def step(self, action: dict[str, Any]) -> CircuitObservation:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step")
        if self.is_done:
            return self._observation()

        components = {
            "R": self.state.current_r_ohms,
            "C": self.state.current_c_farads,
        }
        action_name = str(action.get("action", ""))
        try:
            updated = apply_action(components, action_name, self._bounds())
        except ValueError as exc:
            self.state.last_action_error = str(exc)
            return self._observation()

        self.state.current_r_ohms = updated["R"]
        self.state.current_c_farads = updated["C"]
        self.state.last_action_error = None
        self.state.step_count += 1
        self._refresh_metrics()
        step_score = self.score()
        self.state.cumulative_reward += step_score
        self.state.best_score = max(self.state.best_score, step_score)
        return self._observation()

    def score(self) -> float:
        if self.state is None:
            raise RuntimeError("Environment must be reset before score")

        return normalized_score(
            self.state.normalized_error,
            self.state.current_cost,
            self.state.step_count,
            self.state.max_steps,
            self.state.cost_weight,
            self.state.step_weight,
        )

    def _observation(self) -> CircuitObservation:
        assert self.state is not None
        return CircuitObservation(
            task_id=self.state.task_id,
            circuit_type=self.state.circuit_type,
            target_hz=self.state.target_hz,
            current_r_ohms=self.state.current_r_ohms,
            current_c_farads=self.state.current_c_farads,
            current_output_hz=self.state.current_output_hz,
            normalized_error=self.state.normalized_error,
            current_cost=self.state.current_cost,
            remaining_steps=self.state.max_steps - self.state.step_count,
            solved=self.state.done and self.state.normalized_error <= self._success_tolerance_ratio(),
            last_action_error=self.state.last_action_error,
        )

    def _bounds(self) -> dict[str, tuple[float, float]]:
        assert self.state is not None
        return {
            "R": (self.state.min_r_ohms, self.state.max_r_ohms),
            "C": (self.state.min_c_farads, self.state.max_c_farads),
        }

    def _success_tolerance_ratio(self) -> float:
        assert self.state is not None
        return self.state.success_tolerance_pct / 100.0

    def _refresh_metrics(self) -> None:
        assert self.state is not None
        components = {
            "R": self.state.current_r_ohms,
            "C": self.state.current_c_farads,
        }
        self.state.current_output_hz = cutoff_frequency_hz(components)
        self.state.normalized_error = abs(self.state.current_output_hz - self.state.target_hz) / max(
            self.state.target_hz,
            1e-9,
        )
        self.state.current_cost = component_cost(components, self._bounds())
        self.state.done = (
            self.state.normalized_error <= self._success_tolerance_ratio()
            or self.state.step_count >= self.state.max_steps
        )

        # Gain is still a useful derived metric for UI/debugging callers, so keep
        # the circuit-type mapping exercised during every state refresh.
        gain_db(self.state.circuit_type, components, self.state.target_hz)
