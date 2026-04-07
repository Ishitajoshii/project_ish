"""Reset/step/state logic that wraps simulator and grader behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models import CircuitObservation, CircuitState
from server.simulator import (
    ACTION_SCALE_FACTOR,
    SUCCESS_TOLERANCE,
    apply_action,
    evaluate_circuit_state,
)


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
            current_hz=0.0,
            target_hz=float(self.task["target_hz"]),
            normalized_error=0.0,
            current_cost=0.0,
            step_count=0,
            max_steps=int(self.task["max_steps"]),
            cumulative_reward=0.0,
            best_score=0.0,
            done=False,
            min_r_ohms=float(self.task["min_r_ohms"]),
            max_r_ohms=float(self.task["max_r_ohms"]),
            min_c_farads=float(self.task["min_c_farads"]),
            max_c_farads=float(self.task["max_c_farads"]),
        )
        self._refresh_metrics()
        return self._observation()

    def step(self, action: dict[str, Any]) -> CircuitObservation:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step")
        if self.is_done:
            return self._observation()

        action_name = str(action.get("action", ""))
        new_r, new_c, error = apply_action(
            self.state.current_r_ohms,
            self.state.current_c_farads,
            action_name,
            ACTION_SCALE_FACTOR,
            self.state.min_r_ohms,
            self.state.max_r_ohms,
            self.state.min_c_farads,
            self.state.max_c_farads,
        )
        if error is not None:
            self.state.last_action_error = error
            return self._observation()

        self.state.current_r_ohms = new_r
        self.state.current_c_farads = new_c
        self.state.last_action_error = None
        self.state.step_count += 1
        self._refresh_metrics()
        return self._observation()

    def score(self) -> float:
        if self.state is None:
            raise RuntimeError("Environment must be reset before score")

        return self.state.best_score

    def _observation(self) -> CircuitObservation:
        assert self.state is not None
        return CircuitObservation(
            task_id=self.state.task_id,
            circuit_type=self.state.circuit_type,
            target_hz=self.state.target_hz,
            current_r_ohms=self.state.current_r_ohms,
            current_c_farads=self.state.current_c_farads,
            current_hz=self.state.current_hz,
            normalized_error=self.state.normalized_error,
            current_cost=self.state.current_cost,
            remaining_steps=self.state.max_steps - self.state.step_count,
            last_action_error=self.state.last_action_error,
        )

    def _success_tolerance_ratio(self) -> float:
        return SUCCESS_TOLERANCE

    def _refresh_metrics(self) -> None:
        assert self.state is not None
        evaluation = evaluate_circuit_state(
            self.state.current_r_ohms,
            self.state.current_c_farads,
            self.state.target_hz,
            self.state.step_count,
            self.state.max_steps,
            self._success_tolerance_ratio(),
            self.state.min_r_ohms,
            self.state.max_r_ohms,
            self.state.min_c_farads,
            self.state.max_c_farads,
        )
        self.state.current_hz = float(evaluation["current_hz"])
        self.state.normalized_error = float(evaluation["normalized_error"])
        self.state.current_cost = float(evaluation["normalized_cost"])
        self.state.done = bool(evaluation["done"])
        if self.state.step_count > 0:
            current_reward = float(evaluation["reward"])
            self.state.cumulative_reward += current_reward
            self.state.best_score = max(self.state.best_score, current_reward)
