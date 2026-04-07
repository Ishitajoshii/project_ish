"""Episode lifecycle manager around the pure simulator helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from models import CircuitAction, CircuitObservation, CircuitState, CircuitTaskSpec
from server.simulator import (
    ACTION_SCALE_FACTOR,
    SUCCESS_TOLERANCE,
    apply_action,
    clamp_value,
    evaluate_circuit_state,
)


@dataclass
class CircuitEnvironment:
    """Manage one active circuit-tuning episode over a task registry."""

    tasks: Mapping[str, CircuitTaskSpec | dict[str, Any]] | CircuitTaskSpec | dict[str, Any]
    action_scale_factor: float = ACTION_SCALE_FACTOR
    task: CircuitTaskSpec | None = field(init=False, default=None)
    current_r_ohms: float = field(init=False, default=0.0)
    current_c_farads: float = field(init=False, default=0.0)
    current_hz: float = field(init=False, default=0.0)
    normalized_error: float = field(init=False, default=0.0)
    current_cost: float = field(init=False, default=0.0)
    step_count: int = field(init=False, default=0)
    cumulative_reward: float = field(init=False, default=0.0)
    best_score: float = field(init=False, default=0.0)
    done: bool = field(init=False, default=False)
    last_action_error: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.tasks = self._normalize_tasks(self.tasks)

    @property
    def is_done(self) -> bool:
        return self.task is not None and self.done

    def reset(self, task_id: str) -> CircuitObservation:
        task = self._select_task(task_id)
        self.task = task
        self.current_r_ohms = task.initial_r_ohms
        self.current_c_farads = task.initial_c_farads
        self.current_hz = 0.0
        self.normalized_error = 0.0
        self.current_cost = 0.0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.best_score = 0.0
        self.done = False
        self.last_action_error = None
        metrics = evaluate_circuit_state(
            r_ohms=self.current_r_ohms,
            c_farads=self.current_c_farads,
            target_hz=task.target_hz,
            step_count=self.step_count,
            max_steps=task.max_steps,
            success_tolerance=SUCCESS_TOLERANCE,
            min_r_ohms=task.min_r_ohms,
            max_r_ohms=task.max_r_ohms,
            min_c_farads=task.min_c_farads,
            max_c_farads=task.max_c_farads,
        )
        self.current_hz = float(metrics["current_hz"])
        self.normalized_error = float(metrics["normalized_error"])
        self.current_cost = float(metrics["normalized_cost"])
        return self._build_observation(
            current_hz=self.current_hz,
            normalized_error=self.normalized_error,
            current_cost=self.current_cost,
        )

    def step(self, action: CircuitAction | dict[str, Any]) -> tuple[CircuitObservation, float, bool]:
        task = self._require_task()
        if self.is_done:
            raise RuntimeError("episode is already done; call reset() to start a new task")

        if isinstance(action, CircuitAction):
            action_name = action.action.value
        else:
            action_name = str(action.get("action", ""))
        new_r, new_c, error = apply_action(
            self.current_r_ohms,
            self.current_c_farads,
            action_name,
            self.action_scale_factor,
            task.min_r_ohms,
            task.max_r_ohms,
            task.min_c_farads,
            task.max_c_farads,
        )
        self.current_r_ohms = new_r
        self.current_c_farads = new_c
        self.last_action_error = error
        self.step_count += 1
        metrics = evaluate_circuit_state(
            r_ohms=self.current_r_ohms,
            c_farads=self.current_c_farads,
            target_hz=task.target_hz,
            step_count=self.step_count,
            max_steps=task.max_steps,
            success_tolerance=SUCCESS_TOLERANCE,
            min_r_ohms=task.min_r_ohms,
            max_r_ohms=task.max_r_ohms,
            min_c_farads=task.min_c_farads,
            max_c_farads=task.max_c_farads,
        )

        reward = float(metrics["reward"])
        done = bool(metrics["done"])
        self.current_hz = float(metrics["current_hz"])
        self.normalized_error = float(metrics["normalized_error"])
        self.current_cost = float(metrics["normalized_cost"])
        self.cumulative_reward += reward
        self.best_score = max(self.best_score, reward)
        self.done = done

        observation = self._build_observation(
            current_hz=self.current_hz,
            normalized_error=self.normalized_error,
            current_cost=self.current_cost,
        )
        return observation, reward, done

    def state(self) -> CircuitState:
        self._require_task()
        return self._build_state()

    def score(self) -> float:
        self._require_task()
        return clamp_value(self.best_score, 0.0, 1.0)

    def close(self) -> None:
        self.task = None
        self.current_r_ohms = 0.0
        self.current_c_farads = 0.0
        self.current_hz = 0.0
        self.normalized_error = 0.0
        self.current_cost = 0.0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.best_score = 0.0
        self.done = False
        self.last_action_error = None

    def _require_task(self) -> CircuitTaskSpec:
        if self.task is None:
            raise RuntimeError("environment must be reset before use")
        return self.task

    def _build_observation(
        self,
        current_hz: float,
        normalized_error: float,
        current_cost: float,
    ) -> CircuitObservation:
        task = self._require_task()
        return CircuitObservation(
            task_id=task.task_id,
            circuit_type=task.circuit_type,
            target_hz=task.target_hz,
            current_r_ohms=self.current_r_ohms,
            current_c_farads=self.current_c_farads,
            current_hz=current_hz,
            normalized_error=normalized_error,
            current_cost=current_cost,
            remaining_steps=task.max_steps - self.step_count,
            last_action_error=self.last_action_error,
        )

    def _build_state(self) -> CircuitState:
        task = self._require_task()
        return CircuitState(
            task_id=task.task_id,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            best_score=self.best_score,
            done=self.done,
            current_r_ohms=self.current_r_ohms,
            current_c_farads=self.current_c_farads,
            current_hz=self.current_hz,
        )

    def _select_task(self, task_id: str) -> CircuitTaskSpec:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task_id: {task_id}") from exc

    @staticmethod
    def _normalize_tasks(
        tasks: Mapping[str, CircuitTaskSpec | dict[str, Any]] | CircuitTaskSpec | dict[str, Any],
    ) -> dict[str, CircuitTaskSpec]:
        if isinstance(tasks, CircuitTaskSpec):
            return {tasks.task_id: tasks}
        if isinstance(tasks, Mapping) and "task_id" in tasks:
            spec = tasks if isinstance(tasks, CircuitTaskSpec) else CircuitTaskSpec.model_validate(tasks)
            return {spec.task_id: spec}

        registry: dict[str, CircuitTaskSpec] = {}
        assert isinstance(tasks, Mapping)
        for task_id, task in tasks.items():
            spec = task if isinstance(task, CircuitTaskSpec) else CircuitTaskSpec.model_validate(task)
            registry[task_id] = spec
        return registry
