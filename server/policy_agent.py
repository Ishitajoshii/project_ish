"""Production circuit-tuning agent built from exact tabular control planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from models import CircuitAction, CircuitObservation, CircuitTaskSpec
from server.environment import CircuitEnvironment
from server.grader import is_success
from server.simulator import (
    ACTION_SCALE_FACTOR,
    SUCCESS_TOLERANCE,
    apply_action,
    evaluate_circuit_state,
    valid_actions,
)

AGENT_NAME = "tabular-value-iteration"
ACTION_PRIORITY = {"r_down": 4, "c_down": 3, "r_up": 2, "c_up": 1}


@dataclass(frozen=True)
class PolicyEntry:
    """Cached best-decision summary for one reachable benchmark state."""

    best_action: str | None
    final_score: float
    steps_to_goal: int


@dataclass
class TaskPolicy:
    """Exact finite-horizon control policy for one benchmark task."""

    task: CircuitTaskSpec
    entries: dict[tuple[str, str, int, str], PolicyEntry] = field(default_factory=dict)

    def action_for(self, observation: CircuitObservation, best_score: float = 0.0) -> str:
        """Return the greedy optimal action for the current observation."""

        step_count = self.task.max_steps - observation.remaining_steps
        entry = self.solve(
            _state_key(
                observation.current_r_ohms,
                observation.current_c_farads,
                step_count,
                best_score,
            )
        )
        if entry.best_action is None:
            raise RuntimeError("policy requested an action from a terminal circuit state")
        return entry.best_action

    def solve(self, state_key: tuple[str, str, int, str]) -> PolicyEntry:
        """Solve one state recursively and cache the best control decision."""

        if state_key in self.entries:
            return self.entries[state_key]

        r_key, c_key, step_count, best_score_key = state_key
        r_ohms = float(r_key)
        c_farads = float(c_key)
        best_score = float(best_score_key)
        metrics = _evaluate_state(self.task, r_ohms, c_farads, step_count)
        if bool(metrics["done"]):
            entry = PolicyEntry(
                best_action=None,
                final_score=best_score,
                steps_to_goal=0,
            )
            self.entries[state_key] = entry
            return entry

        best_action: str | None = None
        best_entry: PolicyEntry | None = None
        best_key: tuple[float, int, float, float, float, int] | None = None

        for action in valid_actions():
            next_r_ohms, next_c_farads, _ = apply_action(
                r_ohms,
                c_farads,
                action,
                ACTION_SCALE_FACTOR,
                self.task.min_r_ohms,
                self.task.max_r_ohms,
                self.task.min_c_farads,
                self.task.max_c_farads,
            )
            next_metrics = _evaluate_state(
                self.task,
                next_r_ohms,
                next_c_farads,
                step_count + 1,
            )
            next_reward = float(next_metrics["reward"])
            next_error = float(next_metrics["normalized_error"])
            next_cost = float(next_metrics["normalized_cost"])
            next_best_score = max(best_score, next_reward)
            child_entry = self.solve(
                _state_key(next_r_ohms, next_c_farads, step_count + 1, next_best_score)
            )
            if child_entry.final_score <= best_score + 1e-12:
                steps_to_goal = 0
            elif next_best_score >= child_entry.final_score - 1e-12:
                steps_to_goal = 1
            else:
                steps_to_goal = 1 + child_entry.steps_to_goal
            candidate_entry = PolicyEntry(
                best_action=action,
                final_score=child_entry.final_score,
                steps_to_goal=steps_to_goal,
            )
            candidate_key = (
                candidate_entry.final_score,
                -candidate_entry.steps_to_goal,
                next_reward,
                -next_error,
                -next_cost,
                ACTION_PRIORITY[action],
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_action = action
                best_entry = candidate_entry

        assert best_action is not None
        assert best_entry is not None
        entry = PolicyEntry(
            best_action=best_action,
            final_score=best_entry.final_score,
            steps_to_goal=best_entry.steps_to_goal,
        )
        self.entries[state_key] = entry
        return entry


@dataclass
class TabularValueIterationAgent:
    """Exact deterministic control agent over the frozen benchmark tasks."""

    tasks: Mapping[str, CircuitTaskSpec]
    task_policies: dict[str, TaskPolicy] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.task_policies = {
            task_id: TaskPolicy(task)
            for task_id, task in self.tasks.items()
        }
        for task_id, task in self.tasks.items():
            self.task_policies[task_id].solve(
                _state_key(task.initial_r_ohms, task.initial_c_farads, 0, 0.0)
            )

    @property
    def agent_name(self) -> str:
        return AGENT_NAME

    def choose_action(self, observation: CircuitObservation, best_score: float = 0.0) -> str:
        """Choose the optimal action for one observation."""

        try:
            policy = self.task_policies[observation.task_id]
        except KeyError as exc:
            raise KeyError(f"unknown task_id: {observation.task_id}") from exc
        return policy.action_for(observation, best_score)

    def peak_reward_for_task(self, task_id: str) -> float:
        """Return the exact best achievable benchmark score for one task."""

        task = self.tasks[task_id]
        entry = self.task_policies[task_id].solve(
            _state_key(task.initial_r_ohms, task.initial_c_farads, 0, 0.0)
        )
        return entry.final_score

    def peak_path_for_task(self, task_id: str) -> list[str]:
        """Return one greedy optimal action path for the task's initial state."""

        task = self.tasks[task_id]
        env = CircuitEnvironment({task_id: task})
        observation = env.reset(task_id)
        actions: list[str] = []
        while not env.is_done:
            action = self.choose_action(observation, env.score())
            actions.append(action)
            observation, _, done = env.step(CircuitAction(action=action))
            if done:
                break
        return actions


def run_policy_episode(
    env: CircuitEnvironment,
    task_id: str,
    agent: TabularValueIterationAgent,
    *,
    baseline_name: str = "policy",
) -> dict[str, Any]:
    """Run one full policy episode and return the shared benchmark payload shape."""

    observation = env.reset(task_id)
    while not env.is_done:
        action_name = agent.choose_action(observation, env.score())
        observation, _, done = env.step(CircuitAction(action=action_name))
        if done:
            break

    state = env.state()
    score = env.score()
    return {
        "baseline_name": baseline_name,
        "task_id": task_id,
        "score": score,
        "success": is_success(score),
        "steps_used": state.step_count,
        "evaluations": None,
        "achieved_hz": float(state.best_hz),
        "current_r_ohms": float(state.best_r_ohms),
        "current_c_farads": float(state.best_c_farads),
        "normalized_error": float(state.best_normalized_error),
        "normalized_cost": float(state.best_normalized_cost),
    }


def _evaluate_state(
    task: CircuitTaskSpec,
    r_ohms: float,
    c_farads: float,
    step_count: int,
) -> dict[str, float | bool]:
    """Compute one state's reward/error/cost tuple under the frozen benchmark rules."""

    return evaluate_circuit_state(
        r_ohms=r_ohms,
        c_farads=c_farads,
        target_hz=task.target_hz,
        step_count=step_count,
        max_steps=task.max_steps,
        success_tolerance=SUCCESS_TOLERANCE,
        min_r_ohms=task.min_r_ohms,
        max_r_ohms=task.max_r_ohms,
        min_c_farads=task.min_c_farads,
        max_c_farads=task.max_c_farads,
    )


def _state_key(
    r_ohms: float,
    c_farads: float,
    step_count: int,
    best_score: float,
) -> tuple[str, str, int, str]:
    """Serialize one physical circuit state into a stable cache key."""

    return (
        f"{r_ohms:.15e}",
        f"{c_farads:.15e}",
        step_count,
        f"{best_score:.15e}",
    )
