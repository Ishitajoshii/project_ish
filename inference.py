"""Evaluator-facing script that emits strict episode log lines to stdout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

from models import CircuitAction
from server.environment import CircuitEnvironment
from server.grader import is_success
from server.agent_harness import (
    HARNESS_BACKEND_NAME,
    AgentHarness,
    HarnessConfig,
    build_model_client,
    load_harness_config,
)
from server.policy_agent import AGENT_NAME as POLICY_AGENT_NAME
from server.policy_agent import TabularValueIterationAgent
from server.task_loader import get_task_ids_in_order, load_task_file, load_tasks

BENCHMARK_NAME = "circuitrl"
VALID_ACTIONS = ("r_up", "r_down", "c_up", "c_down")
AgentBackend = Literal["llm", "policy"]
InferenceConfig = HarnessConfig

load_dotenv()


def load_inference_config() -> InferenceConfig:
    """Read inference configuration from environment variables."""

    return load_harness_config()


def build_inference_client(config: InferenceConfig) -> Any:
    """Create the model client used by the evaluator loop."""

    return build_model_client(config)


def log_start(task: str, env: str, model: str) -> None:
    """Emit the exact required episode-start line."""

    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    """Emit the exact required per-step line."""

    error_value = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Emit the exact required episode-end line."""

    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    success_value = str(success).lower()
    print(
        f"[END] success={success_value} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_inference(
    task_file: str,
    *,
    config: InferenceConfig | None = None,
    client: Any | None = None,
    log_stdout: bool = False,
    agent_backend: AgentBackend = "llm",
    harness_agent: AgentHarness | None = None,
    policy_agent: TabularValueIterationAgent | None = None,
) -> dict[str, Any]:
    """Run one benchmark task with the requested backend and return evaluator payload."""

    task = load_task_file(Path(task_file))
    env = CircuitEnvironment({task.task_id: task})
    score = 0.0
    success = False
    rewards: list[float] = []
    steps_taken = 0
    model_name = POLICY_AGENT_NAME
    details: dict[str, Any] = {}

    if agent_backend == "llm":
        resolved_config = config or load_inference_config()
        resolved_client = client or build_inference_client(resolved_config)
        resolved_agent = harness_agent or AgentHarness(
            tasks={task.task_id: task},
            config=resolved_config,
            client=resolved_client,
        )
        model_name = resolved_config.model_name
        if log_stdout:
            log_start(task=task.task_id, env=BENCHMARK_NAME, model=model_name)
        try:
            result = resolved_agent.run_episode(env, task.task_id)
            rewards = [step.reward for step in result.trace_steps]
            steps_taken = len(result.trace_steps)
            score = result.score
            success = result.success
            if log_stdout:
                for index, step in enumerate(result.trace_steps, start=1):
                    log_step(
                        step=index,
                        action=step.action,
                        reward=step.reward,
                        done=index == len(result.trace_steps),
                        error=None,
                    )
            details = {
                "final_output_hz": result.final_observation.current_hz,
                "normalized_error": result.final_observation.normalized_error,
                "cost": result.final_observation.current_cost,
                "solved": result.state.done and result.final_observation.normalized_error <= 0.02,
                "model_name": model_name,
                "agent_backend": HARNESS_BACKEND_NAME,
                "rewards": rewards,
                "steps": steps_taken,
                "simulator_evaluations": result.simulator_evaluations,
                "trace": [
                    {
                        "step": step.step,
                        "action": step.action,
                        "model_action": step.model_action,
                        "selected_by": step.selected_by,
                        "reward": step.reward,
                        "best_score_after": step.best_score_after,
                        "note": step.note,
                    }
                    for step in result.trace_steps
                ],
            }
        finally:
            if not score and env.task is not None:
                score = env.score()
                success = is_success(score)
            if log_stdout:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    else:
        resolved_policy_agent = policy_agent or TabularValueIterationAgent({task.task_id: task})
        model_name = POLICY_AGENT_NAME
        if log_stdout:
            log_start(task=task.task_id, env=BENCHMARK_NAME, model=model_name)
        try:
            observation = env.reset(task.task_id)
            while not env.is_done:
                action = resolved_policy_agent.choose_action(observation, env.score())
                observation, reward, done = env.step(CircuitAction(action=action))
                rewards.append(reward)
                steps_taken += 1
                if log_stdout:
                    log_step(
                        step=steps_taken,
                        action=action,
                        reward=reward,
                        done=done,
                        error=observation.last_action_error,
                    )
                if done:
                    break
            score = env.score()
            success = is_success(score)
            details = {
                "final_output_hz": observation.current_hz,
                "normalized_error": observation.normalized_error,
                "cost": observation.current_cost,
                "solved": env.is_done and observation.normalized_error <= 0.02,
                "model_name": model_name,
                "agent_backend": "policy",
                "rewards": rewards,
                "steps": steps_taken,
            }
        finally:
            if not score and env.task is not None:
                score = env.score()
                success = is_success(score)
            if log_stdout:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task.task_id,
        "score": score,
        "details": details,
    }


def run_all_inference(
    task_dir: str | None = None,
    *,
    config: InferenceConfig | None = None,
    client: Any | None = None,
    log_stdout: bool = False,
    agent_backend: AgentBackend = "llm",
) -> list[dict[str, Any]]:
    """Run inference across the deterministic benchmark task set."""

    base_dir = Path(task_dir) if task_dir is not None else Path("tasks")
    tasks = load_tasks(base_dir)
    ordered_ids = get_task_ids_in_order(tasks)
    resolved_config = None
    resolved_client = None
    resolved_harness_agent = None
    resolved_policy_agent = None

    if agent_backend == "llm":
        resolved_config = config or load_inference_config()
        resolved_client = client or build_inference_client(resolved_config)
        resolved_harness_agent = AgentHarness(
            tasks=tasks,
            config=resolved_config,
            client=resolved_client,
        )
    else:
        resolved_policy_agent = TabularValueIterationAgent(tasks)

    return [
        run_inference(
            str(base_dir / f"{task_id}.json"),
            config=resolved_config,
            client=resolved_client,
            log_stdout=log_stdout,
            agent_backend=agent_backend,
            harness_agent=resolved_harness_agent,
            policy_agent=resolved_policy_agent,
        )
        for task_id in ordered_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Path to one task JSON; omit to run all benchmark tasks")
    parser.add_argument(
        "--agent-backend",
        choices=("llm", "policy"),
        default="llm",
        help="Use the model-driven harness or the deterministic reference policy backend.",
    )
    args = parser.parse_args()

    config = None
    client = None
    if args.agent_backend == "llm":
        config = load_inference_config()
        client = build_inference_client(config)
    if args.task:
        run_inference(
            args.task,
            config=config,
            client=client,
            log_stdout=True,
            agent_backend=args.agent_backend,
        )
    else:
        run_all_inference(
            config=config,
            client=client,
            log_stdout=True,
            agent_backend=args.agent_backend,
        )


if __name__ == "__main__":
    main()
