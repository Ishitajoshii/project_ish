"""Evaluator-facing script that emits strict episode log lines to stdout."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from models import CircuitAction, CircuitObservation
from server.environment import CircuitEnvironment
from server.grader import is_success
from server.task_loader import get_task_ids_in_order, load_task_file, load_tasks

BENCHMARK_NAME = "circuitrl"
VALID_ACTIONS = ("r_up", "r_down", "c_up", "c_down")

load_dotenv()


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime configuration read from environment variables."""

    api_base_url: str
    model_name: str
    hf_token: str
    image_name: str | None = None


def load_inference_config() -> InferenceConfig:
    """Read inference configuration from environment variables."""

    api_base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct-AWQ"
    if "api.openai.com" in api_base_url:
        hf_token = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_API_KEY")
            or os.getenv("API_KEY")
            or os.getenv("HF_TOKEN")
        )
    else:
        hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_API_KEY")
        )
    image_name = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")

    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for inference")

    return InferenceConfig(
        api_base_url=api_base_url,
        model_name=model_name,
        hf_token=hf_token,
        image_name=image_name,
    )


def build_openai_client(config: InferenceConfig) -> OpenAI:
    """Create the OpenAI-compatible client used for evaluator inference."""

    return OpenAI(
        base_url=config.api_base_url,
        api_key=config.hf_token,
    )


def _build_action_prompt(observation: CircuitObservation) -> list[dict[str, str]]:
    """Build a compact deterministic prompt for action selection."""

    return [
        {
            "role": "system",
            "content": (
                "You are tuning an RC circuit. "
                "Reply with exactly one token from this set and nothing else: "
                "r_up | r_down | c_up | c_down."
            ),
        },
        {
            "role": "user",
            "content": (
                f"task_id={observation.task_id}\n"
                f"target_hz={observation.target_hz}\n"
                f"current_hz={observation.current_hz}\n"
                f"current_r_ohms={observation.current_r_ohms}\n"
                f"current_c_farads={observation.current_c_farads}\n"
                f"normalized_error={observation.normalized_error}\n"
                f"current_cost={observation.current_cost}\n"
                f"remaining_steps={observation.remaining_steps}\n"
                f"last_action_error={observation.last_action_error or 'null'}"
            ),
        },
    ]


def _extract_action(content: str) -> str:
    """Parse one valid action token from model output."""

    cleaned = content.strip().lower()
    if cleaned in VALID_ACTIONS:
        return cleaned

    for action in VALID_ACTIONS:
        if action in cleaned:
            return action

    raise ValueError(f"model returned invalid action: {content!r}")


def _fallback_action(observation: CircuitObservation) -> str:
    """Use a deterministic fallback if the model response is missing or malformed."""

    return "r_up" if observation.current_hz > observation.target_hz else "r_down"


def choose_model_action(
    client: OpenAI,
    config: InferenceConfig,
    observation: CircuitObservation,
) -> str:
    """Ask the configured model for the next discrete circuit action."""

    try:
        response = client.chat.completions.create(
            model=config.model_name,
            messages=_build_action_prompt(observation),
            temperature=0.0,
            max_completion_tokens=16,
        )
        content = response.choices[0].message.content or ""
        return _extract_action(content)
    except Exception:
        return _fallback_action(observation)


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
    client: OpenAI | None = None,
    log_stdout: bool = False,
) -> dict:
    """Run one deterministic policy and return evaluator payload."""

    resolved_config = config or load_inference_config()
    _client = client or build_openai_client(resolved_config)

    task = load_task_file(Path(task_file))
    env = CircuitEnvironment({task.task_id: task})
    obs = env.reset(task.task_id)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    if log_stdout:
        log_start(task=task.task_id, env=BENCHMARK_NAME, model=resolved_config.model_name)

    try:
        while not env.is_done:
            action = choose_model_action(_client, resolved_config, obs)
            obs, reward, done = env.step(CircuitAction(action=action))
            rewards.append(reward)
            steps_taken += 1
            if log_stdout:
                log_step(
                    step=steps_taken,
                    action=action,
                    reward=reward,
                    done=done,
                    error=obs.last_action_error,
                )

        score = env.score()
        success = is_success(score)
        return {
            "task_id": task.task_id,
            "score": score,
            "details": {
                "final_output_hz": obs.current_hz,
                "normalized_error": obs.normalized_error,
                "cost": obs.current_cost,
                "solved": env.is_done and obs.normalized_error <= 0.02,
                "model_name": resolved_config.model_name,
                "rewards": rewards,
                "steps": steps_taken,
            },
        }
    finally:
        if not score and env.task is not None:
            score = env.score()
            success = is_success(score)
        if log_stdout:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def run_all_inference(
    task_dir: str | None = None,
    *,
    config: InferenceConfig | None = None,
    client: OpenAI | None = None,
    log_stdout: bool = False,
) -> list[dict]:
    """Run inference across the deterministic benchmark task set."""

    resolved_config = config or load_inference_config()
    resolved_client = client or build_openai_client(resolved_config)
    base_dir = Path(task_dir) if task_dir is not None else Path("tasks")
    tasks = load_tasks(base_dir)
    ordered_ids = get_task_ids_in_order(tasks)
    return [
        run_inference(
            str(base_dir / f"{task_id}.json"),
            config=resolved_config,
            client=resolved_client,
            log_stdout=log_stdout,
        )
        for task_id in ordered_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Path to one task JSON; omit to run all benchmark tasks")
    args = parser.parse_args()

    config = load_inference_config()
    client = build_openai_client(config)
    results = (
        [run_inference(args.task, config=config, client=client, log_stdout=True)]
        if args.task
        else run_all_inference(config=config, client=client, log_stdout=True)
    )


if __name__ == "__main__":
    main()
