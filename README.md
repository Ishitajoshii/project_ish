---
title: CircuitRL
emoji: "⚡"
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
tags:
  - openenv
---

# CircuitRL

CircuitRL is an OpenEnv benchmark and interactive demo for autonomous analog circuit tuning. An agent adjusts resistor and capacitor values step by step to hit a target cutoff frequency while balancing three engineering goals:

- accuracy against the target specification
- lower component cost
- fewer tuning steps

This repository is built for evaluation, not just presentation. It includes typed OpenEnv models, deterministic tasks and graders, a root-level `inference.py`, a Dockerized Hugging Face Space deployment, and a Vite frontend for step-by-step playback.

Live demo: [CircuitRL Space](https://theg1239-circuitrl-openenv.hf.space)  
Submission source: [GitHub repository](https://github.com/Ishitajoshii/project_ish)

## Why This Benchmark Exists

Analog circuit tuning is a real engineering workflow. Engineers routinely iterate on resistor and capacitor values to get the desired behavior from a simple filter while staying within practical constraints. CircuitRL turns that workflow into a compact, deterministic environment that is suitable for agent evaluation.

The point of the benchmark is not raw physics complexity. The point is to test whether an agent can make good sequential engineering decisions under a shaped reward, bounded action space, and realistic tradeoffs.

## What The Agent Is Solving

Each task starts with an RC filter configuration and a target cutoff frequency. The agent can only take one of four multiplicative tuning actions per step:

- `r_up`
- `r_down`
- `c_up`
- `c_down`

The environment updates the circuit with the standard RC relation:

`f_c = 1 / (2πRC)`

Episodes end when the agent reaches the success tolerance or uses the full step budget.

## Benchmark Tasks

CircuitRL ships with four deterministic tasks, which satisfy the bootcamp requirement of at least three graded tasks with increasing difficulty.

| Task | Difficulty | Objective |
| --- | --- | --- |
| `lp_1khz_budget` | Easy | Tune a low-pass filter toward `1 kHz` within the task budget. |
| `hp_500hz_budget` | Medium | Tune a high-pass filter toward `500 Hz`. |
| `lp_10khz_budget` | Medium | Tune a low-pass filter toward `10 kHz`. |
| `lp_2khz_low_cost` | Hard | Hit `2 kHz` while being more cost-efficient. |

Each task defines:

- circuit type
- target frequency
- initial `R` and `C`
- valid resistor and capacitor bounds
- success tolerance
- maximum step count
- cost and step weighting

## Action, Observation, Reward

### Action Space

`CircuitAction` exposes four legal actions:

- `r_up`
- `r_down`
- `c_up`
- `c_down`

Actions are multiplicative, not additive. Each move applies `×1.2` or `/1.2`, then clamps the updated value into the task bounds.

### Observation Space

`CircuitObservation` contains:

- `task_id`
- `circuit_type`
- `target_hz`
- `current_r_ohms`
- `current_c_farads`
- `current_hz`
- `normalized_error`
- `current_cost`
- `remaining_steps`
- `last_action_error`

`CircuitState` extends the observation with episode-level fields including:

- `step_count`
- `cumulative_reward`
- `best_score`
- best-so-far component values
- `done`

### Reward Model

`CircuitReward` exposes:

- `value`
- `accuracy_score`
- `cost_efficiency`
- `step_efficiency`

The environment returns meaningful reward over the full trajectory, not just at episode end. The final score is normalized to `[0.0, 1.0]`.

`CircuitStepInfo` adds transition metadata such as:

- `task_id`
- `step_count`
- `best_score`
- `current_hz`
- `normalized_error`
- `current_cost`
- `success_threshold`
- `terminated_by`

## Agent And Baselines

The default benchmark agent is a model-driven harness implemented in [server/agent_harness.py](/Users/ishaan/Projects/project_ish/server/agent_harness.py). It uses the `openai` Python client, reads evaluator-compatible environment variables, and proposes one legal action per step using structured JSON output.

The harness is paired with an exact simulator-backed evaluator board, which means every step is judged against the true next-state consequences of all legal actions. If the model proposes a dominated move, the evaluator can revise or override it.

The benchmark also includes baseline comparisons:

- random tuning
- heuristic tuning
- brute-force search
- deterministic reference policy

## Frontend Demo

The UI is a Vite + React console that presents the benchmark as a tool, not a landing page. It supports:

- task selection
- target specification display
- live `R` and `C` telemetry
- error-vs-step plotting
- step-by-step episode playback
- final score summary
- baseline comparison

Frontend API surface:

- `GET /api/ui/catalog`
- `GET /api/ui/preview?task_id=...`
- `GET /api/ui/episode?task_id=...`

## OpenEnv Compliance

CircuitRL implements the required environment surface:

- `reset()`
- `step()`
- `state()`

The submission also includes:

- [openenv.yaml](/Users/ishaan/Projects/project_ish/openenv.yaml)
- typed Pydantic models in [models.py](/Users/ishaan/Projects/project_ish/models.py)
- root-level [inference.py](/Users/ishaan/Projects/project_ish/inference.py)
- root-level [Dockerfile](/Users/ishaan/Projects/project_ish/Dockerfile)
- deterministic tasks in [tasks](/Users/ishaan/Projects/project_ish/tasks)
- tests in [tests](/Users/ishaan/Projects/project_ish/tests)

Additional validator-friendly endpoints:

- `GET /metadata`
- `GET /schema`
- `POST /mcp`
- `GET /health`

## Repository Layout

```text
circuitrl/
├── Dockerfile
├── README.md
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── scripts/
│   └── validate_submission.sh
├── server/
│   ├── agent_harness.py
│   ├── app.py
│   ├── baselines.py
│   ├── environment.py
│   ├── grader.py
│   ├── policy_agent.py
│   ├── simulator.py
│   ├── task_loader.py
│   └── ui_service.py
├── tasks/
├── tests/
└── ui/
```

## Local Development

### 1. Install Python dependencies

```bash
uv sync --extra dev
```

### 2. Install frontend dependencies

```bash
cd ui
npm install
cd ..
```

### 3. Run the backend

```bash
uv run --extra dev uvicorn server.app:app --reload
```

### 4. Run the frontend

```bash
cd ui
npm run dev
```

The frontend runs on `http://127.0.0.1:5173` and proxies `/api/*` to the FastAPI backend on `http://127.0.0.1:8000`.

### 5. Serve the built frontend from the backend

```bash
cd ui
npm run build
cd ..
uv run --extra dev uvicorn server.app:app --reload
```

Once `ui/dist` exists, the backend serves the built app at `/`.

## Inference

The evaluator-facing script is [inference.py](/Users/ishaan/Projects/project_ish/inference.py). It uses the OpenAI client for all model calls and emits the required structured stdout lines:

- `[START]`
- `[STEP]`
- `[END]`

### Environment Variables

| Variable | Purpose |
| --- | --- |
| `API_BASE_URL` | Base URL for the model endpoint. |
| `MODEL_NAME` | Model identifier used for inference. |
| `HF_TOKEN` | Evaluator-compatible token for OpenAI-compatible hosted endpoints. |
| `OPENAI_API_KEY` | Direct API key when using the official OpenAI endpoint. |

### Example: OpenAI endpoint

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-5.2
export OPENAI_API_KEY=your-key

uv run python inference.py
```

### Example: OpenAI-compatible router

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=your-router-model
export HF_TOKEN=your-token

uv run python inference.py
```

### Deterministic reference backend

For regression testing, a deterministic reference policy remains available:

```bash
uv run python inference.py --agent-backend policy --task tasks/lp_1khz_budget.json
```

## Reference Scores

Reference scores from the built-in benchmark set:

| Task | Score |
| --- | ---: |
| `lp_1khz_budget` | `0.835974` |
| `lp_10khz_budget` | `0.850260` |
| `hp_500hz_budget` | `0.831674` |
| `lp_2khz_low_cost` | `0.852066` |

## Docker And Hugging Face Spaces

The canonical deployment target is the root [Dockerfile](/Users/ishaan/Projects/project_ish/Dockerfile).

The project keeps `openenv-core` in the dependency graph because the validator expects it, but the deployment image uses a pruned `uv export` runtime install so validator-only Gradio dependencies are not shipped in the served container.

The live Space is:

- [theg1239/circuitrl-openenv](https://huggingface.co/spaces/theg1239/circuitrl-openenv)

## Validation

Run the full local submission loop:

```bash
uv run --extra dev pytest
uv run openenv validate .
docker build .
```

Or use the bundled helper:

```bash
./scripts/validate_submission.sh
./scripts/validate_submission.sh https://theg1239-circuitrl-openenv.hf.space
```

## Current Status

The current submission state has been checked locally and on the live Space:

- tests pass
- `openenv validate` passes
- Docker builds from the repo root
- the live Hugging Face Space responds on `/health`
- the live Hugging Face Space responds on `/reset`
- the frontend-backed API routes return valid task and episode payloads
