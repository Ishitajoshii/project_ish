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

CircuitRL is an OpenEnv-compliant benchmark and demo for autonomous analog circuit tuning. It frames analog design as a sequential decision-making problem: an agent adjusts resistor and capacitor values step by step to match a target circuit specification while minimizing component cost.

The benchmark is designed for fast, deterministic evaluation and a strong demo experience. It supports Hugging Face Space deployment, Docker-based validation, typed OpenEnv models, structured inference logs, and multiple tasks with normalized graders.

The production runtime is a FastAPI backend plus a Vite frontend bundle. `openenv-core` remains a declared project dependency because the OpenEnv validator requires it, but the Docker image builds a pruned lock-derived runtime environment with `uv export --prune openenv-core`, so legacy Gradio dependencies are not installed in the served image.

## Why CircuitRL

Circuit tuning is usually iterative. Engineers often try multiple resistor and capacitor combinations before reaching the desired behavior. CircuitRL turns that process into a sequential optimization task with deterministic scoring, strong baseline comparisons, and a production reference controller that converges quickly under the benchmark rules.

This project is built around three ideas:
- match a target electrical specification
- minimize engineering cost
- beat naive baselines such as random search, heuristics, and brute-force scans

## Core Demo Story

A user sees:
- a target spec such as `1 kHz low-pass filter`
- live updates to `R` and `C`
- a convergence graph showing error dropping over time
- a final card showing `Target vs Achieved`
- a comparison against baseline strategies

Example outcome:
- Target: `1000 Hz`
- Achieved: `998 Hz`
- Cost Score: `0.82`
- Solved In: `6 steps`

## Features

- OpenEnv-compliant environment with `reset()`, `step()`, and `state()`
- Typed action, observation, and state models
- Deterministic RC filter simulation
- Multi-objective scoring with accuracy and cost
- 3+ benchmark tasks with graders
- Model-driven decision harness that uses the `openai` Python client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- Baseline comparisons:
	- random tuning
	- heuristic tuning
	- brute-force search
	- deterministic reference controller
- Vite + React frontend for episode playback
- Hugging Face Space deployment
- Docker build support
- Evaluation-friendly `inference.py`
- Animated UI with convergence plots

## Benchmark Tasks

Initial task set:
- `lp_1khz_budget` — easy
- `hp_500hz_budget` — medium
- `lp_10khz_budget` — medium
- `lp_2khz_low_cost` — hard

Each task defines:
- circuit type
- target frequency
- initial component values
- valid `R` and `C` ranges
- max steps
- scoring weights
- success threshold

## Action Space

`CircuitAction` exposes four legal multiplicative moves:
- `r_up`
- `r_down`
- `c_up`
- `c_down`

Each action applies a `x1.2` or `/1.2` update and then clamps values into the task bounds.

## Observation Space

`CircuitObservation` exposes:
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

`CircuitState` extends that with episode-level fields such as `step_count`, `cumulative_reward`, `best_score`, best-so-far component values, and `done`.

## Reward And Info

`CircuitReward` exposes:
- `value`
- `accuracy_score`
- `cost_efficiency`
- `step_efficiency`

`CircuitStepInfo` exposes:
- `task_id`
- `step_count`
- `best_score`
- `current_hz`
- `normalized_error`
- `current_cost`
- `success_threshold`
- `terminated_by`

## How It Works

CircuitRL uses the RC cutoff frequency equation:

`f_c = 1 / (2πRC)`

The agent interacts with the environment through a discrete multiplicative action space:
- increase `R`
- decrease `R`
- increase `C`
- decrease `C`

Each action updates the circuit, recomputes the current cutoff frequency, and produces a reward based on:
- closeness to the target
- component cost efficiency
- step efficiency

Final scores are normalized to the range `[0.0, 1.0]`.

Episodes terminate when the normalized error falls inside the success tolerance or when the task reaches its maximum step budget.

## Reference Scores

The repo includes a deterministic reference controller for regression testing and baseline comparison. Its exact scores on the built-in tasks are:

- `lp_1khz_budget` → `0.835974`
- `lp_10khz_budget` → `0.850260`
- `hp_500hz_budget` → `0.831674`
- `lp_2khz_low_cost` → `0.852066`

## Repository Structure

```text
circuitrl/
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── models.py
├── client.py
├── tasks/
│   ├── lp_1khz_budget.json
│   ├── lp_10khz_budget.json
│   ├── hp_500hz_budget.json
│   └── lp_2khz_low_cost.json
├── server/
│   ├── app.py
│   ├── ui_service.py
│   ├── environment.py
│   ├── grader.py
│   ├── baselines.py
│   ├── task_loader.py
│   ├── simulator.py
│   └── Dockerfile
├── ui/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── circuit_rl_app.tsx
│       ├── api_client.ts
│       ├── ui_types.ts
│       └── components/
└── tests/
    ├── test_app.py
    ├── test_simulator.py
    ├── test_grader.py
    └── test_environment.py
```

## Local Development

Run the backend API:

```bash
uv run --extra dev uvicorn server.app:app --reload
```

Run the Vite frontend in a second terminal:

```bash
cd ui
npm install
npm run dev
```

The frontend runs on `http://127.0.0.1:5173` and proxies `/api/*` to the FastAPI backend on `http://127.0.0.1:8000`.

If you want the Python server to serve the built frontend directly:

```bash
cd ui
npm run build
cd ..
uv run --extra dev uvicorn server.app:app --reload
```

Once `ui/dist` exists, the backend serves the built app at `/` and keeps the API under `/api/*`.

Run evaluator-style inference with the production policy backend:

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=your-model \
HF_TOKEN=your-token \
uv run python inference.py --task tasks/lp_1khz_budget.json
```

The default path uses the model-driven harness through the `openai` client and emits the required `[START]`, `[STEP]`, and `[END]` logs for every task.

The deterministic reference backend remains available for local regression checks:

```bash
uv run python inference.py --agent-backend policy --task tasks/lp_1khz_budget.json
```

Useful environment variables:

- `API_BASE_URL` — endpoint for the model API
- `MODEL_NAME` — model identifier for inference
- `HF_TOKEN` — evaluator-compatible API token
- `OPENAI_API_KEY` — optional direct key when running against the OpenAI endpoint

## Hugging Face Spaces

The repo is configured for a Docker Space via the README YAML frontmatter:
- `sdk: docker`
- `app_port: 8000`
- `tags: [openenv]`

Set your Space runtime variables in the Space settings:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

## Validation

Run the full local validation loop before submission:

```bash
uv run --extra dev pytest
uv run openenv validate .
docker build .
```

Or run the bundled helper:

```bash
./scripts/validate_submission.sh
./scripts/validate_submission.sh https://your-space.hf.space
```

## Frontend API Endpoints

- `GET /api/ui/catalog`
- `GET /api/ui/preview?task_id=...`
- `GET /api/ui/episode?task_id=...`
