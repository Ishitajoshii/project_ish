# CircuitRL

CircuitRL is an OpenEnv-compliant benchmark and demo for autonomous analog circuit tuning. It frames analog design as a sequential decision-making problem: an agent adjusts resistor and capacitor values step by step to match a target circuit specification while minimizing component cost.

The benchmark is designed for fast, deterministic evaluation and a strong demo experience. It supports Hugging Face Space deployment, Docker-based validation, typed OpenEnv models, structured inference logs, and multiple tasks with normalized graders.

## Why CircuitRL

Circuit tuning is usually iterative. Engineers often try multiple resistor and capacitor combinations before reaching the desired behavior. CircuitRL turns that process into an RL-style optimization task where the agent learns to converge quickly and intelligently.

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
- Baseline comparisons:
	- random tuning
	- heuristic tuning
	- brute-force search
	- RL/LLM-driven agent
- Hugging Face Space deployment
- Docker build support
- Evaluation-friendly `inference.py`
- Animated UI with convergence plots

## Benchmark Tasks

Initial task set:
- `lp_1khz_budget`
- `lp_10khz_budget`
- `hp_500hz_budget`
- `lp_2khz_low_cost`

Each task defines:
- circuit type
- target frequency
- initial component values
- valid `R` and `C` ranges
- max steps
- scoring weights
- success threshold

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
│   ├── environment.py
│   ├── grader.py
│   ├── baselines.py
│   ├── task_loader.py
│   ├── simulator.py
│   └── Dockerfile
├── ui/
│   └── app.py
└── tests/
		├── test_simulator.py
		├── test_grader.py
		└── test_environment.py
```
