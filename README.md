# circuitrl

A lightweight benchmark scaffold for circuit-design reinforcement learning with deterministic tasks and a local OpenEnv-style server.

## What is included

- `server/`: simulation, environment transitions, grading, baselines, API app
- `tasks/`: deterministic benchmark definitions
- `ui/`: tiny local UI for manual inspection
- `inference.py`: evaluator-facing entrypoint with strict stdout behavior
- `tests/`: quick confidence checks

## Quick start

1. Install dependencies:

```bash
pip install -e .
```

2. Run tests:

```bash
pytest -q
```

3. Start API server:

```bash
uvicorn server.app:app --reload --port 8000
```

4. Run local inference script:

```bash
python inference.py --task tasks/lp_1khz_budget.json
```
