"""Polished Gradio frontend for CircuitRL benchmark playback."""

from __future__ import annotations

import math
import time
from html import escape
from typing import Any

import gradio as gr

from models import CircuitAction, CircuitObservation, CircuitTaskSpec
from server.baselines import run_bruteforce_baseline, run_heuristic_baseline, run_random_baseline
from server.environment import CircuitEnvironment
from server.grader import is_success
from server.simulator import (
    ACTION_SCALE_FACTOR,
    SUCCESS_TOLERANCE,
    apply_action,
    evaluate_circuit_state,
    normalize_log_value,
    valid_actions,
)
from server.task_loader import get_task_ids_in_order, load_tasks

TASKS = load_tasks()
TASK_IDS = get_task_ids_in_order(TASKS)
DEFAULT_TASK_ID = TASK_IDS[0]
AGENT_LABEL = "CircuitRL agent"
ACTION_LABELS = {"init": "Bench Reset", "r_up": "Increase R", "r_down": "Decrease R", "c_up": "Increase C", "c_down": "Decrease C"}
ACTION_SYMBOLS = {"init": "INIT", "r_up": "R+", "r_down": "R-", "c_up": "C+", "c_down": "C-"}
ACTION_PRIORITY = {"r_down": 4, "c_down": 3, "r_up": 2, "c_up": 1}
CSS = """
:root { --bg:#111514; --panel:rgba(21,26,24,.9); --panel-soft:rgba(27,34,31,.78); --panel-strong:#1b211f; --ink:#f4ede1; --muted:#a9b2aa; --copper:#c47a4d; --copper-soft:rgba(196,122,77,.18); --teal:#6fa29a; --teal-soft:rgba(111,162,154,.18); --line:rgba(255,255,255,.08); --ok:#84c4b2; --warn:#f0b26a; --shadow:0 20px 60px rgba(0,0,0,.28); }
body,.gradio-container { background:radial-gradient(circle at top left, rgba(196,122,77,.16), transparent 28%), radial-gradient(circle at 82% 18%, rgba(111,162,154,.12), transparent 24%), linear-gradient(180deg, #131816 0%, #0d1110 100%); color:var(--ink); font-family:"IBM Plex Sans","Aptos","Segoe UI",sans-serif; }
.gradio-container { max-width:1240px!important; padding:22px 18px 44px!important; }
.block,.gr-group,.gr-box,.gr-panel { border:none!important; background:transparent!important; box-shadow:none!important; }
.hero,.controls-shell,.panel { border:1px solid var(--line); border-radius:22px; box-shadow:var(--shadow); }
.hero { position:relative; overflow:hidden; padding:28px 30px 26px; background:linear-gradient(135deg, rgba(244,237,225,.04), transparent 46%), linear-gradient(180deg, rgba(196,122,77,.06), rgba(21,26,24,.78)), var(--panel); }
.hero::after { content:""; position:absolute; inset:0; background-image:linear-gradient(rgba(255,255,255,.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.04) 1px, transparent 1px); background-size:36px 36px; mask-image:linear-gradient(180deg, rgba(0,0,0,.95), transparent); pointer-events:none; }
.hero-kicker,.panel-title,.spec-item-label,.metric-label,.score-label,.baseline-table th { text-transform:uppercase; letter-spacing:.16em; color:var(--muted); }
.hero-kicker { display:inline-flex; gap:10px; font-size:11px; }
.hero h1 { margin:14px 0 10px; font-size:clamp(2.1rem, 5vw, 4rem); line-height:.98; letter-spacing:-.03em; }
.hero p,.story-copy,.placeholder,.score-note,.panel-subtitle,.metric-meta,.score-meta,.spec-footnote,.chart-legend { color:#d8d2c8; line-height:1.55; }
.hero-band,.status-line,.timeline,.chart-legend,.score-hero { display:flex; flex-wrap:wrap; gap:10px; }
.hero-chip,.chip,.tone-chip,.timeline-chip { display:inline-flex; align-items:center; gap:8px; border-radius:999px; border:1px solid rgba(255,255,255,.1); padding:7px 12px; font-size:12px; letter-spacing:.04em; text-transform:uppercase; background:rgba(255,255,255,.03); }
.controls-shell,.panel { background:linear-gradient(180deg, rgba(255,255,255,.02), transparent 18%), var(--panel-soft); }
.controls-shell { padding:16px 18px 10px; }
.controls-shell button { min-height:48px!important; border-radius:14px!important; background:linear-gradient(135deg, #cc8455, #a85e36)!important; border:none!important; color:#fff6ee!important; font-weight:600!important; }
.controls-shell .wrap,.controls-shell .form { background:transparent!important; }
.controls-shell label,.controls-shell .info { color:var(--muted)!important; }
.controls-shell input,.controls-shell textarea,.controls-shell .wrap-inner,.controls-shell .dropdown,.controls-shell .slider-container { background:rgba(255,255,255,.03)!important; border-color:rgba(255,255,255,.08)!important; }
.panel { padding:20px; }
.panel-head { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:18px; }
.spec-grid,.metric-grid,.score-grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); }
.spec-item,.metric-card,.score-item { border:1px solid rgba(255,255,255,.08); border-radius:18px; padding:16px; background:rgba(255,255,255,.02); }
.spec-item-value,.metric-value,.score-value { display:block; margin-top:10px; font-size:clamp(1.1rem, 3vw, 1.7rem); line-height:1.05; color:var(--ink); }
.mono { font-family:"IBM Plex Mono","Cascadia Code","Consolas",monospace; }
.metric-wide { grid-column:span 2; }
.signal-track,.baseline-bar { overflow:hidden; background:rgba(255,255,255,.06); border-radius:999px; }
.signal-track { margin-top:14px; height:10px; }
.signal-fill,.baseline-bar > span { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg, var(--teal), var(--copper)); }
.tone-chip.ok { background:rgba(132,196,178,.14); color:#d7f4e9; }
.tone-chip.warn { background:rgba(240,178,106,.15); color:#ffe7c9; }
.tone-chip.soft { background:rgba(111,162,154,.14); color:#daefeb; }
.chart-wrap svg { width:100%; height:auto; display:block; }
.story-title { margin:2px 0 10px; font-size:clamp(1.2rem, 3vw, 1.8rem); line-height:1.08; }
.timeline-chip.active { background:var(--copper-soft); border-color:rgba(196,122,77,.5); color:#ffe2cf; }
.timeline-chip.best { background:var(--teal-soft); border-color:rgba(111,162,154,.45); color:#d8efea; }
.score-shell { background:radial-gradient(circle at top right, rgba(196,122,77,.14), transparent 34%), var(--panel-strong); }
.score-big { font-size:clamp(2.3rem, 8vw, 4.2rem); line-height:.95; letter-spacing:-.04em; }
.baseline-table { width:100%; border-collapse:collapse; font-size:.95rem; }
.baseline-table th,.baseline-table td { padding:12px 10px; border-top:1px solid rgba(255,255,255,.08); text-align:left; vertical-align:middle; }
.baseline-table tr:first-child th,.baseline-table tr:first-child td { border-top:none; }
.baseline-table tr.lead { background:rgba(196,122,77,.08); }
.baseline-bar { width:140px; height:8px; margin-bottom:6px; }
.method-tag { display:inline-flex; align-items:center; gap:8px; font-weight:600; }
.method-dot { width:10px; height:10px; border-radius:50%; background:var(--copper); }
.method-dot.baseline { background:var(--teal); }
@media (max-width:900px) { .metric-wide { grid-column:span 1; } .hero { padding:22px 20px; } .panel,.controls-shell { padding:18px 16px; } }
"""


def format_frequency(hz: float) -> str:
    """Format frequencies with readable engineering prefixes."""

    abs_hz = abs(hz)
    if abs_hz >= 1_000_000:
        return f"{hz / 1_000_000:.2f} MHz"
    if abs_hz >= 1_000:
        return f"{hz / 1_000:.2f} kHz"
    return f"{hz:.2f} Hz"


def format_resistance(ohms: float) -> str:
    """Format resistance in engineering units."""

    abs_ohms = abs(ohms)
    if abs_ohms >= 1_000_000:
        return f"{ohms / 1_000_000:.2f} Mohm"
    if abs_ohms >= 1_000:
        return f"{ohms / 1_000:.2f} kohm"
    return f"{ohms:.0f} ohm"


def format_capacitance(farads: float) -> str:
    """Format capacitance in engineering units."""

    abs_farads = abs(farads)
    if abs_farads >= 1:
        return f"{farads:.3f} F"
    if abs_farads >= 1e-3:
        return f"{farads * 1e3:.2f} mF"
    if abs_farads >= 1e-6:
        return f"{farads * 1e6:.2f} uF"
    if abs_farads >= 1e-9:
        return f"{farads * 1e9:.2f} nF"
    return f"{farads * 1e12:.2f} pF"


def format_percent(value: float) -> str:
    """Format normalized values as percentages."""

    return f"{value * 100:.1f}%"


def format_score(value: float) -> str:
    """Format a benchmark score for UI cards."""

    return f"{value:.3f}"


def task_title(task: CircuitTaskSpec) -> str:
    """Build a concise task title."""

    filter_label = "Low-pass" if task.circuit_type == "low_pass" else "High-pass"
    task_flavor = "budget" if "budget" in task.task_id else "low-cost"
    return f"{filter_label} | {format_frequency(task.target_hz)} | {task_flavor}"


def format_evaluations(value: int | None) -> str:
    """Render evaluation counts for the comparison table."""

    return "stepwise" if value is None else str(value)


def render_hero() -> str:
    """Static hero shell for the frontend."""

    return """
    <section class="hero">
      <div class="hero-kicker">CircuitRL <span>precision analog optimization benchmark</span></div>
      <h1>Watch the controller tune the circuit, not just print a score.</h1>
      <p>
        This console plays back the real RC tuning episode step by step. Each move adjusts
        resistor and capacitor values against the benchmark reward, then stacks the resulting
        solution against deterministic random, heuristic, and brute-force baselines.
      </p>
      <div class="hero-band">
        <span class="hero-chip"><strong>Signal</strong> Target-vs-achieved cutoff lock</span>
        <span class="hero-chip"><strong>Telemetry</strong> Error trace across episode steps</span>
        <span class="hero-chip"><strong>Context</strong> Cost-aware decisions with baseline readout</span>
      </div>
    </section>
    """


def build_frame(
    task: CircuitTaskSpec,
    observation: CircuitObservation,
    *,
    step: int,
    action: str,
    reward: float | None,
    best_score: float,
    note: str,
) -> dict[str, Any]:
    """Convert one observation into a chart-friendly playback frame."""

    return {
        "step": step,
        "action": action,
        "action_label": ACTION_LABELS[action],
        "action_symbol": ACTION_SYMBOLS[action],
        "reward": reward,
        "best_score": best_score,
        "note": note,
        "current_r_ohms": observation.current_r_ohms,
        "current_c_farads": observation.current_c_farads,
        "current_hz": observation.current_hz,
        "normalized_error": observation.normalized_error,
        "current_cost": observation.current_cost,
        "remaining_steps": observation.remaining_steps,
        "delta_hz": observation.current_hz - task.target_hz,
        "within_tolerance": observation.normalized_error <= (task.success_tolerance_pct / 100.0),
    }


def describe_transition(
    task: CircuitTaskSpec,
    previous: CircuitObservation,
    current: CircuitObservation,
    action: str,
) -> str:
    """Build a short engineering narrative for one control step."""

    error_before = previous.normalized_error
    error_after = current.normalized_error
    direction = "pull the cutoff down" if action in {"r_up", "c_up"} else "push the cutoff up"
    component = "resistor" if action.startswith("r") else "capacitor"
    improvement = error_before - error_after
    if improvement > 1e-9:
        outcome = (
            f"Error tightens from {format_percent(error_before)} to {format_percent(error_after)}, "
            "so the move pays off immediately."
        )
    elif improvement < -1e-9:
        outcome = (
            f"Error widens to {format_percent(error_after)}, but the controller keeps the move because "
            "the reward still balances cost and remaining step budget."
        )
    else:
        outcome = "The move keeps error flat and mainly shifts the cost profile."
    return f"The controller changes the {component} to {direction}. {outcome}"


def evaluate_candidate_action(
    task: CircuitTaskSpec,
    observation: CircuitObservation,
    *,
    next_step: int,
    action: str,
) -> dict[str, float | str | bool | None]:
    """Score one candidate move using the real benchmark reward."""

    next_r, next_c, action_error = apply_action(
        observation.current_r_ohms,
        observation.current_c_farads,
        action,
        ACTION_SCALE_FACTOR,
        task.min_r_ohms,
        task.max_r_ohms,
        task.min_c_farads,
        task.max_c_farads,
    )
    metrics = evaluate_circuit_state(
        r_ohms=next_r,
        c_farads=next_c,
        target_hz=task.target_hz,
        step_count=next_step,
        max_steps=task.max_steps,
        success_tolerance=SUCCESS_TOLERANCE,
        min_r_ohms=task.min_r_ohms,
        max_r_ohms=task.max_r_ohms,
        min_c_farads=task.min_c_farads,
        max_c_farads=task.max_c_farads,
    )
    return {
        "action": action,
        "reward": float(metrics["reward"]),
        "normalized_error": float(metrics["normalized_error"]),
        "normalized_cost": float(metrics["normalized_cost"]),
        "current_hz": float(metrics["current_hz"]),
        "done": bool(metrics["done"]),
        "action_error": action_error,
    }


def choose_agent_action(task: CircuitTaskSpec, observation: CircuitObservation, *, next_step: int) -> str:
    """Select the strongest next action from the discrete action set."""

    best_action: str | None = None
    best_key: tuple[float, float, float, int] | None = None
    for action in valid_actions():
        candidate = evaluate_candidate_action(task, observation, next_step=next_step, action=action)
        candidate_key = (
            float(candidate["reward"]),
            -float(candidate["normalized_error"]),
            -float(candidate["normalized_cost"]),
            ACTION_PRIORITY[action],
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_action = action
    assert best_action is not None
    return best_action


def build_initial_payload(task_id: str) -> dict[str, Any]:
    """Build the initial task preview before episode playback begins."""

    task = TASKS[task_id]
    env = CircuitEnvironment({task.task_id: task})
    observation = env.reset(task_id)
    frame = build_frame(
        task,
        observation,
        step=0,
        action="init",
        reward=None,
        best_score=0.0,
        note="Bench staged. The controller has the target spec, the starting RC pair, and a full step budget.",
    )
    return {"task": task.model_dump(), "frames": [frame], "summary": None, "comparisons": [], "summary_visible": False}


def build_episode_payload(task_id: str) -> dict[str, Any]:
    """Run one deterministic demo episode and capture the full trajectory."""

    task = TASKS[task_id]
    env = CircuitEnvironment({task.task_id: task})
    observation = env.reset(task_id)
    frames = [
        build_frame(
            task,
            observation,
            step=0,
            action="init",
            reward=None,
            best_score=0.0,
            note="Bench staged. The controller reads the mismatch, then scores all four legal moves against the real reward.",
        )
    ]
    while not env.is_done:
        action = choose_agent_action(task, observation, next_step=env.step_count + 1)
        previous_observation = observation
        observation, reward, done = env.step(CircuitAction(action=action))
        frames.append(
            build_frame(
                task,
                observation,
                step=env.step_count,
                action=action,
                reward=reward,
                best_score=env.score(),
                note=describe_transition(task, previous_observation, observation, action),
            )
        )
        if done:
            break

    state = env.state()
    agent_score = float(env.score())
    comparisons = [
        {
            "baseline_name": "agent",
            "label": AGENT_LABEL,
            "score": agent_score,
            "success": is_success(agent_score),
            "steps_used": state.step_count,
            "evaluations": state.step_count * len(valid_actions()),
            "achieved_hz": float(state.best_hz or observation.current_hz or 0.0),
            "current_r_ohms": float(state.best_r_ohms or observation.current_r_ohms),
            "current_c_farads": float(state.best_c_farads or observation.current_c_farads),
            "normalized_error": float(state.best_normalized_error or observation.normalized_error),
            "normalized_cost": float(state.best_normalized_cost or observation.current_cost),
        },
        {**run_random_baseline(CircuitEnvironment({task.task_id: task}), task_id, seed=7), "label": "Random"},
        {**run_heuristic_baseline(CircuitEnvironment({task.task_id: task}), task_id), "label": "Heuristic"},
        {**run_bruteforce_baseline(task), "label": "Brute-force"},
    ]
    return {
        "task": task.model_dump(),
        "frames": frames,
        "summary": {
            "score": agent_score,
            "success": is_success(agent_score),
            "steps_used": state.step_count,
            "achieved_hz": float(state.best_hz or observation.current_hz or 0.0),
            "best_error": float(state.best_normalized_error or observation.normalized_error),
            "best_cost": float(state.best_normalized_cost or observation.current_cost),
            "best_r_ohms": float(state.best_r_ohms or observation.current_r_ohms),
            "best_c_farads": float(state.best_c_farads or observation.current_c_farads),
        },
        "comparisons": comparisons,
        "summary_visible": False,
    }


def get_task(payload: dict[str, Any]) -> CircuitTaskSpec:
    """Hydrate task metadata from gr.State payloads."""

    return CircuitTaskSpec.model_validate(payload["task"])


def signal_tone(task: CircuitTaskSpec, frame: dict[str, Any]) -> tuple[str, str]:
    """Return UI tone and copy for the active frame."""

    if frame["within_tolerance"]:
        return "ok", f"Within the {task.success_tolerance_pct:.1f}% target band"
    if abs(frame["delta_hz"]) / task.target_hz <= 0.1:
        return "soft", "Near the lock zone"
    return "warn", "Still converging"


def render_target_card(task: CircuitTaskSpec) -> str:
    """Render the target specification card."""

    ideal_product = 1.0 / (2.0 * math.pi * task.target_hz)
    filter_chip = "low-pass" if task.circuit_type == "low_pass" else "high-pass"
    return f"""
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="panel-title">Target Specification</p>
          <h2 class="story-title">{escape(task_title(task))}</h2>
        </div>
        <span class="chip mono">{escape(task.task_id)}</span>
      </div>
      <div class="hero-band" style="margin-top:0;">
        <span class="chip"><strong>Filter</strong> {escape(filter_chip)}</span>
        <span class="chip"><strong>Tolerance</strong> +/-{task.success_tolerance_pct:.1f}%</span>
        <span class="chip"><strong>Action scale</strong> x{ACTION_SCALE_FACTOR:.1f}</span>
      </div>
      <div class="spec-grid" style="margin-top:18px;">
        <div class="spec-item">
          <span class="spec-item-label">Target cutoff</span>
          <span class="spec-item-value mono">{format_frequency(task.target_hz)}</span>
          <div class="spec-footnote">Real task ID from the benchmark catalog.</div>
        </div>
        <div class="spec-item">
          <span class="spec-item-label">Initial pair</span>
          <span class="spec-item-value mono">{format_resistance(task.initial_r_ohms)} / {format_capacitance(task.initial_c_farads)}</span>
          <div class="spec-footnote">Starting point before the agent spends any steps.</div>
        </div>
        <div class="spec-item">
          <span class="spec-item-label">Step budget</span>
          <span class="spec-item-value mono">{task.max_steps} steps</span>
          <div class="spec-footnote">Reward weights accuracy, cost, and step efficiency.</div>
        </div>
        <div class="spec-item">
          <span class="spec-item-label">Target RC product</span>
          <span class="spec-item-value mono">{format_capacitance(ideal_product)} * ohm</span>
          <div class="spec-footnote">Equivalent R x C time constant needed for the requested cutoff.</div>
        </div>
      </div>
    </section>
    """


def render_metric_card(payload: dict[str, Any], active_index: int) -> str:
    """Render live RC telemetry."""

    task = get_task(payload)
    frame = payload["frames"][active_index]
    tone_class, tone_text = signal_tone(task, frame)
    closeness = max(0.0, 1.0 - frame["normalized_error"])
    step_ratio = frame["step"] / max(task.max_steps, 1)
    r_shift = frame["current_r_ohms"] / task.initial_r_ohms
    c_shift = frame["current_c_farads"] / task.initial_c_farads
    r_position = normalize_log_value(frame["current_r_ohms"], task.min_r_ohms, task.max_r_ohms)
    c_position = normalize_log_value(frame["current_c_farads"], task.min_c_farads, task.max_c_farads)
    reward_value = "Staged" if frame["reward"] is None else format_score(float(frame["reward"]))
    return f"""
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="panel-title">Live Telemetry</p>
          <p class="panel-subtitle">Step {frame["step"]} of {task.max_steps} | {escape(frame["action_label"])}</p>
        </div>
        <span class="tone-chip {tone_class}">{escape(tone_text)}</span>
      </div>
      <div class="metric-grid">
        <article class="metric-card">
          <span class="metric-label">Resistor</span>
          <span class="metric-value mono">{format_resistance(frame["current_r_ohms"])}</span>
          <div class="metric-meta">Range position {format_percent(r_position)} | start x{r_shift:.2f}</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Capacitor</span>
          <span class="metric-value mono">{format_capacitance(frame["current_c_farads"])}</span>
          <div class="metric-meta">Range position {format_percent(c_position)} | start x{c_shift:.2f}</div>
        </article>
        <article class="metric-card metric-wide">
          <span class="metric-label">Target vs Achieved</span>
          <div class="status-line">
            <span class="metric-value mono">{format_frequency(task.target_hz)}</span>
            <span class="metric-value mono">{format_frequency(frame["current_hz"])}</span>
          </div>
          <div class="metric-meta">Delta {format_frequency(frame["delta_hz"])} | error {format_percent(frame["normalized_error"])}</div>
          <div class="signal-track"><div class="signal-fill" style="width:{closeness * 100:.1f}%"></div></div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Reward snapshot</span>
          <span class="metric-value mono">{reward_value}</span>
          <div class="metric-meta">Best score so far {format_score(frame["best_score"])}</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Cost / Step budget</span>
          <span class="metric-value mono">{format_percent(frame["current_cost"])}</span>
          <div class="metric-meta">Steps consumed {format_percent(step_ratio)} | {frame["remaining_steps"]} remaining</div>
        </article>
      </div>
    </section>
    """


def render_chart(payload: dict[str, Any], active_index: int) -> str:
    """Render the error-vs-step chart as inline SVG."""

    task = get_task(payload)
    frames = payload["frames"][: active_index + 1]
    width, height, left, right, top, bottom = 780, 320, 54, 20, 22, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    error_values = [frame["normalized_error"] * 100.0 for frame in frames]
    max_error = max(max(error_values, default=0.0), task.success_tolerance_pct, 6.0) * 1.15

    def x_for(step: int) -> float:
        return left + (step / max(task.max_steps, 1)) * plot_width

    def y_for(error_pct: float) -> float:
        ratio = min(max(error_pct / max_error, 0.0), 1.0)
        return top + (1.0 - ratio) * plot_height

    polyline_points = " ".join(f"{x_for(frame['step']):.2f},{y_for(frame['normalized_error'] * 100.0):.2f}" for frame in frames)
    area_points = f"{left:.2f},{height - bottom:.2f} {polyline_points} {x_for(frames[-1]['step']):.2f},{height - bottom:.2f}"
    tolerance_y = y_for(task.success_tolerance_pct)
    grid_lines = []
    for tick in range(5):
        value = (max_error / 4.0) * tick
        y = y_for(value)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="rgba(255,255,255,0.08)" stroke-width="1" />'
            f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" fill="#9ea69f" font-size="12">{value:.0f}%</text>'
        )
    step_ticks = []
    for step in range(task.max_steps + 1):
        if step not in {0, task.max_steps} and step % 2 == 1:
            continue
        step_ticks.append(
            f'<text x="{x_for(step):.2f}" y="{height - 14}" text-anchor="middle" fill="#9ea69f" font-size="12">{step}</text>'
        )
    points_markup = []
    best_score = max(frame["best_score"] for frame in frames)
    for frame in frames:
        active = frame["step"] == active_index
        best = abs(frame["best_score"] - best_score) <= 1e-12 and frame["step"] != 0
        outer = "#c47a4d" if active else "#6fa29a" if best else "#f4ede1"
        radius = 7 if active else 5
        points_markup.append(
            f'<circle cx="{x_for(frame["step"]):.2f}" cy="{y_for(frame["normalized_error"] * 100.0):.2f}" r="{radius}" fill="{outer}" stroke="#111514" stroke-width="3" />'
        )
    active_frame = frames[-1]
    return f"""
    <section class="panel chart-wrap">
      <div class="panel-head">
        <div>
          <p class="panel-title">Error vs Step</p>
          <p class="panel-subtitle">The line extends as playback advances through the episode.</p>
        </div>
        <span class="chip mono">active error {format_percent(active_frame["normalized_error"])}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Error versus step chart">
        <defs>
          <linearGradient id="error-area" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="rgba(111,162,154,0.28)" /><stop offset="100%" stop-color="rgba(111,162,154,0.02)" /></linearGradient>
          <linearGradient id="error-line" x1="0" x2="1" y1="0" y2="0"><stop offset="0%" stop-color="#6fa29a" /><stop offset="100%" stop-color="#c47a4d" /></linearGradient>
        </defs>
        <rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" rx="18" fill="rgba(255,255,255,0.015)" />
        {''.join(grid_lines)}
        <line x1="{left}" y1="{tolerance_y:.2f}" x2="{width - right}" y2="{tolerance_y:.2f}" stroke="#6fa29a" stroke-width="1.5" stroke-dasharray="6 6" />
        <text x="{width - right - 4}" y="{tolerance_y - 8:.2f}" text-anchor="end" fill="#a8d5cc" font-size="12">tolerance {task.success_tolerance_pct:.1f}%</text>
        <polygon points="{area_points}" fill="url(#error-area)" />
        <polyline points="{polyline_points}" fill="none" stroke="url(#error-line)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        {''.join(points_markup)}
        {''.join(step_ticks)}
      </svg>
      <div class="chart-legend">
        <span>active step</span>
        <span>best score so far</span>
        <span>sampled episode point</span>
      </div>
    </section>
    """


def render_story(payload: dict[str, Any], active_index: int) -> str:
    """Render playback narrative and step timeline."""

    task = get_task(payload)
    frame = payload["frames"][active_index]
    frames = payload["frames"]
    step_intro = (
        "The controller is idle and ready to start scoring moves."
        if frame["step"] == 0
        else f"Step {frame['step']} commits {escape(frame['action_label'])} to shape the RC product."
    )
    timeline_markup = []
    best_score = max(sample["best_score"] for sample in frames)
    for sample in frames[: active_index + 1]:
        classes = []
        if sample["step"] == active_index:
            classes.append("active")
        if abs(sample["best_score"] - best_score) <= 1e-12 and sample["step"] != 0:
            classes.append("best")
        class_attr = " ".join(["timeline-chip", *classes])
        timeline_markup.append(
            f'<span class="{class_attr}"><strong>{sample["action_symbol"]}</strong> step {sample["step"]} | {format_percent(sample["normalized_error"])}</span>'
        )
    return f"""
    <section class="panel">
      <div class="panel-head">
        <div>
          <p class="panel-title">Step Playback</p>
          <p class="panel-subtitle">Greedy lookahead over the real benchmark reward.</p>
        </div>
        <span class="chip">{escape(task.circuit_type.replace('_', '-'))}</span>
      </div>
      <h3 class="story-title">{escape(step_intro)}</h3>
      <p class="story-copy">{escape(frame["note"])}</p>
      <div class="timeline">{''.join(timeline_markup)}</div>
    </section>
    """


def render_score_card(payload: dict[str, Any]) -> str:
    """Render the final outcome card."""

    task = get_task(payload)
    summary = payload.get("summary")
    visible = bool(payload.get("summary_visible"))
    if not summary or not visible:
        return """
        <section class="panel score-shell">
          <div class="panel-head"><div><p class="panel-title">Final Score</p><p class="panel-subtitle">Outcome card appears when playback completes.</p></div></div>
          <div class="placeholder">Run the episode to reveal the best score, achieved cutoff, and the final RC pair chosen by the controller.</div>
        </section>
        """
    success_chip = "ok" if summary["success"] else "warn"
    success_label = "Benchmark-clearing" if summary["success"] else "Close, but below pass line"
    return f"""
    <section class="panel score-shell">
      <div class="panel-head">
        <div><p class="panel-title">Final Score</p><p class="panel-subtitle">Best state observed over the full episode.</p></div>
        <span class="tone-chip {success_chip}">{success_label}</span>
      </div>
      <div class="score-hero">
        <div>
          <div class="score-big mono">{format_score(summary["score"])}</div>
          <div class="score-note">Target {format_frequency(task.target_hz)} | achieved {format_frequency(summary["achieved_hz"])} | best error {format_percent(summary["best_error"])}</div>
        </div>
        <div class="chip mono">{summary["steps_used"]} steps used</div>
      </div>
      <div class="score-grid">
        <div class="score-item"><span class="score-label">Best resistor</span><span class="score-value mono">{format_resistance(summary["best_r_ohms"])}</span><div class="score-meta">Chosen along the highest-reward trajectory segment.</div></div>
        <div class="score-item"><span class="score-label">Best capacitor</span><span class="score-value mono">{format_capacitance(summary["best_c_farads"])}</span><div class="score-meta">Pairs with the resistor to land near the requested cutoff.</div></div>
        <div class="score-item"><span class="score-label">Normalized cost</span><span class="score-value mono">{format_percent(summary["best_cost"])}</span><div class="score-meta">Lower is cheaper in the benchmark's log-scaled component space.</div></div>
      </div>
    </section>
    """


def render_baselines(payload: dict[str, Any]) -> str:
    """Render the clean baseline comparison table."""

    comparisons = payload.get("comparisons", [])
    visible = bool(payload.get("summary_visible"))
    if not comparisons or not visible:
        return """
        <section class="panel">
          <div class="panel-head"><div><p class="panel-title">Baseline Comparison</p><p class="panel-subtitle">Random, heuristic, and brute-force results land here after playback.</p></div></div>
          <div class="placeholder">The comparison panel stays hidden until the episode finishes, so judges can watch the tuning story first and read the benchmark stack second.</div>
        </section>
        """
    best_score = max(float(row["score"]) for row in comparisons)
    rows = []
    for row in comparisons:
        is_agent = row["baseline_name"] == "agent"
        row_class = "lead" if is_agent else ""
        dot_class = "method-dot" if is_agent else "method-dot baseline"
        verdict = "pass" if row["success"] else "track"
        steps = "n/a" if row["steps_used"] == 0 else str(row["steps_used"])
        delta = row["score"] - best_score
        delta_copy = "best" if abs(delta) <= 1e-12 else f"{delta:+.3f}"
        rows.append(
            f"""
            <tr class="{row_class}">
              <td><span class="method-tag"><span class="{dot_class}"></span>{escape(str(row['label']))}</span></td>
              <td><div class="baseline-bar"><span style="width:{float(row['score']) * 100:.1f}%"></span></div><div class="mono">{format_score(float(row['score']))}</div></td>
              <td class="mono">{format_frequency(float(row['achieved_hz']))}</td>
              <td class="mono">{format_percent(float(row['normalized_error']))}</td>
              <td class="mono">{steps}</td>
              <td class="mono">{format_evaluations(row['evaluations'])}</td>
              <td class="mono">{delta_copy}</td>
              <td class="mono">{verdict}</td>
            </tr>
            """
        )
    return f"""
    <section class="panel">
      <div class="panel-head">
        <div><p class="panel-title">Baseline Comparison</p><p class="panel-subtitle">Same task, same reward, easy-to-read side-by-side score stack.</p></div>
        <span class="chip mono">judge view</span>
      </div>
      <table class="baseline-table">
        <tr><th>Method</th><th>Score</th><th>Achieved</th><th>Error</th><th>Steps</th><th>Evals</th><th>Delta to Best</th><th>Status</th></tr>
        {''.join(rows)}
      </table>
    </section>
    """


def render_payload(payload: dict[str, Any], active_index: int) -> tuple[str, str, str, str, str, str]:
    """Render all dynamic UI panels for one active playback frame."""

    task = get_task(payload)
    safe_index = max(0, min(active_index, len(payload["frames"]) - 1))
    return (
        render_target_card(task),
        render_metric_card(payload, safe_index),
        render_chart(payload, safe_index),
        render_story(payload, safe_index),
        render_score_card(payload),
        render_baselines(payload),
    )


def scrub_step(step_value: float, payload: dict[str, Any]) -> tuple[str, str, str, str]:
    """Update the live telemetry panels when the user scrubs through the run."""

    safe_index = int(step_value)
    _, metric_html, chart_html, story_html, score_html, _ = render_payload(payload, safe_index)
    return metric_html, chart_html, story_html, score_html


def select_task(task_id: str) -> tuple[dict[str, Any], str, str, str, str, str, str, dict[str, Any]]:
    """Reset the UI to a clean preview for a new task."""

    payload = build_initial_payload(task_id)
    outputs = render_payload(payload, 0)
    return (payload, *outputs, gr.update(value=0, maximum=0))


def play_episode(task_id: str, playback_delay: float):
    """Animate the episode playback frame by frame."""

    payload = build_episode_payload(task_id)
    total_frames = len(payload["frames"])
    for index in range(total_frames):
        payload["summary_visible"] = index == total_frames - 1
        outputs = render_payload(payload, index)
        yield (payload, *outputs, gr.update(value=index, maximum=total_frames - 1))
        if index < total_frames - 1:
            time.sleep(float(playback_delay))


initial_payload = build_initial_payload(DEFAULT_TASK_ID)
initial_target, initial_metrics, initial_chart, initial_story, initial_score, initial_baselines = render_payload(initial_payload, 0)

with gr.Blocks(title="CircuitRL Console", css=CSS) as demo:
    run_state = gr.State(initial_payload)
    gr.HTML(render_hero())
    with gr.Row():
        with gr.Column(scale=4, min_width=280):
            with gr.Group(elem_classes=["controls-shell"]):
                task_selector = gr.Dropdown(
                    choices=TASK_IDS,
                    value=DEFAULT_TASK_ID,
                    label="Benchmark task",
                    info="Real task IDs from the deterministic CircuitRL catalog.",
                )
                playback_speed = gr.Slider(0.15, 0.95, value=0.35, step=0.05, label="Playback cadence (seconds per step)")
                run_button = gr.Button("Run CircuitRL Episode", variant="primary")
                step_scrubber = gr.Slider(
                    minimum=0,
                    maximum=0,
                    value=0,
                    step=1,
                    label="Episode step",
                    info="After playback, drag to inspect any visited step.",
                )
        with gr.Column(scale=8):
            target_card = gr.HTML(value=initial_target)
    metrics_card = gr.HTML(value=initial_metrics)
    with gr.Row():
        with gr.Column(scale=7, min_width=320):
            chart_card = gr.HTML(value=initial_chart)
        with gr.Column(scale=5, min_width=280):
            story_card = gr.HTML(value=initial_story)
    with gr.Row():
        with gr.Column(scale=5, min_width=300):
            score_card = gr.HTML(value=initial_score)
        with gr.Column(scale=7, min_width=340):
            baselines_card = gr.HTML(value=initial_baselines)

    task_selector.change(
        select_task,
        inputs=[task_selector],
        outputs=[run_state, target_card, metrics_card, chart_card, story_card, score_card, baselines_card, step_scrubber],
    )
    run_button.click(
        play_episode,
        inputs=[task_selector, playback_speed],
        outputs=[run_state, target_card, metrics_card, chart_card, story_card, score_card, baselines_card, step_scrubber],
        show_progress="minimal",
    )
    step_scrubber.change(
        scrub_step,
        inputs=[step_scrubber, run_state],
        outputs=[metrics_card, chart_card, story_card, score_card],
    )

demo.queue(default_concurrency_limit=4)

if __name__ == "__main__":
    demo.launch()
