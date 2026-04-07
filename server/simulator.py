"""Pure deterministic circuit math and state-transition helpers."""

from __future__ import annotations

import math
from typing import Final

ACTION_SCALE_FACTOR = 1.2
WEIGHT_ACCURACY = 0.7
WEIGHT_COST = 0.2
WEIGHT_STEP = 0.1
SUCCESS_TOLERANCE = 0.02

ACTION_MULTIPLIERS: Final[dict[str, tuple[str, float]]] = {
    "r_up": ("R", ACTION_SCALE_FACTOR),
    "r_down": ("R", 1.0 / ACTION_SCALE_FACTOR),
    "c_up": ("C", ACTION_SCALE_FACTOR),
    "c_down": ("C", 1.0 / ACTION_SCALE_FACTOR),
}


def valid_actions() -> tuple[str, ...]:
    """Return the supported discrete action identifiers."""

    return tuple(ACTION_MULTIPLIERS)


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """Clamp one numeric value into its valid inclusive range."""

    return min(max(value, min_value), max_value)


def compute_cutoff_hz(r_ohms: float, c_farads: float) -> float:
    """Return RC cutoff frequency using 1 / (2 * pi * R * C)."""

    if r_ohms <= 0:
        raise ValueError("r_ohms must be > 0")
    if c_farads <= 0:
        raise ValueError("c_farads must be > 0")
    return 1.0 / (2.0 * math.pi * r_ohms * c_farads)


def apply_action(
    r_ohms: float,
    c_farads: float,
    action: str,
    scale_factor: float,
    min_r_ohms: float,
    max_r_ohms: float,
    min_c_farads: float,
    max_c_farads: float,
) -> tuple[float, float, str | None]:
    """Apply one action, clamp the result, and return the updated RC pair."""

    new_r = r_ohms
    new_c = c_farads
    error = None

    if action == "r_up":
        new_r = r_ohms * scale_factor
    elif action == "r_down":
        new_r = r_ohms / scale_factor
    elif action == "c_up":
        new_c = c_farads * scale_factor
    elif action == "c_down":
        new_c = c_farads / scale_factor
    else:
        return r_ohms, c_farads, f"unsupported action: {action}"

    new_r = clamp_value(new_r, min_r_ohms, max_r_ohms)
    new_c = clamp_value(new_c, min_c_farads, max_c_farads)
    return new_r, new_c, error


def compute_normalized_error(current_hz: float, target_hz: float) -> float:
    """Return absolute normalized frequency error."""

    safe_target = max(target_hz, 1e-12)
    return abs(current_hz - target_hz) / safe_target


def normalize_log_value(value: float, min_value: float, max_value: float) -> float:
    """Normalize one value into [0, 1] on a base-10 log scale."""

    safe_min = max(min_value, 1e-12)
    safe_max = max(max_value, safe_min + 1e-12)
    safe_value = max(value, 1e-12)
    log_min = math.log10(safe_min)
    log_max = math.log10(safe_max)
    log_value = math.log10(safe_value)
    span = max(log_max - log_min, 1e-12)
    normalized = (log_value - log_min) / span
    return clamp_value(normalized, 0.0, 1.0)


def compute_normalized_cost(
    r_ohms: float,
    c_farads: float,
    min_r_ohms: float,
    max_r_ohms: float,
    min_c_farads: float,
    max_c_farads: float,
) -> float:
    """Return normalized log-space RC cost in [0, 1]."""

    norm_log_r = normalize_log_value(r_ohms, min_r_ohms, max_r_ohms)
    norm_log_c = normalize_log_value(c_farads, min_c_farads, max_c_farads)
    normalized_cost = 0.5 * norm_log_r + 0.5 * norm_log_c
    return clamp_value(normalized_cost, 0.0, 1.0)


def compute_step_efficiency(step_count: int, max_steps: int) -> float:
    """Return normalized step efficiency in [0, 1]."""

    safe_max_steps = max(max_steps, 1)
    return clamp_value(1.0 - (step_count / safe_max_steps), 0.0, 1.0)


def compute_reward(
    current_hz: float,
    target_hz: float,
    normalized_cost: float,
    step_count: int,
    max_steps: int,
) -> float:
    """Return frozen benchmark reward in [0, 1]."""

    normalized_error = compute_normalized_error(current_hz, target_hz)
    accuracy_score = max(0.0, 1.0 - normalized_error)
    cost_efficiency = 1.0 - clamp_value(normalized_cost, 0.0, 1.0)
    step_efficiency = compute_step_efficiency(step_count, max_steps)
    reward = (
        WEIGHT_ACCURACY * accuracy_score
        + WEIGHT_COST * cost_efficiency
        + WEIGHT_STEP * step_efficiency
    )
    reward = clamp_value(reward, 0.0, 1.0)
    return 1.0 if abs(reward - 1.0) <= 1e-12 else reward


def is_done(
    normalized_error: float,
    step_count: int,
    max_steps: int,
    success_tolerance: float,
) -> bool:
    """Return whether the episode should terminate."""

    return normalized_error <= success_tolerance or step_count >= max_steps


def evaluate_circuit_state(
    r_ohms: float,
    c_farads: float,
    target_hz: float,
    step_count: int,
    max_steps: int,
    success_tolerance: float,
    min_r_ohms: float,
    max_r_ohms: float,
    min_c_farads: float,
    max_c_farads: float,
) -> dict[str, float | bool]:
    """Compute derived circuit metrics for one RC state."""

    current_hz = compute_cutoff_hz(r_ohms, c_farads)
    normalized_error = compute_normalized_error(current_hz, target_hz)
    normalized_cost = compute_normalized_cost(
        r_ohms,
        c_farads,
        min_r_ohms,
        max_r_ohms,
        min_c_farads,
        max_c_farads,
    )
    reward = compute_reward(
        current_hz,
        target_hz,
        normalized_cost,
        step_count,
        max_steps,
    )
    done = is_done(normalized_error, step_count, max_steps, success_tolerance)
    return {
        "current_hz": current_hz,
        "normalized_error": normalized_error,
        "normalized_cost": normalized_cost,
        "reward": reward,
        "done": done,
    }


def gain_db(circuit_type: str, components: dict[str, float], probe_hz: float) -> float:
    """Compute transfer gain (dB) for first-order lowpass/highpass approximation."""

    fc = compute_cutoff_hz(components["R"], components["C"])
    x = probe_hz / max(fc, 1e-12)

    if circuit_type in {"low_pass", "lowpass"}:
        mag = 1.0 / math.sqrt(1.0 + x * x)
    elif circuit_type in {"high_pass", "highpass"}:
        mag = x / math.sqrt(1.0 + x * x)
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

    return 20.0 * math.log10(max(mag, 1e-12))


def cutoff_frequency_hz(components: dict[str, float]) -> float:
    """Backward-compatible wrapper around compute_cutoff_hz."""

    return compute_cutoff_hz(components["R"], components["C"])


def component_cost(components: dict[str, float], bounds: dict[str, tuple[float, float]]) -> float:
    """Backward-compatible wrapper around compute_normalized_cost."""

    return compute_normalized_cost(
        components["R"],
        components["C"],
        bounds["R"][0],
        bounds["R"][1],
        bounds["C"][0],
        bounds["C"][1],
    )
