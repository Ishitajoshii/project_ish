"""Circuit equations and action application for a simple RC-style benchmark."""

from __future__ import annotations

import math
from typing import Any


def _clamp_positive(value: float) -> float:
    """Avoid non-physical negatives and near-zero instability in equations."""

    return max(value, 1e-12)


def apply_action(
    components: dict[str, float],
    action: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Return updated components after applying a multiplicative delta to one key."""

    out = dict(components)
    key = str(action["component"]).upper()
    if key not in {"R", "C"}:
        raise ValueError(f"Unsupported component: {key}")

    delta = float(action["delta"])
    factor = 1.0 + abs(delta)
    updated = out[key]
    if delta > 0.0:
        updated *= factor
    elif delta < 0.0:
        updated /= factor

    lower, upper = bounds[key]
    out[key] = min(max(_clamp_positive(updated), lower), upper)
    return out


def cutoff_frequency_hz(components: dict[str, float]) -> float:
    """Compute RC cutoff frequency using f_c = 1/(2*pi*R*C)."""

    r = _clamp_positive(components["R"])
    c = _clamp_positive(components["C"])
    return 1.0 / (2.0 * math.pi * r * c)


def gain_db(circuit_type: str, components: dict[str, float], probe_hz: float) -> float:
    """Compute transfer gain (dB) for first-order lowpass/highpass approximation."""

    fc = cutoff_frequency_hz(components)
    x = probe_hz / max(fc, 1e-12)

    if circuit_type in {"low_pass", "lowpass"}:
        mag = 1.0 / math.sqrt(1.0 + x * x)
    elif circuit_type in {"high_pass", "highpass"}:
        mag = x / math.sqrt(1.0 + x * x)
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

    return 20.0 * math.log10(max(mag, 1e-12))


def component_cost(components: dict[str, float], bounds: dict[str, tuple[float, float]]) -> float:
    """Compute normalized log-space component cost proxy in the range [0, 1]."""

    normalized_terms = []
    for key in ("R", "C"):
        lower, upper = bounds[key]
        log_lower = math.log10(max(lower, 1e-12))
        log_upper = math.log10(max(upper, 1e-12))
        log_value = math.log10(max(components[key], 1e-12))
        span = max(log_upper - log_lower, 1e-12)
        normalized = (log_value - log_lower) / span
        normalized_terms.append(min(max(normalized, 0.0), 1.0))
    return sum(normalized_terms) / len(normalized_terms)
