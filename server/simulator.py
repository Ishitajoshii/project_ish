"""Circuit equations and action application for a simple RC-style benchmark."""

from __future__ import annotations

import math
from typing import Any


def _clamp_component(value: float) -> float:
    """Avoid non-physical negatives and near-zero instability in equations."""

    return max(value, 1e-12)


def apply_action(components: dict[str, float], action: dict[str, Any]) -> dict[str, float]:
    """Return updated components after applying an additive delta to one key."""

    out = dict(components)
    key = action["component"]
    out[key] = _clamp_component(out[key] + float(action["delta"]))
    return out


def cutoff_frequency_hz(components: dict[str, float]) -> float:
    """Compute RC cutoff frequency using f_c = 1/(2*pi*R*C)."""

    r = _clamp_component(components["R"])
    c = _clamp_component(components["C"])
    return 1.0 / (2.0 * math.pi * r * c)


def gain_db(mode: str, components: dict[str, float], probe_hz: float) -> float:
    """Compute transfer gain (dB) for first-order lowpass/highpass approximation."""

    fc = cutoff_frequency_hz(components)
    x = probe_hz / max(fc, 1e-12)

    if mode == "lowpass":
        mag = 1.0 / math.sqrt(1.0 + x * x)
    elif mode == "highpass":
        mag = x / math.sqrt(1.0 + x * x)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return 20.0 * math.log10(max(mag, 1e-12))


def component_cost(components: dict[str, float], weights: dict[str, float]) -> float:
    """Compute linear manufacturing/procurement proxy cost."""

    return sum(float(components[k]) * float(weights.get(k, 0.0)) for k in components)
