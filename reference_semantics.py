"""Small, dependency-free reference semantics for the G0 oracle."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, sin, sqrt
from typing import Sequence, Tuple


@dataclass(frozen=True)
class GoldenOutput:
    loss: float
    born: Tuple[float, ...]
    selected: Tuple[float, ...]
    prepared_z: Tuple[float, ...]


def golden_oracle(theta: float, x: Sequence[float], target: Sequence[complex | float]) -> GoldenOutput:
    """Evaluate the ideal four-port G0 example from the roadmap.

    The state is ``RY(theta)`` on ``data`` followed by ``H`` on ``aux``. The
    input description is normalized only when its norm is non-zero, matching
    the source contract used by the R1 rule.
    """

    if len(x) != 4 or len(target) != 4:
        raise ValueError("G0 expects vectors of length four")
    c, s = cos(theta / 2.0), sin(theta / 2.0)
    amplitude = (c / sqrt(2), c / sqrt(2), s / sqrt(2), s / sqrt(2))
    born = tuple(value * value for value in amplitude)
    selected = (born[0], born[2])
    loss = sum(abs(complex(value) - complex(reference)) ** 2 for value, reference in zip(amplitude, target))
    norm_sq = sum(float(value) * float(value) for value in x)
    if not isfinite(norm_sq) or norm_sq <= 0:
        raise ValueError("StatePrep(normalize=False) receives a zero or non-finite description")
    normalized = tuple(float(value) / sqrt(norm_sq) for value in x)
    prepared_z = tuple(value * value for value in normalized)
    return GoldenOutput(loss, born, selected, prepared_z)


def selected_event_mass(probabilities: Sequence[float], step: int = 2) -> float:
    """Return selected submass; this is not a marginal or conditional law."""

    if step <= 0:
        raise ValueError("step must be positive")
    return sum(probabilities[::step])


def sample_law(probabilities: Sequence[float], shots: int) -> dict[tuple[int, ...], float]:
    """Enumerate a small iid sample law for regression tests."""

    if shots <= 0:
        raise ValueError("shots must be positive")
    if len(probabilities) ** shots > 100_000:
        raise ValueError("reference enumeration is bounded")
    result: dict[tuple[int, ...], float] = {}

    def visit(prefix: tuple[int, ...]) -> None:
        if len(prefix) == shots:
            probability = 1.0
            for index in prefix:
                probability *= probabilities[index]
            result[prefix] = probability
            return
        for index in range(len(probabilities)):
            visit(prefix + (index,))

    visit(())
    return result
