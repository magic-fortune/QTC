"""Restricted observation laws and deterministic push-forward helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import prod
from typing import Callable, Generic, Hashable, Iterable, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T", bound=Hashable)
U = TypeVar("U", bound=Hashable)


@dataclass(frozen=True)
class Outcome(Generic[T]):
    kind: str
    payload: T | None = None
    tag: str = ""
    visible_trace: Tuple[Tuple[str, str], ...] = ()

    @classmethod
    def success(cls, payload: T, trace: Mapping[str, str] | None = None) -> "Outcome[T]":
        return cls("Success", payload, visible_trace=tuple(sorted((trace or {}).items())))

    @classmethod
    def failure(cls, tag: str, trace: Mapping[str, str] | None = None) -> "Outcome[T]":
        return cls("Failure", tag=tag, visible_trace=tuple(sorted((trace or {}).items())))


@dataclass(frozen=True)
class DiscreteLaw(Generic[T]):
    """A finite probability mass function used by the reference semantics."""

    probabilities: Mapping[T, float]

    def normalized(self, tolerance: float = 1e-12) -> "DiscreteLaw[T]":
        total = sum(self.probabilities.values())
        if total <= tolerance:
            raise ValueError("a discrete law must have positive mass")
        return DiscreteLaw({key: value / total for key, value in self.probabilities.items()})

    def pushforward(self, function: Callable[[T], U]) -> "DiscreteLaw[U]":
        result: Counter[U] = Counter()
        for value, probability in self.probabilities.items():
            result[function(value)] += probability
        return DiscreteLaw(dict(result))

    def close_to(self, other: "DiscreteLaw[T]", tolerance: float = 1e-12) -> bool:
        keys = set(self.probabilities) | set(other.probabilities)
        return all(abs(self.probabilities.get(key, 0.0) - other.probabilities.get(key, 0.0)) <= tolerance for key in keys)


def dirac(value: T) -> DiscreteLaw[T]:
    return DiscreteLaw({value: 1.0})


def measurement_law(probabilities: Mapping[T, float], shots: int | None = None) -> DiscreteLaw[object]:
    """Return an analytic law or a compact iid sample law descriptor.

    For a finite reference interpreter, analytic laws are represented directly.
    A finite-shot request is represented by ordered sample tuples when the
    support and shot count are small; large requests should stay symbolic in a
    production backend.
    """

    if shots is None:
        return DiscreteLaw(dict(probabilities))
    support = tuple(probabilities)
    if len(support) ** shots > 100_000:
        raise ValueError("finite-shot reference expansion is intentionally bounded")
    sample_law: dict[tuple[object, ...], float] = {}
    for sequence in _product(support, shots):
        probability = prod(probabilities[item] for item in sequence)
        sample_law[sequence] = probability
    return DiscreteLaw(sample_law)


def _product(values: Sequence[T], length: int) -> Iterable[Tuple[T, ...]]:
    if length == 0:
        yield ()
        return
    for prefix in _product(values, length - 1):
        for value in values:
            yield prefix + (value,)
