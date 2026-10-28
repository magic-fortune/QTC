"""Validation of counterexamples against the original input domain."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class WitnessResult:
    status: str
    reason: str
    values: Mapping[str, object]


def verify_nonzero_witness(
    witness: Sequence[float],
    *,
    original_domain: Callable[[Sequence[float]], bool] | None = None,
) -> WitnessResult:
    """Verify a zero-norm witness without silently changing Δorig."""

    values = {"finite": all(isfinite(value) for value in witness), "norm_sq": sum(value * value for value in witness)}
    if original_domain is not None and not original_domain(witness):
        return WitnessResult("Rejected", "WitnessOutsideOriginalDomain", values)
    if not values["finite"]:
        return WitnessResult("Rejected", "WitnessNotFinite", values)
    if values["norm_sq"] != 0:
        return WitnessResult("Rejected", "WitnessIsNotZeroNorm", values)
    return WitnessResult("Verified", "ZeroNormReachableInOriginalDomain", values)
