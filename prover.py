"""Small proof utilities with explicit bounds on what QTC can decide."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Tuple

from .contracts import ProofStatus
from .domains import AbstractState, NormalizationKind


@dataclass(frozen=True)
class ProofResult:
    status: ProofStatus
    reason: str
    assumptions: Tuple[str, ...] = ()
    counterexample: Mapping[str, object] = None  # type: ignore[assignment]


def prove_normalization(state: AbstractState) -> ProofResult:
    """Discharge only facts represented by the normalization domain."""

    kind = state.normalization.kind
    if kind is NormalizationKind.PROVEN_ONE:
        return ProofResult(ProofStatus.PROVEN, "norm_sq=1")
    if kind is NormalizationKind.CONDITIONAL:
        return ProofResult(ProofStatus.CONDITIONAL, state.normalization.predicate, (state.normalization.predicate,))
    if kind is NormalizationKind.ZERO:
        return ProofResult(ProofStatus.REFUTED, "zero norm witness", counterexample={"witness": state.normalization.witness})
    return ProofResult(ProofStatus.UNKNOWN, "normalization fact is not represented")


def phase_ast(kind: str, value: int | Fraction | str | None = None, **fields: object) -> dict[str, object]:
    """Build the restricted phase AST accepted by the phase checker."""

    result: dict[str, object] = {"kind": kind}
    if value is not None:
        result["value"] = value
    result.update(fields)
    return result


def evaluate_affine_phase(ast: Mapping[str, object], parameters: Mapping[str, Fraction]) -> Fraction | None:
    """Evaluate the rational affine phase subset; unsupported nodes return None."""

    kind = ast.get("kind")
    if kind == "rational":
        value = ast.get("value")
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int):
            return Fraction(value)
        if isinstance(value, str):
            numerator, denominator = value.split("/", 1) if "/" in value else (value, "1")
            return Fraction(int(numerator), int(denominator))
    if kind == "param":
        name = ast.get("name")
        return parameters.get(str(name))
    if kind == "pi":
        return None
    if kind in {"add", "sub"}:
        left = evaluate_affine_phase(ast.get("left", {}), parameters)  # type: ignore[arg-type]
        right = evaluate_affine_phase(ast.get("right", {}), parameters)  # type: ignore[arg-type]
        if left is None or right is None:
            return None
        return left + right if kind == "add" else left - right
    if kind == "neg":
        value = evaluate_affine_phase(ast.get("value_ast", {}), parameters)  # type: ignore[arg-type]
        return -value if value is not None else None
    if kind == "scale":
        factor = ast.get("factor")
        value = evaluate_affine_phase(ast.get("value_ast", {}), parameters)  # type: ignore[arg-type]
        if value is None or not isinstance(factor, (int, Fraction)):
            return None
        return Fraction(factor) * value
    return None
