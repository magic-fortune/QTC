"""A small structural checker for exact and projective kernel identities."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

from .contracts import ProofStatus
from .evidence_schemas import PhaseLiftCertificate


@dataclass(frozen=True)
class Gate:
    name: str
    wires: Tuple[str | int, ...] = ()
    parameter: str | Fraction | None = None


@dataclass(frozen=True)
class GateWord:
    gates: Tuple[Gate, ...]
    global_phase: Fraction = Fraction(0)

    def normalized(self) -> "GateWord":
        return GateWord(tuple(self.gates), self.global_phase % 2)


@dataclass(frozen=True)
class PhaseCheckResult:
    status: ProofStatus
    reason: str
    correction: str | None = None


def words_equal(left: GateWord, right: GateWord) -> bool:
    return left.gates == right.gates


def residual_phase(left: GateWord, right: GateWord) -> Fraction:
    return (left.global_phase - right.global_phase) % 2


def check_controlled(
    source: GateWord,
    target: GateWord,
    certificate: PhaseLiftCertificate,
    *,
    correction_available: bool = False,
) -> PhaseCheckResult:
    """Check the exact sufficient condition used by CONTROL-EXACT/CORRECT."""

    if not words_equal(source, target):
        return PhaseCheckResult(ProofStatus.UNKNOWN, "NeedOperatorEqualityEvidence")
    phase = residual_phase(source, target)
    if certificate.equality_kind == "EXACT_OPERATOR":
        if phase == 0:
            return PhaseCheckResult(ProofStatus.PROVEN, "ExactOperatorEquality")
        return PhaseCheckResult(ProofStatus.REFUTED, "ExactCertificateHasNonzeroResidualPhase")
    if certificate.equality_kind != "PROJECTIVE_OPERATOR":
        return PhaseCheckResult(ProofStatus.UNKNOWN, "UnknownEqualityKind")
    if certificate.allowed_observation == "UNCONTROLLED_CHANNEL":
        return PhaseCheckResult(ProofStatus.PROVEN, "ProjectiveEqualitySufficientForChannel")
    if not correction_available:
        return PhaseCheckResult(ProofStatus.REFUTED, "ProjectiveEqualityInsufficientForControl")
    correction = "Z(control)" if phase == 1 else f"Phase({phase}) control"
    return PhaseCheckResult(ProofStatus.PROVEN, "ControlledPhaseCorrection", correction)
