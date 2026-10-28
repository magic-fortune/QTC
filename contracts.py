"""Consumer contracts and the three orthogonal QTC result states."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Tuple


class SpecStatus(str, Enum):
    MISSING_CONTRACT = "MissingContract"
    MISSING_SUMMARY = "MissingSummary"
    UNDERSPECIFIED = "Underspecified"
    RESOLVED = "Resolved"
    AMBIGUOUS = "Ambiguous"


class ProofStatus(str, Enum):
    PROVEN = "Proven"
    CONDITIONAL = "Conditional"
    REFUTED = "Refuted"
    UNKNOWN = "Unknown"


class PlanStatus(str, Enum):
    EMIT = "Emit"
    EMIT_GUARDED = "EmitGuarded"
    KEEP_ORIGINAL = "KeepOriginal"
    BLOCKED = "Blocked"


class ConsumerKind(str, Enum):
    CLASSICAL_ANALYSIS = "classical_analysis"
    PREPARE_PURE = "prepare_pure"
    MEASURE = "measure"
    CONTROLLED_USE = "controlled_use"


@dataclass(frozen=True)
class ShotSpec:
    """The observation model requested by a measurement consumer."""

    kind: str = "analytic"
    shots: int | None = None
    epsilon: float | None = None
    delta: float | None = None

    def validate(self) -> None:
        if self.kind not in {"analytic", "samples", "approx"}:
            raise ValueError(f"unsupported shot model: {self.kind}")
        if self.kind == "samples" and (self.shots is None or self.shots <= 0):
            raise ValueError("sample observations require a positive shot count")
        if self.kind == "approx":
            if self.shots is None or self.shots <= 0:
                raise ValueError("approximate observations require a positive shot count")
            if self.epsilon is None or self.epsilon <= 0 or self.delta is None or not 0 < self.delta < 1:
                raise ValueError("approx observations require 0 < epsilon and 0 < delta < 1")


@dataclass(frozen=True)
class ConsumerContract:
    """A resolved physical use contract attached to one IR use."""

    contract_id: str
    kind: ConsumerKind
    observation: str = ""
    requires_normalized: bool = False
    requires_phase_exact: bool = False
    requires_classical_array: bool = False
    preserve_failure: bool = True
    allow_guard: bool = False
    wire_order: Tuple[str | int, ...] = ()
    selected_events: Tuple[Mapping[str | int, int], ...] = ()
    shots: ShotSpec = field(default_factory=ShotSpec)
    required_capabilities: Tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.contract_id:
            raise ValueError("contract_id must not be empty")
        self.shots.validate()
        if self.kind is ConsumerKind.MEASURE and not self.observation:
            raise ValueError("measure contracts require an observation name")
        if self.kind is ConsumerKind.CONTROLLED_USE and not self.requires_phase_exact:
            raise ValueError("controlled_use requires exact unitary phase evidence")


@dataclass(frozen=True)
class ContractResolution:
    """Resolution result kept separate from proof and plan status."""

    status: SpecStatus
    contract: ConsumerContract | None = None
    reason: str = ""
    missing_fields: Tuple[str, ...] = ()

    @classmethod
    def resolved(cls, contract: ConsumerContract) -> "ContractResolution":
        contract.validate()
        return cls(SpecStatus.RESOLVED, contract=contract)

    @classmethod
    def missing(cls, reason: str, *fields: str) -> "ContractResolution":
        return cls(SpecStatus.MISSING_CONTRACT, reason=reason, missing_fields=tuple(fields))

    @classmethod
    def underspecified(cls, reason: str, *fields: str) -> "ContractResolution":
        return cls(SpecStatus.UNDERSPECIFIED, reason=reason, missing_fields=tuple(fields))
