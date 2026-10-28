"""Versioned, hash-bound evidence records used by the independent checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from ._serialization import canonical_json, digest


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    schema_version: int
    digest: str

    def is_placeholder(self) -> bool:
        return not self.digest or "<" in self.digest or "placeholder" in self.digest.lower()


@dataclass(frozen=True)
class EvidenceRecord:
    schema_version: int = 1

    @property
    def record_digest(self) -> str:
        return digest(self)

    def ref(self, kind: str | None = None) -> EvidenceRef:
        return EvidenceRef(kind or type(self).__name__, self.schema_version, self.record_digest)

    def canonical_json(self) -> str:
        return canonical_json(self)


@dataclass(frozen=True)
class SourceEvidence(EvidenceRecord):
    source_id: str = ""
    source_code_digest: str = ""
    qualified_entry: str = ""
    adapter_id: str = ""
    adapter_version: str = ""
    adapter_digest: str = ""
    kernel_gate_ir: Tuple[Mapping[str, Any], ...] = ()
    parameter_bindings: Tuple[Mapping[str, Any], ...] = ()
    return_spec: Mapping[str, Any] = field(default_factory=dict)
    dependency_versions: Mapping[str, str] = field(default_factory=dict)
    derivation_rule: str = ""
    parent_refs: Tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class PhaseCertificate(EvidenceRecord):
    source_ref: EvidenceRef = field(default_factory=lambda: EvidenceRef("SourceEvidence", 1, ""))
    kernel_ref: EvidenceRef = field(default_factory=lambda: EvidenceRef("Kernel", 1, ""))
    phase_kind: str = "multiply_ket"
    angle_ast: Mapping[str, Any] = field(default_factory=dict)
    phase_reference_binding: str = ""
    allowed_observations: Tuple[str, ...] = ()
    derivation_rule: str = ""
    parent_refs: Tuple[EvidenceRef, ...] = ()
    trusted_adapter_ref: EvidenceRef | None = None
    assumptions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseLiftCertificate(EvidenceRecord):
    source_unitary_ref: EvidenceRef = field(default_factory=lambda: EvidenceRef("Kernel", 1, ""))
    target_unitary_ref: EvidenceRef = field(default_factory=lambda: EvidenceRef("Kernel", 1, ""))
    equality_kind: str = "EXACT_OPERATOR"
    residual_angle_ast: Mapping[str, Any] = field(default_factory=dict)
    base_word_equality_evidence: EvidenceRef | None = None
    parameter_domain_ref: EvidenceRef | None = None
    ordered_wire_ref: EvidenceRef | None = None
    allowed_observation: str = "UNCONTROLLED_CHANNEL"
    correction_plan_ref: EvidenceRef | None = None
    derivation_rule: str = ""
    parent_refs: Tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class BasisMapCertificate(EvidenceRecord):
    source_return_spec_ref: EvidenceRef = field(default_factory=lambda: EvidenceRef("ReturnSpec", 1, ""))
    ordered_wire_labels: Tuple[str | int, ...] = ()
    basis_spec: str = "computational_z"
    index_map_ast: Mapping[str, Any] = field(default_factory=dict)
    output_event_predicates: Tuple[Mapping[str, Any], ...] = ()
    mass_kind: str = "FULL"
    derivation_rule: str = ""
    parent_refs: Tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class ObserverSummary(EvidenceRecord):
    observer_source_digest: str = ""
    input_bindings_and_layouts: Tuple[Mapping[str, Any], ...] = ()
    formula_ast: Mapping[str, Any] = field(default_factory=dict)
    numeric_model: str = "ideal_math"
    phase_action_or_requirement: str = ""
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    failure_and_visible_trace_policy: Mapping[str, Any] = field(default_factory=dict)
    derivation_rule: str = ""
    parent_refs: Tuple[EvidenceRef, ...] = ()


@dataclass
class EvidenceRegistry:
    """In-memory evidence store used by compilation and certificate checking."""

    records: dict[str, EvidenceRecord] = field(default_factory=dict)

    def add(self, record: EvidenceRecord, kind: str | None = None) -> EvidenceRef:
        reference = record.ref(kind)
        self.records[reference.digest] = record
        return reference

    def resolve(self, reference: EvidenceRef) -> EvidenceRecord | None:
        record = self.records.get(reference.digest)
        if record is None or record.record_digest != reference.digest:
            return None
        return record

    def require(self, reference: EvidenceRef) -> EvidenceRecord:
        record = self.resolve(reference)
        if record is None:
            raise KeyError(f"evidence not found or digest mismatch: {reference}")
        return record
