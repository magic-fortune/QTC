"""Optional PennyLane source registration with versioned evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from .evidence_schemas import EvidenceRef, EvidenceRegistry, SourceEvidence
from .ir import AvailabilitySpec, ReturnSpec, SourceSummary


@dataclass(frozen=True)
class AdapterResult:
    summary: SourceSummary | None
    evidence_ref: EvidenceRef | None
    diagnostics: Tuple[str, ...] = ()


def inspect_qnode(qnode: Any, registry: EvidenceRegistry, *, source_id: str, wire_order: Tuple[str | int, ...]) -> AdapterResult:
    """Collect a conservative source summary from a fixed QNode template.

    The adapter never treats one concrete tape as proof of all parameterized
    executions. Callers must provide the fixed-template evidence explicitly.
    """

    try:
        import pennylane as qml
    except Exception as exc:
        # Optional framework imports can fail after package discovery (for
        # example, an incompatible jaxlib raises RuntimeError rather than
        # ImportError).  Preserve the explicit unsupported/unknown boundary
        # instead of leaking an environment error through the adapter.
        return AdapterResult(None, None, (f"PennyLane unavailable: {type(exc).__name__}: {exc}",))
    try:
        device = getattr(qnode, "device", None)
        wires = tuple(wire_order or getattr(device, "wires", ()))
        primitive = str(getattr(qnode, "return_type", "unknown"))
        source_evidence = SourceEvidence(
            source_id=source_id,
            source_code_digest="declared",
            qualified_entry=getattr(qnode, "__name__", source_id),
            adapter_id="QTC.PennyLaneAdapter",
            adapter_version=getattr(qml, "__version__", "unknown"),
            adapter_digest="trusted-adapter-declared",
            return_spec={"primitive": primitive, "ordered_wires": wires, "basis": "computational_z", "shot_model": "analytic"},
            derivation_rule="registered_qnode_summary",
        )
        source_ref = registry.add(source_evidence, "SourceEvidence")
        # These references are deliberately explicit. A real implementation
        # populates layout/phase/kernel records before issuing a certificate.
        summary = SourceSummary(
            schema_version=1,
            source_id=source_id,
            source_kind="qnode",
            primitive=primitive,
            source_code_ref=source_ref,
            adapter_ref=source_ref,
            kernel_ref=source_ref,
            parameter_bindings=(),
            return_spec=ReturnSpec(primitive, wires),
            layout_ref=source_ref,
            phase_ref=source_ref,
            availability=getattr(qnode, "_qtc_availability", None) or AvailabilitySpec(classical_array=True),
        )
        return AdapterResult(summary, source_ref)
    except Exception as exc:
        return AdapterResult(None, None, (f"QNode summary unavailable: {exc}",))
