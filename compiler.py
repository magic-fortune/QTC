"""Evidence-guided elaboration of CoreGraph uses into a small Plan IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .analysis import AnalysisBundle, UseAnalysis
from .contracts import ConsumerKind, PlanStatus, ProofStatus, SpecStatus
from .domains import NormalizationKind, RepresentationKind
from ._serialization import digest, to_data


@dataclass(frozen=True)
class BackendProfile:
    name: str
    classical_data_producers: Tuple[str, ...] = ()
    analytic_returns: Tuple[str, ...] = ()
    measurements: Tuple[str, ...] = ()
    gate_set: Tuple[str, ...] = ()
    fresh_zero: bool = False
    phase_exact_control: bool = False
    resource_bounds: Mapping[str, float] = field(default_factory=dict)
    capabilities_known: bool = True

    @classmethod
    def reference(cls) -> "BackendProfile":
        return cls(
            name="ideal_reference",
            classical_data_producers=("ket", "probability", "measurement"),
            analytic_returns=("state", "probs", "expval"),
            measurements=("computational_z", "pauli_product"),
            gate_set=("I", "H", "X", "Y", "Z", "RX", "RY", "RZ", "CNOT", "Phase"),
            fresh_zero=True,
            phase_exact_control=True,
        )

    @classmethod
    def hardware_measurement_only(cls) -> "BackendProfile":
        return cls(
            name="hardware_measurement_only",
            classical_data_producers=("measurement",),
            analytic_returns=(),
            measurements=("computational_z",),
            gate_set=("I", "X", "H", "RX", "RY", "RZ", "CNOT"),
            fresh_zero=True,
            phase_exact_control=False,
        )


@dataclass(frozen=True)
class PlanAction:
    action_id: str
    kind: str
    use_id: str
    value_id: str
    rule: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    graph_digest: str
    backend: str
    status: PlanStatus
    actions: Tuple[PlanAction, ...]
    diagnostics: Tuple[str, ...] = ()
    compile_domain: str = ""

    @property
    def digest(self) -> str:
        return digest(self)

    def canonical(self) -> Mapping[str, Any]:
        return to_data(self)


def compile_graph(
    analysis: AnalysisBundle,
    *,
    backend: BackendProfile | None = None,
    compile_domain: str = "",
) -> Plan:
    """Compile each resolved use independently and aggregate its plan status."""

    backend = backend or BackendProfile.reference()
    actions: list[PlanAction] = []
    diagnostics: list[str] = []
    guarded = False
    blocked = False
    for use_id, result in analysis.uses.items():
        action, status, message = _compile_use(result, analysis, backend)
        if action is not None:
            actions.append(action)
        if status is PlanStatus.EMIT_GUARDED:
            guarded = True
        elif status is PlanStatus.BLOCKED:
            blocked = True
        if message:
            diagnostics.append(f"{use_id}: {message}")
    sample_bindings: dict[str, tuple[Any, ...]] = {}
    for action in actions:
        if action.kind != "SampleMeasure" or not action.attributes.get("execution_id"):
            continue
        identity = action.attributes["execution_id"]
        signature = (action.value_id, action.attributes.get("shots"))
        if identity in sample_bindings and sample_bindings[identity] != signature:
            blocked = True
            diagnostics.append(f"{action.use_id}: ExecutionIdentityBindingConflict")
        sample_bindings[identity] = signature
    status = PlanStatus.BLOCKED if blocked else PlanStatus.EMIT_GUARDED if guarded else PlanStatus.EMIT
    return Plan(analysis.graph.digest(), backend.name, status, tuple(actions), tuple(diagnostics), compile_domain)


def _compile_use(
    result: UseAnalysis,
    analysis: AnalysisBundle,
    backend: BackendProfile,
) -> tuple[PlanAction | None, PlanStatus, str]:
    if result.contract.status is not SpecStatus.RESOLVED or result.contract.contract is None:
        return None, PlanStatus.BLOCKED, result.contract.status.value
    contract = result.contract.contract
    state = analysis.state_for_use(result.use.use_id)
    semantic_error = str(contract.metadata.get("semantic_error", ""))
    if semantic_error:
        return None, PlanStatus.BLOCKED, semantic_error
    if result.proof_status is ProofStatus.REFUTED:
        return None, PlanStatus.BLOCKED, result.diagnostics[0] if result.diagnostics else "ConsumerContractRefuted"
    if contract.kind is ConsumerKind.CLASSICAL_ANALYSIS:
        if result.proof_status is not ProofStatus.PROVEN:
            return None, PlanStatus.BLOCKED, result.diagnostics[0] if result.diagnostics else "ClassicalConsumerProofUnavailable"
        if not backend.capabilities_known:
            return None, PlanStatus.BLOCKED, "BackendCapabilityUnknown"
        # A classical-array consumer needs an actual state/probability array.
        # A measurement-only backend cannot satisfy this by returning samples;
        # the representation and the access port are different contracts.
        if contract.requires_classical_array and not any(
            port in backend.classical_data_producers
            for port in ("ket", "probability")
        ):
            return None, PlanStatus.BLOCKED, "ClassicalArrayCapabilityUnavailable"
        return PlanAction(
            f"a-{result.use.use_id}", "KeepOriginal", result.use.use_id, result.use.value_id,
            "KeepOriginalObserver", {"observation": contract.observation},
        ), PlanStatus.KEEP_ORIGINAL, ""
    if contract.kind is ConsumerKind.PREPARE_PURE:
        if not backend.capabilities_known:
            return None, PlanStatus.BLOCKED, "BackendCapabilityUnknown"
        if not backend.fresh_zero or "preparation_recipe" not in state.availability.must_ports:
            return None, PlanStatus.BLOCKED, "PreparationCapabilityUnavailable"
        if state.normalization.kind is NormalizationKind.ZERO:
            return None, PlanStatus.BLOCKED, "ZeroNormalizationWitness"
        if state.normalization.kind is NormalizationKind.PROVEN_ONE:
            status = PlanStatus.EMIT
        elif state.normalization.kind is NormalizationKind.CONDITIONAL and contract.allow_guard:
            status = PlanStatus.EMIT_GUARDED
        else:
            return None, PlanStatus.BLOCKED, "NonZeroNormalizationObligation"
        return PlanAction(
            f"a-{result.use.use_id}", "PrepareFromDescription", result.use.use_id, result.use.value_id,
            "R1-StrictAfterNormalization", {"fresh_zero": True, "failure_visible": contract.preserve_failure},
        ), status, ""
    if contract.kind is ConsumerKind.MEASURE:
        if not backend.capabilities_known:
            return None, PlanStatus.BLOCKED, "BackendCapabilityUnknown"
        if contract.requires_normalized and result.proof_status is not ProofStatus.PROVEN:
            return None, PlanStatus.BLOCKED, result.diagnostics[0] if result.diagnostics else "MeasurementProofUnavailable"
        if contract.observation not in backend.measurements:
            return None, PlanStatus.BLOCKED, "MeasurementCapabilityUnavailable"
        if contract.shots.kind == "analytic" and not backend.analytic_returns:
            return None, PlanStatus.BLOCKED, "AnalyticReturnUnavailable"
        if state.layout.event_mass_kind == "SELECTED_SUBMASS" and not state.layout.index_map:
            return None, PlanStatus.BLOCKED, "EventIdentityUnknown"
        if state.layout.index_map:
            return PlanAction(
                f"a-{result.use.use_id}", "GatherEvents", result.use.use_id, result.use.value_id,
                "R3-Select", {
                    "index_map": state.layout.index_map,
                    "mass_kind": state.layout.event_mass_kind,
                    "wire_order": state.layout.wire_order,
                },
            ), PlanStatus.EMIT, ""
        if contract.shots.kind == "analytic":
            if contract.observation == "computational_z" and RepresentationKind.PROBABILITY_DESCRIPTION in state.representation.kinds:
                return PlanAction(
                    f"a-{result.use.use_id}", "AnalyticMeasure", result.use.use_id, result.use.value_id,
                    "R2-Born", {"wire_order": state.layout.wire_order},
                ), PlanStatus.EMIT, ""
            if RepresentationKind.EXPECTATION_DESCRIPTION in state.representation.kinds:
                return PlanAction(
                    f"a-{result.use.use_id}", "AnalyticMeasure", result.use.use_id, result.use.value_id,
                    "R2-Expectation", {"wire_order": state.layout.wire_order},
                ), PlanStatus.EMIT, ""
            if RepresentationKind.KET_DESCRIPTION in state.representation.kinds:
                return PlanAction(
                    f"a-{result.use.use_id}", "AnalyticMeasure", result.use.use_id, result.use.value_id,
                    "R2-Born", {"wire_order": state.layout.wire_order},
                ), PlanStatus.EMIT, ""
            return None, PlanStatus.BLOCKED, "BornSourceUnavailable"
        if contract.shots.kind in {"samples", "approx"}:
            if "samples" not in state.availability.must_ports and not backend.measurements:
                return None, PlanStatus.BLOCKED, "SampleCapabilityUnavailable"
            # Joint sampling semantics depend on reusing the same execution.
            # A missing identity is not an implementation detail: it permits
            # an otherwise identical plan to silently resample.
            raw_identity_requirement = contract.metadata.get("execution_identity_required", False)
            requires_identity = (
                raw_identity_requirement is True
                or str(raw_identity_requirement).strip().lower() in {"1", "true", "yes", "required"}
                or contract.observation == "joint_sample_observation"
                or sum(use.value_id == result.use.value_id for use in analysis.graph.uses) > 1
            )
            if requires_identity and not state.approximation.execution_id:
                return None, PlanStatus.BLOCKED, "ExecutionIdentityUnavailable"
            return PlanAction(
                f"a-{result.use.use_id}", "SampleMeasure", result.use.use_id, result.use.value_id,
                "R2-SampleLaw", {
                    "shots": contract.shots.shots,
                    "execution_id": state.approximation.execution_id,
                    "execution_identity_required": requires_identity,
                },
            ), PlanStatus.EMIT, ""
    if contract.kind is ConsumerKind.CONTROLLED_USE:
        if not backend.phase_exact_control:
            return None, PlanStatus.BLOCKED, "ExactControlledPhaseUnavailable"
        if result.proof_status is not ProofStatus.PROVEN:
            return None, PlanStatus.BLOCKED, "ControlledProofNotDischarged"
        return PlanAction(
            f"a-{result.use.use_id}", "ControlledKernel", result.use.use_id, result.use.value_id,
            "CONTROL-EXACT", {"phase_exact": True},
        ), PlanStatus.EMIT, ""
    if state.layout.index_map:
        return PlanAction(
            f"a-{result.use.use_id}", "GatherEvents", result.use.use_id, result.use.value_id,
            "R3-Select", {"index_map": state.layout.index_map, "mass_kind": state.layout.event_mass_kind},
        ), PlanStatus.EMIT, ""
    return None, PlanStatus.BLOCKED, "UnsupportedConsumer"
