"""Forward abstract interpretation and use-indexed obligations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Tuple

from .contracts import ContractResolution, ConsumerContract, ConsumerKind, ProofStatus, SpecStatus
from .domains import (
    AbstractState,
    ApproximationDomain,
    ApproximationKind,
    AvailabilityDomain,
    LayoutDomain,
    NormalizationDomain,
    NormalizationKind,
    PhaseDomain,
    PhaseKind,
    RepresentationDomain,
    RepresentationKind,
    state_for_source,
)
from .ir import CoreGraph, Node, Use


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    use_id: str | None
    value_id: str
    predicate: str
    reason: str
    status: ProofStatus = ProofStatus.UNKNOWN
    counterexample: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class UseAnalysis:
    use: Use
    contract: ContractResolution
    proof_status: ProofStatus
    obligations: Tuple[str, ...] = ()
    diagnostics: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisBundle:
    graph: CoreGraph
    states: Mapping[str, AbstractState]
    obligations: Tuple[Obligation, ...]
    uses: Mapping[str, UseAnalysis]
    # A consumer-indexed state after QTC demand refinement.  The ordinary
    # forward state remains available in ``states`` for baseline comparisons;
    # this map prevents one use's demand from being incorrectly shared with a
    # different use of the same merged value.
    use_states: Mapping[str, AbstractState] = field(default_factory=dict)

    def state(self, value_id: str) -> AbstractState:
        return self.states[value_id]

    def state_for_use(self, use_id: str) -> AbstractState:
        result = self.uses[use_id]
        return self.use_states.get(use_id, self.states[result.use.value_id])


def analyze(
    graph: CoreGraph,
    contracts: Mapping[str, ConsumerContract | ContractResolution],
    *,
    input_states: Mapping[str, AbstractState] | None = None,
    source_states: Mapping[str, AbstractState] | None = None,
) -> AnalysisBundle:
    """Analyze a canonical graph and retain each use's contract separately.

    The transfer functions are intentionally conservative. Unknown opcodes are
    represented as numeric values with an explicit obligation instead of being
    guessed into a quantum representation.
    """

    states: dict[str, AbstractState] = dict(input_states or {})
    source_states = source_states or {}
    obligations: list[Obligation] = []
    for node in graph.nodes:
        if node.value_id in states:
            continue
        states[node.value_id] = _transfer(node, states, source_states, obligations)

    use_results: dict[str, UseAnalysis] = {}
    use_states: dict[str, AbstractState] = {}
    for use in graph.uses:
        candidate = contracts.get(use.contract_id or use.consumer_id)
        if candidate is None:
            resolution = ContractResolution.missing("no contract registered for physical boundary")
            use_results[use.use_id] = UseAnalysis(use, resolution, ProofStatus.UNKNOWN, diagnostics=("MissingContract",))
            continue
        resolution = candidate if isinstance(candidate, ContractResolution) else ContractResolution.resolved(candidate)
        if resolution.status is not SpecStatus.RESOLVED:
            use_results[use.use_id] = UseAnalysis(use, resolution, ProofStatus.UNKNOWN, diagnostics=(resolution.status.value,))
            continue
        state = _demand_state_for_use(graph, use, states, resolution.contract)
        use_states[use.use_id] = state
        use_obligations, diagnostics, status = _obligations_for_use(use, state, resolution.contract, obligations)
        use_results[use.use_id] = UseAnalysis(use, resolution, status, tuple(use_obligations), tuple(diagnostics))
    return AnalysisBundle(graph, states, tuple(obligations), use_results, use_states)


def _demand_state_for_use(
    graph: CoreGraph,
    use: Use,
    states: Mapping[str, AbstractState],
    contract: ConsumerContract,
) -> AbstractState:
    """Apply a use-indexed branch demand to a merge node.

    ``demand_branch`` is an explicit contract fact produced by the source
    adapter.  It identifies the branch selected by the consumer, so QTC can
    avoid joining an unrelated branch that has incompatible layout,
    availability, phase, or normalization facts.  Missing or malformed
    demand remains conservative and uses the ordinary forward state.
    """

    state = states[use.value_id]
    selected = contract.metadata.get("demand_branch")
    if selected is None:
        return state
    node = graph.node(use.value_id)
    if node.opcode not in {"merge", "phi", "demand_merge"}:
        return state
    try:
        index = int(selected)
    except (TypeError, ValueError):
        return state
    if index < 0 or index >= len(node.inputs):
        return state
    return states.get(node.inputs[index], state)


def _inherit_measurement_input(base: AbstractState, states: Mapping[str, AbstractState], node: Node) -> AbstractState:
    """Carry normalization and source obligations through a measured value."""

    if not node.inputs or node.inputs[0] not in states:
        return base
    source = states[node.inputs[0]]
    # A measurement node may consume a symbolic circuit/input placeholder.
    # Preserve the producer's declared wire layout and approximation when the
    # placeholder carries no quantum layout of its own.
    layout = source.layout if (source.layout.wire_order or source.layout.index_map) else base.layout
    availability = base.availability if source.availability.may_origins == frozenset({"classical_input"}) else source.availability
    normalization = source.normalization if (
        source.normalization.kind is not NormalizationKind.UNKNOWN
        or RepresentationKind.KET_DESCRIPTION in source.representation.kinds
    ) else base.normalization
    phase = source.phase if source.phase.kind is not PhaseKind.UNKNOWN else base.phase
    return AbstractState(
        representation=base.representation,
        availability=availability,
        phase=phase,
        layout=layout,
        normalization=normalization,
        approximation=base.approximation if source.approximation.kind is ApproximationKind.UNKNOWN else source.approximation,
        cost=source.cost,
        obligations=source.obligations,
        source_ids=source.source_ids | base.source_ids,
    )


def _transfer(
    node: Node,
    states: Mapping[str, AbstractState],
    source_states: Mapping[str, AbstractState],
    obligations: list[Obligation],
) -> AbstractState:
    op = node.opcode
    attrs = node.attributes
    if op in {"input", "placeholder"}:
        return states.get(
            node.value_id,
            AbstractState(
                representation=RepresentationDomain(frozenset({RepresentationKind.NUMERIC}), tuple(attrs.get("shape", ()))),
                availability=AvailabilityDomain(frozenset({"classical_input"}), frozenset({"classical_array"})),
                phase=PhaseDomain(PhaseKind.INVARIANT),
                normalization=NormalizationDomain(NormalizationKind.UNKNOWN),
                source_ids=frozenset({node.value_id}),
            ),
        )
    if op in {"ket", "source_ket"}:
        source_id = str(attrs.get("source_id", node.value_id))
        base = source_states.get(
            source_id,
            state_for_source(
                representation=RepresentationKind.KET_DESCRIPTION,
                wire_order=tuple(attrs.get("wire_order", ())),
                origin="simulator_state",
                classical_array=bool(attrs.get("classical_array", True)),
                preparation_recipe=bool(attrs.get("preparation_recipe", False)),
                normalized=bool(attrs.get("normalized", True)),
                phase=PhaseDomain(PhaseKind.HOMOGENEOUS, frozenset({(1,)})),
                shape=tuple(attrs.get("shape", ())),
            ),
        )
        if attrs.get("zero_witness"):
            base = replace(base, normalization=NormalizationDomain(NormalizationKind.ZERO, witness="source literal zero input"))
        base = replace(base, source_ids=base.source_ids | frozenset({source_id}))
        if node.inputs and node.inputs[0] in states:
            source = states[node.inputs[0]]
            if (
                "preparation_recipe" in source.availability.must_ports
                or RepresentationKind.KET_DESCRIPTION in source.representation.kinds
            ):
                return AbstractState(
                    representation=base.representation,
                    availability=AvailabilityDomain(
                        source.availability.may_origins,
                        source.availability.must_ports | {"preparation_recipe"},
                        source.availability.conditions,
                    ),
                    phase=source.phase,
                    layout=source.layout,
                    normalization=source.normalization,
                    approximation=source.approximation,
                    cost=source.cost,
                    obligations=source.obligations,
                    source_ids=source.source_ids | base.source_ids,
                )
        return base
    if op in {"probs", "probability_source"}:
        base = source_states.get(
            str(attrs.get("source_id", node.value_id)),
            state_for_source(
                representation=RepresentationKind.PROBABILITY_DESCRIPTION,
                wire_order=tuple(attrs.get("wire_order", ())),
                origin="measurement",
                classical_array=True,
                samples=attrs.get("shot_model") == "samples",
                normalized=True,
                phase=PhaseDomain(PhaseKind.INVARIANT),
                approximation=ApproximationDomain(
                    ApproximationKind.FINITE_SHOT if attrs.get("shot_model") == "samples" else ApproximationKind.EXACT,
                    shots=attrs.get("shots"),
                    execution_id=attrs.get("execution_id"),
                ),
                shape=tuple(attrs.get("shape", ())),
            ),
        )
        base = replace(base, source_ids=base.source_ids | frozenset({str(attrs.get("source_id", node.value_id))}))
        return _inherit_measurement_input(base, states, node)
    if op in {"expval", "expectation_source"}:
        base = source_states.get(
            str(attrs.get("source_id", node.value_id)),
            state_for_source(
                representation=RepresentationKind.EXPECTATION_DESCRIPTION,
                wire_order=tuple(attrs.get("wire_order", ())),
                origin="measurement",
                classical_array=True,
                normalized=True,
                phase=PhaseDomain(PhaseKind.INVARIANT),
                approximation=ApproximationDomain(ApproximationKind.EXACT),
                shape=tuple(attrs.get("shape", ())),
            ),
        )
        base = replace(base, source_ids=base.source_ids | frozenset({str(attrs.get("source_id", node.value_id))}))
        return _inherit_measurement_input(base, states, node)
    if op in {"samples", "sample_source"}:
        shots = attrs.get("shots")
        base = source_states.get(
            str(attrs.get("source_id", node.value_id)),
            state_for_source(
                representation=RepresentationKind.PROBABILITY_DESCRIPTION,
                wire_order=tuple(attrs.get("wire_order", ())),
                origin="measurement",
                classical_array=True,
                samples=True,
                normalized=True,
                phase=PhaseDomain(PhaseKind.INVARIANT),
                approximation=ApproximationDomain(ApproximationKind.FINITE_SHOT, shots=shots),
                shape=tuple(attrs.get("shape", ())),
            ),
        )
        base = replace(base, source_ids=base.source_ids | frozenset({str(attrs.get("source_id", node.value_id))}))
        return _inherit_measurement_input(base, states, node)
    inputs = [states.get(value, AbstractState()) for value in node.inputs]
    if op in {"merge", "phi", "demand_merge"} and inputs:
        merged = inputs[0]
        for item in inputs[1:]:
            merged = merged.join(item)
        return merged
    if op in {"abs", "magnitude"} and inputs:
        source = inputs[0]
        return AbstractState(
            representation=RepresentationDomain(frozenset({RepresentationKind.NUMERIC}), source.representation.shape),
            availability=source.availability,
            phase=PhaseDomain(PhaseKind.INVARIANT),
            layout=source.layout,
            normalization=NormalizationDomain(NormalizationKind.UNKNOWN),
            approximation=source.approximation,
            cost=source.cost,
            source_ids=source.source_ids,
        )
    if op in {"abs2", "square_magnitude", "probability_from_ket"} and inputs:
        source = inputs[0]
        normalization = (
            NormalizationDomain(NormalizationKind.PROVEN_ONE)
            if source.normalization.kind is NormalizationKind.PROVEN_ONE
            else NormalizationDomain(NormalizationKind.UNKNOWN)
        )
        return AbstractState(
            representation=RepresentationDomain(frozenset({RepresentationKind.PROBABILITY_DESCRIPTION, RepresentationKind.NUMERIC}), source.representation.shape),
            availability=source.availability,
            phase=PhaseDomain(PhaseKind.INVARIANT),
            layout=LayoutDomain(source.layout.wire_order, source.layout.basis, source.layout.sample_axis, source.layout.index_map, "FULL"),
            normalization=normalization,
            approximation=source.approximation,
            cost=source.cost,
            source_ids=source.source_ids,
        )
    if op in {"norm_sq", "norm_squared"} and inputs:
        source = inputs[0]
        return AbstractState(
            representation=RepresentationDomain(frozenset({RepresentationKind.NUMERIC}), ()),
            availability=source.availability,
            phase=PhaseDomain(PhaseKind.INVARIANT),
            normalization=NormalizationDomain(NormalizationKind.UNKNOWN),
            approximation=source.approximation,
            cost=source.cost,
            source_ids=source.source_ids,
        )
    if op == "sqrt" and inputs:
        return inputs[0]
    if op in {"divide", "normalize_nonzero"} and inputs:
        source = inputs[0]
        predicate = str(attrs.get("predicate", f"norm_sq({node.inputs[0]}) > 0 and finite({node.inputs[0]})"))
        obligation = Obligation(f"ob-{node.value_id}-nonzero", None, node.value_id, predicate, "normalization denominator must be finite and non-zero")
        obligations.append(obligation)
        if attrs.get("zero_witness", False):
            normalization = NormalizationDomain(NormalizationKind.ZERO, witness="source cancellation or literal zero input")
        else:
            normalization = (
                NormalizationDomain(NormalizationKind.PROVEN_ONE)
                if attrs.get("domain_proven", False)
                else NormalizationDomain(NormalizationKind.CONDITIONAL, predicate=predicate)
            )
        return AbstractState(
            representation=RepresentationDomain(frozenset({RepresentationKind.KET_DESCRIPTION, RepresentationKind.NUMERIC}), source.representation.shape),
            availability=AvailabilityDomain(source.availability.may_origins, source.availability.must_ports | {"preparation_recipe"}, source.availability.conditions),
            phase=source.phase,
            layout=source.layout,
            normalization=normalization,
            approximation=source.approximation,
            cost=source.cost,
            obligations=(obligation.obligation_id,),
            source_ids=source.source_ids,
        )
    if op == "linear":
        return _linear(inputs, attrs)
    if op in {"view", "slice", "gather"} and inputs:
        source = inputs[0]
        index_map = tuple((str(key), int(value)) for key, value in attrs.get("index_map", ()))
        mass_kind = str(attrs.get("mass_kind", "SELECTED_SUBMASS"))
        return AbstractState(
            representation=source.representation,
            availability=source.availability,
            phase=source.phase,
            layout=LayoutDomain(source.layout.wire_order, source.layout.basis, source.layout.sample_axis, index_map, mass_kind),
            normalization=source.normalization if mass_kind == "FULL" else NormalizationDomain(NormalizationKind.UNKNOWN),
            approximation=source.approximation,
            cost=source.cost,
            source_ids=source.source_ids,
        )
    if op in {"observer", "registered_observer"} and inputs:
        source = inputs[0]
        return AbstractState(
            representation=RepresentationDomain(frozenset({RepresentationKind.NUMERIC}), tuple(attrs.get("shape", ()))),
            availability=source.availability,
            phase=PhaseDomain(PhaseKind.SENSITIVE if attrs.get("phase_sensitive", True) else PhaseKind.INVARIANT),
            normalization=NormalizationDomain(NormalizationKind.UNKNOWN),
            approximation=source.approximation,
            cost=source.cost,
            source_ids=source.source_ids,
        )
    if op in {"apply_matrix", "matmul"} and inputs:
        source = inputs[-1]
        obligation = Obligation(f"ob-{node.value_id}-unitary", None, node.value_id, "kernel_is_unitary", "matrix application needs a kernel certificate")
        obligations.append(obligation)
        return AbstractState(
            representation=source.representation,
            availability=source.availability,
            phase=source.phase,
            layout=source.layout,
            normalization=NormalizationDomain(NormalizationKind.CONDITIONAL, predicate="kernel_is_unitary"),
            approximation=source.approximation,
            cost=source.cost,
            obligations=(obligation.obligation_id,),
            source_ids=source.source_ids,
        )
    obligation = Obligation(f"ob-{node.value_id}-opcode", None, node.value_id, f"supported_opcode({op})", "unsupported core operation")
    obligations.append(obligation)
    return AbstractState(
        representation=RepresentationDomain(frozenset({RepresentationKind.NUMERIC})),
        availability=AvailabilityDomain(frozenset({"unknown"}), frozenset()),
        phase=PhaseDomain(PhaseKind.UNKNOWN),
        normalization=NormalizationDomain(NormalizationKind.UNKNOWN),
        approximation=ApproximationDomain(ApproximationKind.UNKNOWN),
        obligations=(obligation.obligation_id,),
    )


def _linear(inputs: list[AbstractState], attributes: Mapping[str, object]) -> AbstractState:
    if not inputs:
        return AbstractState()
    phase = inputs[0].phase
    for state in inputs[1:]:
        phase = phase.join(state.phase)
    source = inputs[0]
    shape = source.representation.shape if all(item.representation.shape == source.representation.shape for item in inputs) else ()
    source_ids = frozenset().union(*(item.source_ids for item in inputs))
    coefficients = attributes.get("coefficients", ())
    try:
        coefficients = tuple(float(value) for value in coefficients)
    except (TypeError, ValueError):
        coefficients = ()
    same_source = bool(inputs) and all(
        item.source_ids and item.source_ids == inputs[0].source_ids for item in inputs
    )
    normalization = NormalizationDomain(NormalizationKind.UNKNOWN)
    if same_source and len(coefficients) == len(inputs) and sum(coefficients) == 0:
        normalization = NormalizationDomain(
            NormalizationKind.ZERO,
            witness="same-source linear coefficients cancel",
        )
    return AbstractState(
        representation=RepresentationDomain(frozenset({RepresentationKind.NUMERIC, RepresentationKind.KET_DESCRIPTION}), shape),
        availability=source.availability,
        phase=phase,
        layout=source.layout,
        normalization=normalization,
        approximation=source.approximation,
        cost=source.cost,
        source_ids=source_ids,
    )


def _obligations_for_use(
    use: Use,
    state: AbstractState,
    contract: ConsumerContract,
    obligations: list[Obligation],
) -> tuple[list[str], list[str], ProofStatus]:
    own: list[str] = []
    diagnostics: list[str] = []
    status = ProofStatus.PROVEN
    if contract.requires_classical_array and "classical_array" not in state.availability.must_ports:
        diagnostics.append("ClassicalArrayUnavailable")
        status = ProofStatus.UNKNOWN
    if contract.requires_normalized or contract.kind is ConsumerKind.PREPARE_PURE:
        if state.normalization.kind is NormalizationKind.ZERO:
            diagnostics.append("ZeroNormalizationWitness")
            status = ProofStatus.REFUTED
        elif state.normalization.kind is NormalizationKind.PROVEN_ONE:
            pass
        elif state.normalization.kind is NormalizationKind.CONDITIONAL:
            obligation = Obligation(
                f"ob-{use.use_id}-normalization",
                use.use_id,
                use.value_id,
                state.normalization.predicate,
                "consumer requires a normalized state",
                ProofStatus.CONDITIONAL,
            )
            obligations.append(obligation)
            own.append(obligation.obligation_id)
            status = ProofStatus.CONDITIONAL
            if not contract.allow_guard:
                diagnostics.append("NonZeroGuardRequired")
                status = ProofStatus.UNKNOWN
        else:
            diagnostics.append("NormalizationUnknown")
            status = ProofStatus.UNKNOWN
    if contract.requires_phase_exact and state.phase.kind is not PhaseKind.HOMOGENEOUS:
        diagnostics.append("ExactPhaseEvidenceRequired")
        status = ProofStatus.UNKNOWN
    if contract.kind is ConsumerKind.MEASURE and contract.shots.kind == "analytic":
        if state.approximation.kind not in {ApproximationKind.EXACT, ApproximationKind.UNKNOWN}:
            diagnostics.append("AnalyticObservationUnavailable")
            status = ProofStatus.UNKNOWN
    # A consumer's wire tuple identifies the event coordinates expected by
    # the downstream API.  When both sides state a concrete order, a
    # permutation is a semantic mismatch and can be rejected immediately.
    if contract.wire_order and state.layout.wire_order and tuple(contract.wire_order) != tuple(state.layout.wire_order):
        diagnostics.append("EventIdentityMismatch")
        status = ProofStatus.REFUTED
    if contract.kind is ConsumerKind.MEASURE and state.layout.event_mass_kind == "SELECTED_SUBMASS" and not state.layout.index_map:
        diagnostics.append("EventIdentityUnknown")
        status = ProofStatus.UNKNOWN
    if state.obligations:
        own.extend(state.obligations)
    return own, diagnostics, status
