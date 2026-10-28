"""Certificate bundle schema shared by the compiler and independent checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

from ._serialization import canonical_json, digest
from .compiler import Plan
from .contracts import ConsumerContract, PlanStatus
from .evidence_schemas import EvidenceRef, EvidenceRegistry
from .ir import CoreGraph


@dataclass(frozen=True)
class CertificateClaim:
    use_id: str
    rule: str
    observation: str
    evidence: Tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class CertificateBundle:
    schema: str
    source_ir_digest: str
    original_domain: str
    compile_domain: str
    model: str
    observer_ports: Tuple[str, ...]
    claims: Tuple[CertificateClaim, ...]
    target_plan_digest: str
    dependencies: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_plan(
        cls,
        graph: CoreGraph,
        plan: Plan,
        contracts: Mapping[str, ConsumerContract],
        *,
        original_domain: str,
        compile_domain: str,
        model: str = "ideal_math",
        evidence_by_use: Mapping[str, Tuple[EvidenceRef, ...]] | None = None,
        dependencies: Mapping[str, str] | None = None,
    ) -> "CertificateBundle":
        evidence_by_use = evidence_by_use or {}
        use_contracts = {
            use.use_id: contracts.get(use.use_id) or contracts.get(use.contract_id or "")
            for use in graph.uses
        }
        claims = tuple(
            CertificateClaim(
                use_id=action.use_id,
                rule=action.rule,
                observation=use_contracts.get(action.use_id).observation
                if use_contracts.get(action.use_id) is not None else "",
                evidence=evidence_by_use.get(action.use_id, ()),
            )
            for action in plan.actions
        )
        return cls(
            schema="qtc-bundle-v2",
            source_ir_digest=graph.digest(),
            original_domain=original_domain,
            compile_domain=compile_domain,
            model=model,
            observer_ports=tuple(claim.observation for claim in claims if claim.observation),
            claims=claims,
            target_plan_digest=plan.digest,
            dependencies=dict(dependencies or {}),
        )

    @property
    def digest(self) -> str:
        return digest(self)

    def canonical_json(self) -> str:
        return canonical_json(self)


@dataclass(frozen=True)
class CheckResult:
    status: str
    reasons: Tuple[str, ...] = ()
    checked_claims: int = 0


def check_bundle(
    bundle: CertificateBundle,
    graph: CoreGraph,
    plan: Plan,
    registry: EvidenceRegistry,
) -> CheckResult:
    """Check bundle scope and evidence references without trusting analyzer labels."""

    reasons: list[str] = []
    if plan.status is PlanStatus.BLOCKED:
        reasons.append("BlockedPlanCannotBeCertified")
    if bundle.schema != "qtc-bundle-v2":
        reasons.append("UnsupportedSchema")
    if bundle.source_ir_digest != graph.digest():
        reasons.append("SourceDigestMismatch")
    if bundle.target_plan_digest != plan.digest:
        reasons.append("PlanDigestMismatch")
    if "<" in bundle.source_ir_digest or "<" in bundle.target_plan_digest:
        reasons.append("PlaceholderDigest")
    action_ids = {action.use_id for action in plan.actions}
    actions_by_use = {action.use_id: action for action in plan.actions}
    sample_bindings: dict[str, tuple[object, ...]] = {}
    for action in plan.actions:
        if action.kind == "SampleMeasure" and action.attributes.get("execution_id"):
            identity = action.attributes["execution_id"]
            signature = (action.value_id, action.attributes.get("shots"))
            if identity in sample_bindings and sample_bindings[identity] != signature:
                reasons.append(f"ExecutionIdentityBindingConflict:{action.use_id}")
            sample_bindings[identity] = signature
    claim_ids = {claim.use_id for claim in bundle.claims}
    for missing_use in sorted(action_ids - claim_ids):
        reasons.append(f"MissingClaim:{missing_use}")
    expected_kinds = {
        "KeepOriginalObserver": {"ObserverSummary"},
        "R1-StrictAfterNormalization": {"SourceEvidence"},
        "R2-Born": {"SourceEvidence", "BasisMapCertificate"},
        "R2-Expectation": {"SourceEvidence", "BasisMapCertificate"},
        "R2-SampleLaw": {"SourceEvidence", "BasisMapCertificate"},
        "R3-Select": {"BasisMapCertificate"},
        "CONTROL-EXACT": {"PhaseLiftCertificate"},
    }
    for claim in bundle.claims:
        if claim.use_id not in action_ids:
            reasons.append(f"ClaimWithoutPlan:{claim.use_id}")
        elif claim.rule != actions_by_use[claim.use_id].rule:
            reasons.append(f"ClaimRuleMismatch:{claim.use_id}")
        if claim.rule in {"R1-StrictAfterNormalization", "R2-Born", "R2-Expectation", "R2-SampleLaw", "R3-Select", "CONTROL-EXACT"} and not claim.evidence:
            reasons.append(f"MissingRuleEvidence:{claim.use_id}")
        if claim.rule == "R2-SampleLaw" and claim.use_id in actions_by_use:
            action = actions_by_use[claim.use_id]
            shared = sum(other.kind == "SampleMeasure" and other.value_id == action.value_id for other in plan.actions) > 1
            if (action.attributes.get("execution_identity_required") or shared) and not action.attributes.get("execution_id"):
                reasons.append(f"MissingExecutionIdentity:{claim.use_id}")
        allowed_kinds = expected_kinds.get(claim.rule)
        if allowed_kinds and claim.evidence and not any(reference.kind in allowed_kinds for reference in claim.evidence):
            reasons.append(f"WrongEvidenceKind:{claim.use_id}")
        for reference in claim.evidence:
            if reference.is_placeholder():
                reasons.append(f"PlaceholderEvidence:{claim.use_id}")
            elif registry.resolve(reference) is None:
                reasons.append(f"MissingEvidence:{claim.use_id}:{reference.digest}")
        # A valid digest only proves that the bundle refers to a particular
        # plan and evidence record.  For R3, the observable event semantics
        # also have to agree across those records.  This catches a semantic
        # mutation whose hashes were recomputed after changing a selected
        # submass into a marginal/conditional result.
        if claim.rule == "R3-Select" and claim.use_id in actions_by_use:
            action = actions_by_use[claim.use_id]
            basis_records = [registry.resolve(reference) for reference in claim.evidence]
            basis_records = [record for record in basis_records if record is not None and record.__class__.__name__ == "BasisMapCertificate"]
            if basis_records:
                basis = basis_records[0]
                expected_mass = action.attributes.get("mass_kind")
                if expected_mass and getattr(basis, "mass_kind", None) != expected_mass:
                    reasons.append(f"BasisMassKindMismatch:{claim.use_id}")
                expected_wires = tuple(action.attributes.get("wire_order", ()))
                if expected_wires and tuple(getattr(basis, "ordered_wire_labels", ())) != expected_wires:
                    reasons.append(f"BasisWireOrderMismatch:{claim.use_id}")
    if reasons:
        return CheckResult("Rejected", tuple(reasons), len(bundle.claims))
    return CheckResult("Verified", checked_claims=len(bundle.claims))
