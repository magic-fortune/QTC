"""Finite abstract domains used by the QTC forward analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import inf
from typing import FrozenSet, Iterable, Mapping, Tuple


class RepresentationKind(str, Enum):
    NUMERIC = "Numeric"
    KET_DESCRIPTION = "KetDesc"
    DENSITY_DESCRIPTION = "DensityDesc"
    PROBABILITY_DESCRIPTION = "ProbabilityDesc"
    EXPECTATION_DESCRIPTION = "ExpectationDesc"


@dataclass(frozen=True)
class RepresentationDomain:
    kinds: FrozenSet[RepresentationKind] = frozenset()
    shape: Tuple[int | None, ...] = ()

    def join(self, other: "RepresentationDomain") -> "RepresentationDomain":
        shape = self.shape if self.shape == other.shape else ()
        return RepresentationDomain(self.kinds | other.kinds, shape)


@dataclass(frozen=True)
class AvailabilityDomain:
    may_origins: FrozenSet[str] = frozenset()
    must_ports: FrozenSet[str] = frozenset()
    conditions: Tuple[str, ...] = ()

    def join(self, other: "AvailabilityDomain") -> "AvailabilityDomain":
        conditions = tuple(sorted(set(self.conditions) | set(other.conditions)))
        return AvailabilityDomain(self.may_origins | other.may_origins, self.must_ports & other.must_ports, conditions)


class PhaseKind(str, Enum):
    HOMOGENEOUS = "Homogeneous"
    INVARIANT = "Invariant"
    MAY_SENSITIVE = "MaySensitive"
    SENSITIVE = "Sensitive"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class PhaseDomain:
    kind: PhaseKind = PhaseKind.UNKNOWN
    degrees: FrozenSet[Tuple[int, ...]] = frozenset()
    witness: str = ""

    def join(self, other: "PhaseDomain") -> "PhaseDomain":
        if self == other:
            return self
        if self.kind is PhaseKind.UNKNOWN or other.kind is PhaseKind.UNKNOWN:
            return PhaseDomain(PhaseKind.UNKNOWN)
        degrees = self.degrees | other.degrees
        if self.kind is PhaseKind.INVARIANT and other.kind is PhaseKind.INVARIANT:
            return PhaseDomain(PhaseKind.INVARIANT)
        if self.kind is PhaseKind.HOMOGENEOUS and other.kind is PhaseKind.HOMOGENEOUS and self.degrees == other.degrees:
            return self
        return PhaseDomain(PhaseKind.MAY_SENSITIVE, degrees)


@dataclass(frozen=True)
class LayoutDomain:
    wire_order: Tuple[str | int, ...] = ()
    basis: str = "unknown"
    sample_axis: int | None = None
    index_map: Tuple[Tuple[str, int], ...] = ()
    event_mass_kind: str = "UNKNOWN"

    def join(self, other: "LayoutDomain") -> "LayoutDomain":
        return LayoutDomain(
            wire_order=self.wire_order if self.wire_order == other.wire_order else (),
            basis=self.basis if self.basis == other.basis else "unknown",
            sample_axis=self.sample_axis if self.sample_axis == other.sample_axis else None,
            index_map=self.index_map if self.index_map == other.index_map else (),
            event_mass_kind=self.event_mass_kind if self.event_mass_kind == other.event_mass_kind else "UNKNOWN",
        )


class NormalizationKind(str, Enum):
    PROVEN_ONE = "Proven(q=1)"
    PROVEN_POSITIVE = "Proven(q>0)"
    ZERO = "Zero(witness)"
    CONDITIONAL = "Conditional(predicate)"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class NormalizationDomain:
    kind: NormalizationKind = NormalizationKind.UNKNOWN
    predicate: str = ""
    witness: str = ""

    def join(self, other: "NormalizationDomain") -> "NormalizationDomain":
        if self == other:
            return self
        if self.kind is NormalizationKind.PROVEN_ONE and other.kind is NormalizationKind.PROVEN_ONE:
            return self
        return NormalizationDomain(NormalizationKind.UNKNOWN)


class ApproximationKind(str, Enum):
    EXACT = "Exact"
    FINITE_SHOT = "FiniteShot"
    BOUNDED_ERROR = "BoundedError"
    CUTOFF_DEPENDENT = "CutoffDependent"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class ApproximationDomain:
    kind: ApproximationKind = ApproximationKind.UNKNOWN
    shots: int | None = None
    execution_id: str | None = None
    epsilon: float | None = None
    delta: float | None = None

    def join(self, other: "ApproximationDomain") -> "ApproximationDomain":
        return self if self == other else ApproximationDomain(ApproximationKind.UNKNOWN)


@dataclass(frozen=True)
class ResourceBound:
    lower: float = 0.0
    upper: float = inf

    def add(self, other: "ResourceBound") -> "ResourceBound":
        return ResourceBound(self.lower + other.lower, self.upper + other.upper)


@dataclass(frozen=True)
class CostDomain:
    resources: Mapping[str, ResourceBound] = field(default_factory=dict)

    def join(self, other: "CostDomain") -> "CostDomain":
        keys = set(self.resources) | set(other.resources)
        return CostDomain(
            {key: ResourceBound(
                min(self.resources.get(key, ResourceBound()).lower, other.resources.get(key, ResourceBound()).lower),
                max(self.resources.get(key, ResourceBound()).upper, other.resources.get(key, ResourceBound()).upper),
            ) for key in keys}
        )


@dataclass(frozen=True)
class AbstractState:
    """Product state for one SSA value."""

    representation: RepresentationDomain = field(default_factory=RepresentationDomain)
    availability: AvailabilityDomain = field(default_factory=AvailabilityDomain)
    phase: PhaseDomain = field(default_factory=PhaseDomain)
    layout: LayoutDomain = field(default_factory=LayoutDomain)
    normalization: NormalizationDomain = field(default_factory=NormalizationDomain)
    approximation: ApproximationDomain = field(default_factory=ApproximationDomain)
    cost: CostDomain = field(default_factory=CostDomain)
    obligations: Tuple[str, ...] = ()
    # Stable producer identities are kept separately from availability
    # origins (e.g. ``simulator_state``).  This lets relational transfers
    # distinguish two values produced by the same source from two unrelated
    # values with the same coarse origin.
    source_ids: FrozenSet[str] = frozenset()

    def join(self, other: "AbstractState") -> "AbstractState":
        return AbstractState(
            representation=self.representation.join(other.representation),
            availability=self.availability.join(other.availability),
            phase=self.phase.join(other.phase),
            layout=self.layout.join(other.layout),
            normalization=self.normalization.join(other.normalization),
            approximation=self.approximation.join(other.approximation),
            cost=self.cost.join(other.cost),
            obligations=tuple(sorted(set(self.obligations) | set(other.obligations))),
            source_ids=self.source_ids | other.source_ids,
        )


def state_for_source(
    *,
    representation: RepresentationKind,
    wire_order: Iterable[str | int] = (),
    origin: str = "source",
    classical_array: bool = False,
    preparation_recipe: bool = False,
    samples: bool = False,
    normalized: bool = False,
    phase: PhaseDomain | None = None,
    approximation: ApproximationDomain | None = None,
    shape: Tuple[int | None, ...] = (),
) -> AbstractState:
    """Construct a precise source state without filling unsupported facts."""

    ports = set()
    if classical_array:
        ports.add("classical_array")
    if preparation_recipe:
        ports.add("preparation_recipe")
    if samples:
        ports.add("samples")
    return AbstractState(
        representation=RepresentationDomain(frozenset({representation}), shape),
        availability=AvailabilityDomain(frozenset({origin}), frozenset(ports)),
        phase=phase or (
            PhaseDomain(PhaseKind.HOMOGENEOUS, frozenset({(1,)}))
            if representation is RepresentationKind.KET_DESCRIPTION
            else PhaseDomain(PhaseKind.INVARIANT)
        ),
        layout=LayoutDomain(tuple(wire_order), "computational_z", None, (), "FULL"),
        normalization=NormalizationDomain(
            NormalizationKind.PROVEN_ONE if normalized else NormalizationKind.UNKNOWN
        ),
        approximation=approximation or ApproximationDomain(ApproximationKind.EXACT),
    )
