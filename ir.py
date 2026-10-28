"""The small, immutable QTC-Core intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from ._serialization import digest, to_data
from .evidence_schemas import EvidenceRef


@dataclass(frozen=True)
class ParameterBinding:
    name: str
    slot: str
    dtype: str = "float64"
    shape: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ReturnSpec:
    primitive: str
    ordered_wires: Tuple[str | int, ...] = ()
    basis: str = "computational_z"
    shot_model: str = "analytic"


@dataclass(frozen=True)
class AvailabilitySpec:
    """Declared data ports and resource conditions for a source."""

    classical_array: bool = False
    preparation_recipe: bool = False
    samples: bool = False
    conditions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSummary:
    schema_version: int
    source_id: str
    source_kind: str
    primitive: str
    source_code_ref: EvidenceRef
    adapter_ref: EvidenceRef
    kernel_ref: EvidenceRef
    parameter_bindings: Tuple[ParameterBinding, ...]
    return_spec: ReturnSpec
    layout_ref: EvidenceRef
    phase_ref: EvidenceRef
    availability: AvailabilitySpec
    obligations: Tuple[EvidenceRef, ...] = ()

    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class Node:
    value_id: str
    opcode: str
    inputs: Tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source_location: str = ""


@dataclass(frozen=True)
class Use:
    use_id: str
    value_id: str
    consumer_id: str
    contract_id: str | None = None
    output_port: str = ""


@dataclass(frozen=True)
class CoreGraph:
    """Acyclic single-assignment graph after front-end normalization."""

    nodes: Tuple[Node, ...]
    uses: Tuple[Use, ...] = ()
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        ids = [node.value_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("CoreGraph contains duplicate value IDs")
        positions = {value_id: index for index, value_id in enumerate(ids)}
        for node in self.nodes:
            if any(value not in positions for value in node.inputs):
                raise ValueError(f"node {node.value_id} references an unknown input")
            if any(positions[value] >= positions[node.value_id] for value in node.inputs):
                raise ValueError(f"node {node.value_id} violates topological order")
        for use in self.uses:
            if use.value_id not in positions:
                raise ValueError(f"use {use.use_id} references an unknown value")

    @property
    def node_map(self) -> Mapping[str, Node]:
        return {node.value_id: node for node in self.nodes}

    def canonical(self) -> Mapping[str, Any]:
        return to_data(self)

    def digest(self) -> str:
        return digest(self)

    def uses_for(self, value_id: str) -> Tuple[Use, ...]:
        return tuple(use for use in self.uses if use.value_id == value_id)

    def node(self, value_id: str) -> Node:
        try:
            return self.node_map[value_id]
        except KeyError as exc:
            raise KeyError(f"unknown value ID: {value_id}") from exc
