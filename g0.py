"""The dependency-free G0 integration fixture from the technical roadmap."""

from __future__ import annotations

from typing import Mapping, Tuple

from .analysis import AnalysisBundle, analyze
from .compiler import BackendProfile, Plan, compile_graph
from .contracts import ConsumerContract, ConsumerKind
from .domains import AbstractState, RepresentationKind, state_for_source
from .ir import CoreGraph, Node, Use


WIRES = ("data", "aux")


def g0_graph(*, compile_domain_proven: bool = True) -> CoreGraph:
    """Build the normalized CoreGraph without requiring FX or PennyLane."""

    nodes = (
        Node("theta", "input", attributes={"shape": ()}),
        Node("x", "input", attributes={"shape": (4,)}),
        Node("target", "input", attributes={"shape": (4,)}),
        Node("psi", "ket", ("theta",), attributes={"source_id": "state", "wire_order": WIRES, "shape": (4,)}),
        Node("p", "probs", ("theta",), attributes={"source_id": "probs", "wire_order": WIRES, "shape": (4,), "shot_model": "analytic"}),
        Node("loss_input", "linear", ("psi", "target"), attributes={"coefficients": (1, -1)}),
        Node("loss", "observer", ("loss_input",), attributes={"formula": "norm_sq", "phase_sensitive": True}),
        Node("born", "abs2", ("psi",)),
        Node("selected", "view", ("p",), attributes={"index_map": (("j", 2),), "mass_kind": "SELECTED_SUBMASS"}),
        Node("norm_sq_x", "norm_sq", ("x",)),
        Node("norm_x", "sqrt", ("norm_sq_x",)),
        Node("normalized", "divide", ("x", "norm_x"), attributes={"domain_proven": compile_domain_proven}),
    )
    uses = (
        Use("u_loss", "loss", "loss", "c_loss", "loss"),
        Use("u_born", "born", "born", "c_born", "born"),
        Use("u_selected", "selected", "selected", "c_selected", "selected"),
        Use("u_prepare", "normalized", "prepare", "c_prepare", "prepared_z"),
    )
    return CoreGraph(nodes, uses=uses, inputs=("theta", "x", "target"), outputs=("loss", "born", "selected", "normalized"))


def g0_contracts() -> Mapping[str, ConsumerContract]:
    """Return the four explicit G0 consumer contracts."""

    return {
        "c_loss": ConsumerContract("c_loss", ConsumerKind.CLASSICAL_ANALYSIS, "state_loss", requires_classical_array=True),
        "c_born": ConsumerContract("c_born", ConsumerKind.MEASURE, "computational_z", wire_order=WIRES),
        "c_selected": ConsumerContract("c_selected", ConsumerKind.MEASURE, "computational_z", wire_order=WIRES),
        "c_prepare": ConsumerContract("c_prepare", ConsumerKind.PREPARE_PURE, "stateprep", allow_guard=False),
    }


def g0_source_states() -> Mapping[str, AbstractState]:
    return {
        "state": state_for_source(
            representation=RepresentationKind.KET_DESCRIPTION,
            wire_order=WIRES,
            origin="simulator_state",
            classical_array=True,
            normalized=True,
            shape=(4,),
        ),
        "probs": state_for_source(
            representation=RepresentationKind.PROBABILITY_DESCRIPTION,
            wire_order=WIRES,
            origin="measurement",
            classical_array=True,
            normalized=True,
            shape=(4,),
        ),
    }


def build_g0(*, compile_domain_proven: bool = True, backend: BackendProfile | None = None) -> Tuple[AnalysisBundle, Plan]:
    graph = g0_graph(compile_domain_proven=compile_domain_proven)
    analysis = analyze(graph, g0_contracts(), source_states=g0_source_states())
    return analysis, compile_graph(analysis, backend=backend, compile_domain="Gamma_nonzero" if compile_domain_proven else "Delta_original")
