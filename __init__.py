"""QTC: contract-driven analysis for restricted quantum-classical tensor dataflow.

The package deliberately keeps the core independent of PyTorch and PennyLane.
Those frameworks are optional front-end adapters; the IR, abstract domains,
rule compiler, and certificate checker are usable in a small Python runtime.
"""

from .analysis import AnalysisBundle, analyze
from .compiler import BackendProfile, Plan, compile_graph
from .contracts import (
    ConsumerContract,
    ConsumerKind,
    PlanStatus,
    ProofStatus,
    SpecStatus,
)
from .ir import CoreGraph, Node, SourceSummary, Use

__all__ = [
    "AnalysisBundle",
    "BackendProfile",
    "ConsumerContract",
    "ConsumerKind",
    "CoreGraph",
    "Node",
    "Plan",
    "PlanStatus",
    "ProofStatus",
    "SourceSummary",
    "SpecStatus",
    "Use",
    "analyze",
    "compile_graph",
]

__version__ = "0.1.0"
