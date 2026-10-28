"""Optional PyTorch FX front-end with explicit admission failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from .ir import CoreGraph, Node


@dataclass(frozen=True)
class FrontendDiagnostic:
    status: str
    reason: str
    source_location: str = ""


@dataclass(frozen=True)
class CaptureResult:
    graph: CoreGraph | None
    diagnostics: Tuple[FrontendDiagnostic, ...] = ()


def capture_fx(model: Any, *example_inputs: Any, leaf_modules: Tuple[type, ...] = ()) -> CaptureResult:
    """Capture a simple FX graph when PyTorch is installed.

    FX is an import adapter only. Dynamic control flow and unknown effects are
    reported as diagnostics so a traced constant cannot receive a QTC proof.
    """

    try:
        import torch.fx as fx
    except ImportError:
        return CaptureResult(None, (FrontendDiagnostic("Unsupported", "PyTorch is not installed"),))

    class _Tracer(fx.Tracer):
        def is_leaf_module(self, module: Any, qualified_name: str) -> bool:
            return isinstance(module, leaf_modules) or super().is_leaf_module(module, qualified_name)

    try:
        traced = _Tracer().trace(model)
        graph_module = fx.GraphModule(model, traced)
    except Exception as exc:  # FX has several version-specific tracing errors.
        return CaptureResult(None, (FrontendDiagnostic("Unknown", f"FX capture failed: {exc}"),))
    nodes: list[Node] = []
    inputs: list[str] = []
    outputs: list[str] = []
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.name)
        if node.op == "output":
            outputs.extend(_output_names(node.args[0]))
            continue
        dependencies = tuple(argument.name for argument in node.all_input_nodes)
        nodes.append(Node(node.name, _normalize_opcode(node), dependencies, _safe_attributes(node), str(node)))
    return CaptureResult(CoreGraph(tuple(nodes), outputs=tuple(outputs), inputs=tuple(inputs)))


def _normalize_opcode(node: Any) -> str:
    if node.op == "call_module":
        return "call_module"
    if node.op == "call_method":
        return str(node.target)
    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target))
    return node.op


def _safe_attributes(node: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if node.op in {"call_method", "call_function"}:
        attrs["target"] = str(node.target)
    return attrs


def _output_names(value: Any) -> list[str]:
    if hasattr(value, "name"):
        return [value.name]
    if isinstance(value, (tuple, list)):
        result: list[str] = []
        for item in value:
            result.extend(_output_names(item))
        return result
    return []
