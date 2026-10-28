"""Optional execution adapter for emitted plans.

The adapter is intentionally narrow: plan actions are executed only when a
caller supplies registered QNodes and the required PennyLane package exists.
No action silently falls back to a different observation model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .compiler import Plan, PlanAction
from .contracts import PlanStatus


@dataclass(frozen=True)
class BackendExecution:
    status: str
    outputs: Mapping[str, Any]
    diagnostics: Tuple[str, ...] = ()


class PennyLaneBackend:
    """Execute the small subset of plans represented by registered callables."""

    def __init__(self, qnodes: Mapping[str, Any] | None = None) -> None:
        self.qnodes = dict(qnodes or {})

    @property
    def available(self) -> bool:
        try:
            import pennylane  # noqa: F401
        except Exception:
            return False
        return True

    def execute(self, plan: Plan, bindings: Mapping[str, Any]) -> BackendExecution:
        # A blocked plan is not an executable partial plan.  Refuse before
        # checking the optional runtime or traversing any action list.
        if plan.status is PlanStatus.BLOCKED:
            return BackendExecution("Blocked", {}, tuple(plan.diagnostics) or ("BlockedPlan",))
        sample_bindings: dict[str, tuple[Any, ...]] = {}
        for action in plan.actions:
            if action.kind != "SampleMeasure":
                continue
            execution_id = action.attributes.get("execution_id")
            if not isinstance(execution_id, str) or not execution_id.strip():
                shared = sum(other.kind == "SampleMeasure" and other.value_id == action.value_id for other in plan.actions) > 1
                if action.attributes.get("execution_identity_required") or shared:
                    return BackendExecution("Blocked", {}, ("ExecutionIdentityUnavailable",))
                continue
            qnode = self.qnodes.get(action.value_id) or self.qnodes.get(action.use_id)
            signature = (action.value_id, action.attributes.get("shots"), id(qnode))
            if execution_id in sample_bindings and sample_bindings[execution_id] != signature:
                return BackendExecution("Blocked", {}, ("ExecutionIdentityBindingConflict",))
            sample_bindings[execution_id] = signature
        if not self.available:
            return BackendExecution("Unsupported", {}, ("PennyLane is not installed",))
        outputs: dict[str, Any] = {}
        diagnostics: list[str] = []
        samples: dict[str, Any] = {}
        for action in plan.actions:
            execution_id = action.attributes.get("execution_id") if action.kind == "SampleMeasure" else None
            if execution_id in samples:
                outputs[action.use_id] = samples[execution_id]
                continue
            result = self._execute_action(action, bindings)
            if result.status != "Executed":
                diagnostics.extend(result.diagnostics)
                return BackendExecution(result.status, outputs, tuple(diagnostics))
            outputs.update(result.outputs)
            if execution_id is not None:
                samples[execution_id] = result.outputs[action.use_id]
        return BackendExecution("Executed", outputs, tuple(diagnostics))

    def _execute_action(self, action: PlanAction, bindings: Mapping[str, Any]) -> BackendExecution:
        if action.kind == "KeepOriginal":
            observer = bindings.get(action.value_id)
            if not callable(observer):
                return BackendExecution("Unsupported", {}, (f"No callable observer for {action.value_id}",))
            return BackendExecution("Executed", {action.use_id: observer()})
        if action.kind in {"AnalyticMeasure", "SampleMeasure", "ControlledKernel"}:
            qnode = self.qnodes.get(action.value_id) or self.qnodes.get(action.use_id)
            if qnode is None or not callable(qnode):
                return BackendExecution("Unsupported", {}, (f"No registered QNode for {action.value_id}",))
            args = bindings.get(action.value_id, ())
            if not isinstance(args, tuple):
                args = (args,)
            return BackendExecution("Executed", {action.use_id: qnode(*args)})
        if action.kind == "GatherEvents":
            source = bindings.get(action.value_id)
            if source is None:
                return BackendExecution("Unsupported", {}, (f"No materialized source for {action.value_id}",))
            index_map = tuple(action.attributes.get("index_map", ()))
            # The Core R3 fixture uses ``("j", step)`` to denote the static
            # event family j*step.  A literal key denotes a fixed index.
            if len(index_map) == 1 and str(index_map[0][0]) == "j":
                step = int(index_map[0][1])
                if step <= 0:
                    return BackendExecution("Unsupported", {}, (f"Invalid event-map step {step}",))
                indices = list(range(0, len(source), step))
            else:
                indices = [int(pair[1]) for pair in index_map]
            return BackendExecution("Executed", {action.use_id: tuple(source[index] for index in indices)})
        if action.kind == "PrepareFromDescription":
            qnode = self.qnodes.get(action.value_id) or self.qnodes.get(action.use_id)
            if qnode is None or not callable(qnode):
                return BackendExecution("Unsupported", {}, ("State preparation requires a registered QNode",))
            args = bindings.get(action.value_id, ())
            if not isinstance(args, tuple):
                args = (args,)
            try:
                return BackendExecution("Executed", {action.use_id: qnode(*args)})
            except Exception as exc:
                return BackendExecution("FrameworkError", {}, (f"State preparation failed: {type(exc).__name__}: {exc}",))
        return BackendExecution("Unsupported", {}, (f"Unknown plan action {action.kind}",))
