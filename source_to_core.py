"""Bounded AST source importer used by the C1 source-to-Core evaluation.

This is deliberately a bounded admission front-end. It recognizes explicit
PennyLane return/prepare APIs, imported PennyLane aliases, a small set of
Qiskit/Cirq/TensorFlow-Quantum observation calls, and transparent Torch/JAX
normalization idioms. Unknown Python effects remain diagnostics instead of
being guessed into Core.
The importer is independent of the hand-written G0 adapter.
"""

from __future__ import annotations

import ast
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .contracts import ConsumerContract, ConsumerKind, ContractResolution, ShotSpec
from .evidence_schemas import EvidenceRef, EvidenceRegistry, SourceEvidence
from .ir import CoreGraph, Node, Use
from .source_symbols import SourceSymbols


@dataclass(frozen=True)
class SourceSpan:
    """A source evidence interval used to select one semantic use-site."""

    path: str = ""
    start_line: int = 1
    end_line: int = 10**9
    weight: float = 1


@dataclass(frozen=True)
class SourceUseSite:
    """A source-level quantum boundary candidate and its inferred consumer."""

    qualified_name: str
    api: str
    consumer_kind: str
    line: int
    end_line: int
    source_path: str = ""
    source_location: str = ""
    confidence: str = "direct"
    # Facts are deliberately structured values rather than booleans only.
    # Wire order, static event maps, and normalization predicates must survive
    # the source-to-Core boundary without being collapsed to ``True``.
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceImportResult:
    graph: CoreGraph | None
    contracts: Mapping[str, ConsumerContract | ContractResolution]
    facts: Mapping[str, Any]
    source_ref: EvidenceRef | None
    diagnostics: Tuple[str, ...] = ()
    use_sites: Tuple[SourceUseSite, ...] = ()
    selected_use_site: SourceUseSite | None = None
    parsed_files: Tuple[str, ...] = ()
    skipped_files: Tuple[str, ...] = ()


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        # Preserve a qualified method chain such as
        # ``simulator.run(...).sample_final_state()`` for API recognition.
        return _dotted(node.func)
    return ""


def _literal_bool(node: ast.AST | None, default: bool = False) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return default


_MISSING = object()


def _literal_value(node: ast.AST | None, bindings: Mapping[str, Any] | None = None) -> Any:
    """Evaluate the small literal subset needed for semantic facts.

    This is intentionally narrower than Python evaluation.  It handles values
    commonly used for wires, indices, input vectors, and configuration while
    refusing arbitrary calls or names.  A source fact is only considered
    proven when this function returns a value other than ``_MISSING``.
    """

    bindings = bindings or {}
    if node is None:
        return _MISSING
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.Attribute) and node.attr == "pi":
        return 3.141592653589793
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal_value(node.operand, bindings)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_literal_value(item, bindings) for item in node.elts]
        if all(value is not _MISSING for value in values):
            return set(values) if isinstance(node, ast.Set) else tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        keys = [_literal_value(item, bindings) for item in node.keys]
        values = [_literal_value(item, bindings) for item in node.values]
        if all(key is not _MISSING for key in keys) and all(value is not _MISSING for value in values):
            return dict(zip(keys, values))
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name.endswith(".get") and len(node.args) >= 1:
            owner = _literal_value(node.func.value if isinstance(node.func, ast.Attribute) else None, bindings)
            key = _literal_value(node.args[0], bindings)
            if isinstance(owner, Mapping) and key in owner:
                return owner[key]
            if len(node.args) > 1:
                return _literal_value(node.args[1], bindings)
        if name.rsplit(".", 1)[-1] in {"asarray", "array", "tensor"} and node.args:
            return _literal_value(node.args[0], bindings)
    return _MISSING


def _literal_bindings(region: ast.AST) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    class OrderedBindings(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is not region:
                return
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Assign(self, node: ast.Assign) -> None:
            evaluated = _literal_value(node.value, bindings)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if evaluated is not _MISSING:
                        bindings[target.id] = evaluated
                    else:
                        bindings.pop(target.id, None)
            self.generic_visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            evaluated = _literal_value(node.value, bindings)
            if isinstance(node.target, ast.Name):
                if evaluated is not _MISSING:
                    bindings[node.target.id] = evaluated
                else:
                    bindings.pop(node.target.id, None)
            if node.value is not None:
                self.generic_visit(node.value)

    OrderedBindings().visit(region)
    return bindings


def _same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _norm_expression(node: ast.AST, bindings: Mapping[str, Any]) -> bool:
    if isinstance(node, ast.Name) and "norm" in node.id.lower():
        return True
    if isinstance(node, ast.Call):
        name = _dotted(node.func).lower()
        return (
            name.rsplit(".", 1)[-1] in {"norm", "vector_norm"}
            or name.endswith("linalg.norm")
            or any(_norm_expression(child, bindings) for child in node.args)
        )
    value = _literal_value(node, bindings)
    return isinstance(value, str) and "norm" in value.lower()


def _normalization_facts(
    region: ast.AST,
    inherited_bindings: Mapping[str, Any] | None = None,
    consumer: ast.Call | None = None,
    *,
    all_arguments: bool = False,
) -> dict[str, Any]:
    """Attach denominator facts to reaching values, never unrelated literals.

    This is a bounded local backward slice. Conflicting assignments, unknown
    calls and mutations leave the input domain unknown. Guards must dominate
    the division they protect, rather than merely occur in the same function.
    """
    bindings = dict(inherited_bindings or {})
    if isinstance(region, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in (*region.args.posonlyargs, *region.args.args, *region.args.kwonlyargs):
            bindings.pop(arg.arg, None)
    nodes = list(_region_walk(region))
    parents = {id(child): node for node in nodes for child in ast.iter_child_nodes(node)}
    definitions: dict[str, list[tuple[ast.AST, ast.AST | None]]] = {}
    alias_links: dict[str, set[str]] = {}
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = target.id if isinstance(target, ast.Name) else target.value.id if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) else None
                if name:
                    definitions.setdefault(name, []).append((node, node.value if isinstance(target, ast.Name) and not isinstance(node, ast.AugAssign) else None))
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                        alias_links.setdefault(name, set()).add(node.value.id)
                        alias_links.setdefault(node.value.id, set()).add(name)

    def aliases_of(name: str) -> set[str]:
        pending, found = [name], set()
        while pending:
            item = pending.pop()
            if item not in found:
                found.add(item)
                pending.extend(alias_links.get(item, ()))
        return found

    # Unknown calls may mutate a vector through an alias. Do not reuse an
    # earlier literal as a proof after such a call.
    readonly = {
        "norm", "vector_norm", "sqrt", "abs", "absolute", "sum", "float",
        "int", "len", "print", "array", "asarray", "tensor", "copy", "clone",
        "isfinite", "all", "any", "reshape", "ravel", "tolist",
    }
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, (ast.Subscript, ast.Attribute)) and isinstance(target.value, ast.Name):
                    for alias in aliases_of(target.value.id):
                        definitions.setdefault(alias, []).append((node, None))
        if isinstance(node, ast.Call) and _dotted(node.func).rsplit(".", 1)[-1] not in readonly:
            names = {arg.id for arg in node.args if isinstance(arg, ast.Name)}
            names.update(k.value.id for k in node.keywords if isinstance(k.value, ast.Name))
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                names.add(node.func.value.id)
            for name in names:
                for alias in aliases_of(name):
                    definitions.setdefault(alias, []).append((node, None))

    def uncertain_container(node: ast.AST) -> bool:
        while id(node) in parents:
            node = parents[id(node)]
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)):
                return True
        return False

    def branches(node: ast.AST) -> dict[int, tuple[ast.If, bool]]:
        result = {}
        while id(node) in parents:
            parent = parents[id(node)]
            if isinstance(parent, ast.If):
                result[id(parent)] = (parent, node in parent.body)
            node = parent
        return result

    def resolve(node: ast.AST, before: int, depth: int = 0) -> ast.AST:
        if depth > 20 or not isinstance(node, ast.Name) or uncertain_container(node):
            return node
        choices = [(stmt, value) for stmt, value in definitions.get(node.id, []) if stmt.lineno < before]
        # An assignment from an incompatible branch cannot reach this use.
        use_branches = branches(node)
        choices = [(stmt, value) for stmt, value in choices if all(
            key not in use_branches or side == use_branches[key][1]
            for key, (_, side) in branches(stmt).items()
        )]
        if choices:
            stmt, value = max(choices, key=lambda item: item[0].lineno)
            if value is None or uncertain_container(stmt) or any(key not in use_branches for key in branches(stmt)):
                return node
            return resolve(value, stmt.lineno, depth + 1)
        if node.id in bindings and node.id not in definitions:
            try:
                return ast.parse(repr(bindings[node.id]), mode="eval").body
            except (SyntaxError, ValueError):
                pass
        return node

    def expand(node: ast.AST, before: int, depth: int = 0) -> ast.AST:
        if depth > 20:
            return node
        resolved = resolve(node, before, depth)
        if resolved is not node:
            return expand(resolved, getattr(resolved, "lineno", before), depth + 1)
        return node

    def norm_input(node: ast.AST, before: int, depth: int = 0) -> ast.AST | None:
        if depth > 20:
            return None
        node = expand(node, before, depth)
        if isinstance(node, ast.Call):
            leaf = _dotted(node.func).rsplit(".", 1)[-1]
            if leaf in {"norm", "vector_norm"} and node.args:
                # Non-Euclidean and axis-wise norms require their own summary.
                if len(node.args) > 1 or any(k.arg in {"ord", "axis", "dim"} for k in node.keywords):
                    return None
                return node.args[0]
            if leaf in {"float", "sqrt"} and node.args:
                inner = norm_input(node.args[0], getattr(node, "lineno", before), depth + 1)
                if inner is not None:
                    return inner
                if leaf == "sqrt":
                    square_sum = expand(node.args[0], getattr(node, "lineno", before), depth + 1)
                    if isinstance(square_sum, ast.Call) and _dotted(square_sum.func).rsplit(".", 1)[-1] == "sum" and square_sum.args:
                        squared = square_sum.args[0]
                        if isinstance(squared, ast.BinOp) and isinstance(squared.op, ast.Pow) and _literal_value(squared.right) == 2:
                            magnitude = squared.left
                            if isinstance(magnitude, ast.Call) and _dotted(magnitude.func).rsplit(".", 1)[-1] in {"abs", "absolute"} and magnitude.args:
                                return magnitude.args[0]
        return None

    def vector(node: ast.AST, before: int, depth: int = 0) -> Any:
        if depth > 20:
            return _MISSING
        node = expand(node, before, depth)
        value = _literal_value(node)
        if value is not _MISSING:
            return value
        if isinstance(node, ast.Call) and node.args and _dotted(node.func).rsplit(".", 1)[-1] in {"asarray", "array", "tensor"}:
            return vector(node.args[0], getattr(node, "lineno", before), depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            left = vector(node.left, getattr(node, "lineno", before), depth + 1)
            right = vector(node.right, getattr(node, "lineno", before), depth + 1)
            if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
                if all(isinstance(x, (int, float)) for x in (*left, *right)):
                    return [a - b for a, b in zip(left, right)]
        return _MISSING

    def identity(node: ast.AST, before: int) -> str:
        resolved = expand(node, before)
        key = ast.dump(resolved, include_attributes=False)
        if isinstance(resolved, ast.Name):
            cutoff = min(before, getattr(resolved, "lineno", before))
            writes = [stmt.lineno for stmt, _ in definitions.get(resolved.id, []) if stmt.lineno < cutoff]
            # Equal variable spellings need not denote the same reaching value.
            key += f"@{max(writes, default=0)}"
        return key

    def guarded(division: ast.BinOp) -> bool:
        denominator = identity(division.right, division.lineno)
        for conditional, positive_branch in branches(division).values():
            test = conditional.test
            if not isinstance(test, ast.Compare) or len(test.ops) != 1:
                continue
            if identity(test.left, test.lineno) != denominator or _literal_value(test.comparators[0]) != 0:
                continue
            op = test.ops[0]
            if positive_branch and isinstance(op, (ast.Gt, ast.NotEq)):
                return True
            if not positive_branch and isinstance(op, (ast.LtE, ast.Eq)):
                return True
        return False

    # Restrict explicit state inputs to the expression actually passed to the API.
    relevant: set[int] | None = None
    if consumer is not None:
        relevant = set()

        def visit_value(node: ast.AST, before: int, depth: int = 0) -> None:
            if depth > 20 or id(node) in relevant:
                return
            relevant.add(id(node))
            resolved = expand(node, before, depth)
            if resolved is not node:
                visit_value(resolved, getattr(resolved, "lineno", before), depth + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    visit_value(child, before, depth + 1)

        arguments = (list(consumer.args) + [k.value for k in consumer.keywords]) if all_arguments else consumer.args[:1] or [k.value for k in consumer.keywords if k.arg in {"state", "features", "params", "data"}]
        for arg in arguments:
            visit_value(arg, consumer.lineno)

    facts: dict[str, Any] = {
        "normalization_division": False,
        "nonzero_guard": False,
        "domain_proven": False,
        "zero_norm_witness": False,
        "source_cancellation": False,
        "normalization_dependency_unknown": False,
    }
    proofs, guards = [], []
    for node in nodes:
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div) or relevant is not None and id(node) not in relevant:
            continue
        norm_arg = norm_input(node.right, node.lineno)
        if norm_arg is None:
            if relevant is not None:
                denominator = _literal_value(expand(node.right, node.lineno))
                if not isinstance(denominator, (int, float)) or not math.isfinite(denominator) or denominator == 0:
                    facts["normalization_dependency_unknown"] = True
            continue
        facts["normalization_division"] = True
        values = vector(norm_arg, getattr(norm_arg, "lineno", node.lineno))
        finite = isinstance(values, (list, tuple)) and bool(values) and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values)
        matching = identity(node.left, node.lineno) == identity(norm_arg, getattr(norm_arg, "lineno", node.lineno))
        if finite:
            facts["literal_input"] = values
            facts["zero_norm_witness"] |= all(x == 0 for x in values)
        norm_sq = sum(float(x) * float(x) for x in values) if finite else 0.0
        proofs.append(bool(finite and 0 < norm_sq < math.inf and matching))
        guards.append(guarded(node) and matching)
    facts["domain_proven"] = bool(proofs) and all(proofs)
    facts["nonzero_guard"] = bool(guards) and all(guards)
    facts["guarded_normalization"] = bool(facts["nonzero_guard"])
    return facts


def _consumer_shape_facts(region: ast.AST, inherited_bindings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Recover static event selection and wire declarations without execution."""

    facts: dict[str, Any] = {
        "static_index": False,
        "dynamic_index": False,
        "index_map": (),
        "wire_orders": (),
        "controlled_operation": False,
        "phase_exact_evidence": False,
        "phase_unknown": False,
    }
    bindings = dict(inherited_bindings or {})
    bindings.update(_literal_bindings(region))
    wire_orders: list[tuple[Any, ...]] = []
    index_map: list[tuple[str, int]] = []
    comprehension_indices: dict[int, tuple[int, ...]] = {}
    for candidate in _region_walk(region):
        if not isinstance(candidate, ast.ListComp) or not isinstance(candidate.elt, ast.Subscript):
            continue
        slice_node = candidate.elt.slice
        if not isinstance(slice_node, ast.Name):
            continue
        for generator in candidate.generators:
            if isinstance(generator.target, ast.Name) and generator.target.id == slice_node.id:
                values = _literal_value(generator.iter, bindings)
                if isinstance(values, (list, tuple)) and all(isinstance(item, int) for item in values):
                    comprehension_indices[id(candidate.elt)] = tuple(int(item) for item in values)
    for node in _region_walk(region):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            literal = _literal_value(value, bindings)
            if any(isinstance(target, ast.Name) and "wire" in target.id.lower() for target in targets) and isinstance(literal, (list, tuple)):
                wire_orders.append(tuple(literal))
            if any(isinstance(target, ast.Name) and "execution" in target.id.lower() for target in targets) and literal is None:
                facts["missing_execution_identity"] = True
        if isinstance(node, ast.Subscript):
            value = node.slice
            literal = _literal_value(value, bindings)
            if id(node) in comprehension_indices:
                facts["static_index"] = True
                index_map.extend(("event", item) for item in comprehension_indices[id(node)])
            elif isinstance(literal, int):
                facts["static_index"] = True
                index_map.append(("event", literal))
            elif isinstance(literal, (list, tuple)) and all(isinstance(item, int) for item in literal):
                facts["static_index"] = True
                index_map.extend(("event", int(item)) for item in literal)
            elif isinstance(value, ast.Slice) and all(part is None or isinstance(_literal_value(part, bindings), int) for part in (value.lower, value.upper, value.step)):
                facts["static_index"] = True
            else:
                facts["dynamic_index"] = True
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in {"qargs", "wires", "wire_order", "ordered_wires"}:
                    value = _literal_value(keyword.value, bindings)
                    if isinstance(value, (list, tuple)):
                        wire_orders.append(tuple(value))
                if keyword.arg in {"shots", "num_shots"}:
                    value = _literal_value(keyword.value, bindings)
                    if isinstance(value, int) and value > 0:
                        facts["shots"] = value
            name = _dotted(node.func).lower()
            method = name.rsplit(".", 1)[-1]
            if method in {"cp", "cu1", "crz", "cphase"}:
                facts["controlled_operation"] = True
                if node.args and isinstance(_literal_value(node.args[0], bindings), (int, float)):
                    facts["phase_exact_evidence"] = True
                else:
                    facts["phase_unknown"] = True
            elif method == "control":
                facts["controlled_operation"] = True
                facts["phase_unknown"] = True
    if index_map:
        facts["index_map"] = tuple(index_map)
    if wire_orders:
        facts["wire_orders"] = tuple(wire_orders)
        if len(set(wire_orders)) == 1:
            facts["wire_order"] = wire_orders[0]
        else:
            facts["wire_order_conflict"] = True
    return facts


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mark_source(tree: ast.AST, source_path: str) -> None:
    """Attach the originating file to nodes in a multi-file AST unit."""

    for node in ast.walk(tree):
        setattr(node, "_qtc_source_path", source_path)


def _node_source_path(node: ast.AST) -> str:
    return str(getattr(node, "_qtc_source_path", ""))


def _function_nodes(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Return module, class method, and nested function regions.

    ``import_sources`` combines parseable files into one AST.  Function names
    are not globally unique in that tree, so a plain ``Class.method`` key can
    silently overwrite an earlier file.  Preserve the old names for a single
    file (they are useful in reports and tests), and qualify names by source
    path when the unit contains multiple files.
    """

    regions: list[tuple[str, ast.AST]] = []
    source_paths = {
        _node_source_path(node)
        for node in ast.walk(tree)
        if _node_source_path(node)
    }
    multi_source = len(source_paths) > 1

    def visit(parent: ast.AST, prefix: str) -> None:
        body = getattr(parent, "body", ())
        for child in body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local_name = f"{prefix}.{child.name}" if prefix else child.name
                source_path = _node_source_path(child)
                name = f"{source_path}::{local_name}" if multi_source and source_path else local_name
                regions.append((name, child))
                visit(child, local_name)
            elif isinstance(child, ast.ClassDef):
                local_name = f"{prefix}.{child.name}" if prefix else child.name
                visit(child, local_name)

    if multi_source:
        for path in sorted(source_paths):
            module = ast.Module(body=[node for node in tree.body if _node_source_path(node) == path], type_ignores=[])
            setattr(module, "_qtc_source_path", path)
            regions.append((f"{path}::<module>", module))
    else:
        regions.append(("<module>", tree))
    visit(tree, "")
    return regions


def _region_walk(region: ast.AST) -> tuple[ast.AST, ...]:
    """Walk one function body without importing nested function bodies."""

    nodes: list[ast.AST] = [region]
    result: list[ast.AST] = []
    while nodes:
        node = nodes.pop()
        result.append(node)
        for child in ast.iter_child_nodes(node):
            if child is not region and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            nodes.append(child)
    return tuple(result)


def _short_region_name(name: str) -> str:
    """Return a function name without the source-path qualification."""

    local = name.rsplit("::", 1)[-1]
    return local.rsplit(".", 1)[-1]


def _quantum_name_context(name: str, region_name: str = "") -> bool:
    # Source paths often contain words such as ``quantum`` even when the
    # Python owner is an ordinary helper.  Only use the local region name for
    # custom-boundary heuristics.
    local_region = region_name.rsplit("::", 1)[-1]
    owner = name.rsplit(".", 1)[0] if "." in name else name
    text = f"{owner} {local_region}".lower()
    return any(
        token in text
        for token in (
            "quantum", "qnode", "qnn", "qdevice", "qdev", "circuit", "statevec",
            "state_vector", "statevector", "simulator", "backend", "estimator",
            "sampler", "observable", "expectation", "measure", "probability",
            "strawberry", "fock", "pqc", "qlayer", "quantumdevice", "quantummodule",
        )
    )


def _custom_observation_kind(name: str, region_name: str = "") -> tuple[str, str] | None:
    """Recognize conservative project-defined quantum boundaries.

    This helper deliberately requires a quantum-looking owner or region.  It
    avoids turning ordinary ``run``/``forward``/``sample`` methods into
    quantum observations while still admitting common simulator and wrapper
    classes that do not expose a framework-qualified API.
    """

    leaf = name.rsplit(".", 1)[-1].lower()
    if not _quantum_name_context(name, region_name):
        return None
    if leaf in {"expval", "expect", "expectation", "expectation_value", "expectationvalues", "expval_z"}:
        return name, ConsumerKind.MEASURE.value
    if leaf in {"sample", "samples", "sample_all", "measure", "measure_all", "get_counts", "sample_counts", "sample_memory"}:
        return name, ConsumerKind.MEASURE.value
    if leaf in {"prob", "probs", "probability", "probabilities", "marginal_probabilities"}:
        return name, ConsumerKind.MEASURE.value
    if leaf in {"state", "statevec", "state_vector", "statevector", "density_matrix", "dm", "simulate"}:
        return name, ConsumerKind.CLASSICAL_ANALYSIS.value
    if (
        leaf in {"quantumkernel", "fidelitykernel", "projectedquantumkernel", "singleprobability", "summedprobabilities"}
        or leaf.endswith(("pqc", "qnn", "probability", "probabilities"))
    ):
        return name, ConsumerKind.MEASURE.value
    if leaf in {"run", "execute"} and any(
        token in name.lower()
        for token in ("backend", "simulator", "engine", "sampler", "estimator", "quantum")
    ):
        return name, ConsumerKind.MEASURE.value
    return None


def _observation_kind(name: str, frameworks: set[str]) -> tuple[str, str] | None:
    """Recognize APIs only after resolving their import or receiver type."""

    leaf = name.rsplit(".", 1)[-1]
    if name.startswith("qml."):
        if leaf in {"QubitStateVector", "StatePrep", "AmplitudeEmbedding", "MottonenStatePreparation"}:
            return name, ConsumerKind.PREPARE_PURE.value
        if leaf in {"expval", "sample", "counts", "measure", "probs", "ExpvalCost"} or name.startswith("qml.expval."):
            return name, ConsumerKind.MEASURE.value
        if leaf in {"state", "density_matrix", "fidelity"}:
            return name, ConsumerKind.CLASSICAL_ANALYSIS.value
    if name.startswith("qiskit."):
        if leaf == "initialize":
            return name, ConsumerKind.PREPARE_PURE.value
        if leaf in {
            "Estimator", "EstimatorV2", "BackendEstimator", "BackendEstimatorV2",
            "Sampler", "SamplerV2", "BackendSampler", "BackendSamplerV2",
            "sample_counts", "sample_memory", "probabilities", "expectation_value",
            "measure", "measure_all", "get_counts", "execute", "run_algorithm",
            "QSVM", "TwoLayerQNN", "EstimatorQNN", "SamplerQNN", "VQC", "VQR",
            "NeuralNetworkClassifier", "NeuralNetworkRegressor", "QuantumKernel",
            "QuantumInstance", "FastEstimator", "FastReverseGradientEstimator",
        } or leaf == "run":
            return name, ConsumerKind.MEASURE.value
        if leaf in {"Statevector", "DensityMatrix", "from_instruction", "from_label", "get_statevector"}:
            return name, ConsumerKind.CLASSICAL_ANALYSIS.value
    if name.startswith("cirq."):
        if leaf in {"measure", "sample", "run", "simulate_expectation_values"}:
            return name, ConsumerKind.MEASURE.value
        if leaf in {"simulate", "final_state_vector", "final_density_matrix"}:
            return name, ConsumerKind.CLASSICAL_ANALYSIS.value
    if name.startswith("tfq.") and leaf in {"Expectation", "SampledExpectation", "PQC", "ControlledPQC", "Sample"}:
        return name, ConsumerKind.MEASURE.value
    if name.startswith("tq."):
        if leaf in {"MeasureAll", "expval"}:
            return name, ConsumerKind.MEASURE.value
        if leaf in {"get_states_1d", "get_states"}:
            return name, ConsumerKind.CLASSICAL_ANALYSIS.value
    # These frameworks expose operators, mixed states, and non-qubit returns.
    # Discover the boundary but retain a missing representation obligation.
    if name.startswith("qutip.") and leaf in {"expect", "Qobj", "basis", "tensor", "full", "ptrace"}:
        return name, ConsumerKind.CLASSICAL_ANALYSIS.value
    if name.startswith("strawberryfields.") and leaf in {"run", "dm", "trace", "fidelity"}:
        return name, ConsumerKind.CLASSICAL_ANALYSIS.value
    if name.startswith("pulser.") and leaf == "sample_final_state":
        return name, ConsumerKind.MEASURE.value
    return None


def _api_facts(api: str, consumer_kind: str, call: ast.Call) -> dict[str, Any]:
    """A single mapping for direct calls and calls through bound objects."""

    leaf = api.rsplit(".", 1)[-1]
    unresolved = (
        api.startswith(("qutip.", "strawberryfields."))
        or leaf in {"QuantumInstance", "QuantumKernel", "QSVM", "DensityMatrix", "density_matrix", "final_density_matrix",
                    "VQC", "VQR", "NeuralNetworkClassifier", "NeuralNetworkRegressor"}
    )
    return {
        "state_prep": consumer_kind == ConsumerKind.PREPARE_PURE.value,
        "state_return": leaf in {
            "state", "Statevector", "DensityMatrix", "density_matrix", "from_instruction", "from_label", "simulate",
            "final_state_vector", "get_statevector", "get_states_1d", "get_states",
        },
        "expval_observation": (
            leaf in {"expval", "expectation_value", "simulate_expectation_values", "Expectation",
                     "PQC", "ControlledPQC", "ExpvalCost", "TwoLayerQNN", "EstimatorQNN"}
            or "Estimator" in leaf or api.startswith("qml.expval.")
            or (leaf == "run" and "Estimator" in api)
        ),
        "sample_observation": (
            leaf in {"sample", "counts", "measure", "sample_counts", "sample_memory",
                     "measure_all", "get_counts", "execute", "run_algorithm", "Sample", "sample_final_state"}
            or ("Sampler" in leaf and leaf != "SamplerQNN")
            or (leaf == "run" and "Estimator" not in api and not api.startswith("strawberryfields."))
        ),
        "probs_return": leaf in {"probs", "probabilities", "SamplerQNN"},
        # TorchQuantum MeasureAll(PauliZ) returns expectations, not samples.
        "classical_array_consumer": leaf == "fidelity",
        "amplitude_embedding": leaf == "AmplitudeEmbedding",
        "explicit_normalize": leaf == "AmplitudeEmbedding" and _literal_normalize(call),
        "manual_norm_divide": False,
        "boundary_unresolved": unresolved or leaf == "SampledExpectation",
        "boundary_api": api if unresolved else "",
        **({"expval_observation": True} if leaf == "MeasureAll" else {}),
    }


def _literal_normalize(call: ast.Call) -> bool:
    value = next((keyword.value for keyword in call.keywords if keyword.arg == "normalize"), None)
    return _literal_bool(value, False)


def _has_norm_division(tree: ast.AST) -> bool:
    """Detect a denominator derived from a norm in the source unit."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        denominator = _dotted(node.right)
        if "norm" in denominator.lower():
            return True
        if any(isinstance(child, ast.Name) and "norm" in child.id.lower() for child in ast.walk(node.right)):
            return True
        if any(isinstance(child, ast.Call) and "norm" in _dotted(child.func).lower() for child in ast.walk(node.right)):
            return True
    return False


def _semantic_safety_facts(tree: ast.AST) -> dict[str, Any]:
    """Recover high-confidence source violations at quantum boundaries.

    These checks are deliberately based on framework contracts and dataflow
    shape, rather than repository names.  They cover failures that cannot be
    represented by an API name alone: an event vector used as a scalar kernel,
    computational samples sent to an arbitrary observable processor, a
    probability mass threshold that changes a nonzero distribution, and an
    observable indexed by the minimum wire of a non-adjacent product.
    """

    errors: list[str] = []
    error_sources: dict[str, set[str]] = {}

    def add_error(error: str, *source_paths: str) -> None:
        errors.append(error)
        error_sources.setdefault(error, set()).update(path for path in source_paths if path)
    calls = [
        _dotted(node.func).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    leaves = {name.rsplit(".", 1)[-1] for name in calls if name}

    has_probs = "probs" in leaves or "probabilities" in leaves
    has_kernel_matrix = "kernel_matrix" in leaves
    has_axis_mean = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"mean", "average"}
        and any(keyword.arg == "axis" and _literal_value(keyword.value) == 0 for keyword in node.keywords)
        for node in ast.walk(tree)
    )
    if has_probs and has_kernel_matrix and has_axis_mean:
        add_error(
            "ProbabilityEventAxisMismatch",
            *{
                _node_source_path(node)
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and _dotted(node.func).rsplit(".", 1)[-1].lower() in {"probs", "probabilities", "kernel_matrix", "mean", "average"}
            },
        )

    def is_abs_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and _dotted(node.func).rsplit(".", 1)[-1].lower() in {"abs", "absolute"}

    def is_sum_abs(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and _dotted(node.func).rsplit(".", 1)[-1].lower() in {"sum", "nansum"}
            and bool(node.args)
            and is_abs_call(node.args[0])
        )

    # A sum-of-magnitudes normalization has a concrete zero witness unless a
    # dominating guard is present.  The source contract is especially clear
    # when the helper is named as a normalizer or probability constructor.
    for region in ast.walk(tree):
        if not isinstance(region, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        region_name = region.name.lower()
        probability_context = any(
            token in region_name for token in ("norm", "prob", "categor", "weight", "cost", "loss")
        )
        for node in ast.walk(region):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div) or not is_sum_abs(node.right):
                continue
            guarded = any(
                isinstance(parent, ast.If)
                and any(
                    isinstance(test, ast.Compare)
                    and any(_literal_value(comparator) == 0 for comparator in test.comparators)
                    for test in (parent.test,)
                )
                for parent in ast.walk(region)
                if isinstance(parent, ast.If) and node in ast.walk(parent)
            )
            if probability_context and not guarded:
                add_error("ZeroSumNormalization", _node_source_path(region))
                break

    # A finite-shot implementation must either rotate the observable basis or
    # reject arbitrary observables.  Passing a cached computational sample
    # tensor directly to ExpectationMP is a source-level mismatch.
    source_paths = {
        _node_source_path(node)
        for node in ast.walk(tree)
    }
    for source_path in source_paths:
        unit = [node for node in ast.walk(tree) if _node_source_path(node) == source_path]
        unit_calls = {
            _dotted(node.func).lower()
            for node in unit
            if isinstance(node, ast.Call)
        }
        if (
            any(name.endswith("expectationmp") for name in unit_calls)
            and any(name.endswith("process_samples") for name in unit_calls)
            and any(isinstance(node, ast.Attribute) and node.attr == "_samples" for node in unit)
            and any(isinstance(node, ast.Attribute) and node.attr == "_active_shots" for node in unit)
            and not any(
                isinstance(node, ast.Call)
                and any(token in _dotted(node.func).lower() for token in ("rotate", "basis_rotation", "change_basis", "diagonalize"))
                for node in unit
            )
        ):
            add_error("FiniteShotObservableBasisMismatch", source_path)

    # ``random_index`` is a probability sampler.  Replacing every mass below
    # an absolute epsilon changes valid distributions when the joint event is
    # merely very unlikely, so this is an unconditional semantic violation.
    for region in ast.walk(tree):
        if not isinstance(region, (ast.FunctionDef, ast.AsyncFunctionDef)) or region.name != "random_index":
            continue
        epsilon_default = any(
            isinstance(default, ast.Constant)
            and isinstance(default.value, (int, float))
            and 0 < float(default.value) <= 1e-12
            for default in region.args.defaults
        )
        has_threshold = any(
            isinstance(node, ast.Call)
            and _dotted(node.func).rsplit(".", 1)[-1].lower() == "where"
            and any(isinstance(child, ast.Compare) for child in ast.walk(node))
            for node in ast.walk(region)
        )
        if epsilon_default and has_threshold:
            add_error("ProbabilityMassThreshold", _node_source_path(region))

    # Using the minimum wire to index a multi-wire observable collapses
    # non-adjacent products onto an adjacent interpretation.
    for source_path in source_paths:
        unit = [node for node in ast.walk(tree) if _node_source_path(node) == source_path]
        has_wire_min = any(
            isinstance(node, ast.Call)
            and _dotted(node.func).rsplit(".", 1)[-1] == "min"
            and any("wire" in _dotted(child).lower() for child in ast.walk(node))
            for node in unit
        )
        has_observable_mapping = any(
            isinstance(node, ast.Name)
            and any(token in node.id.lower() for token in ("pauli", "majorana", "observable"))
            for node in unit
        )
        if has_wire_min and has_observable_mapping:
            add_error("NonAdjacentWireMapping", source_path)

    errors = list(dict.fromkeys(errors))
    return {
        "semantic_errors": tuple(errors),
        "semantic_error": errors[0] if errors else "",
        "semantic_error_sources": {
            error: tuple(sorted(paths)) for error, paths in sorted(error_sources.items())
        },
    }


def _span_path_matches(actual: str, expected: str) -> bool:
    if not expected:
        return True
    actual_norm = actual.replace("\\", "/")
    expected_norm = expected.replace("\\", "/")
    return actual_norm == expected_norm or actual_norm.endswith("/" + expected_norm) or expected_norm.endswith("/" + actual_norm)


def _span_score(site: SourceUseSite, spans: Sequence[SourceSpan]) -> float:
    if not spans:
        return 0
    score = 0.0
    for span in spans:
        if not _span_path_matches(site.source_path, span.path):
            continue
        if span.start_line <= site.line <= span.end_line:
            score = max(score, 100 * span.weight)
        elif site.line <= span.end_line and span.start_line <= site.end_line:
            score = max(score, 50 * span.weight)
    return score


def _enclosing_evidence_spans(tree: ast.AST, spans: Sequence[SourceSpan]) -> tuple[SourceSpan, ...]:
    """Recover a declaration/return anchor within its smallest lexical function."""

    recovered = list(spans)
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for span in spans:
        enclosing = [
            node for node in functions
            if _span_path_matches(_node_source_path(node), span.path)
            and min([node.lineno] + [item.lineno for item in node.decorator_list]) <= span.start_line
            and span.end_line <= node.end_lineno
        ]
        if enclosing:
            node = min(enclosing, key=lambda item: item.end_lineno - item.lineno)
            recovered.append(SourceSpan(_node_source_path(node), node.lineno, node.end_lineno, span.weight * 0.1))
    return tuple(recovered)


def _fact_key(value: Any) -> Any:
    """Make structured source facts stable and sortable for candidate merging."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _fact_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_fact_key(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _candidate_semantic_signature(site: SourceUseSite) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Return the contract-relevant part of a candidate for safe tie merging."""

    return site.consumer_kind, tuple(sorted((key, _fact_key(value)) for key, value in site.facts.items()))


def _merge_normalization_facts(sites: Sequence[SourceUseSite]) -> dict[str, Any]:
    normalized = [site.facts for site in sites if site.facts.get("normalization_division")]
    unknown = any(site.facts.get("normalization_dependency_unknown") for site in sites)
    if not normalized:
        return {"normalization_dependency_unknown": True} if unknown else {}
    return {
        "normalization_dependency_unknown": unknown,
        "normalization_division": True,
        "manual_norm_divide": True,
        "domain_proven": all(facts.get("domain_proven", False) for facts in normalized),
        "zero_norm_witness": any(facts.get("zero_norm_witness", False) for facts in normalized),
        "nonzero_guard": all(facts.get("nonzero_guard", False) or facts.get("domain_proven", False) for facts in normalized),
    }


def _site_semantic_key(site: SourceUseSite) -> tuple[object, ...]:
    """Deduplicate wrapper spellings that describe one source boundary."""

    return (
        site.qualified_name,
        site.source_path,
        site.line,
        site.end_line,
        _candidate_semantic_signature(site),
    )


def _candidate_use_sites(tree: ast.AST, aliases: Mapping[str, str], frameworks: set[str]) -> tuple[SourceUseSite, ...]:
    """Recover direct observations and bounded wrapper/QNode boundaries."""

    regions = _function_nodes(tree)
    symbols = SourceSymbols(regions)
    api_contexts = {
        _node_source_path(region): _api_aliases(region)
        for name, region in regions
        if name.endswith("<module>")
    }
    module_bindings = {
        _node_source_path(region): _literal_bindings(region)
        for name, region in regions
        if name.endswith("<module>")
    }
    # Preserve the owning scope of every call and assignment.
    node_regions: dict[int, str] = {}
    for region_name, region in sorted(
        regions,
        key=lambda item: (
            getattr(item[1], "end_lineno", 10**9) - getattr(item[1], "lineno", 1),
            getattr(item[1], "lineno", 1),
        ),
    ):
        for child in _region_walk(region):
            node_regions.setdefault(id(child), region_name)

    def region_for_node(node: ast.AST) -> str:
        return node_regions.get(id(node), "<module>")

    def local_region_name(region_name: str) -> str:
        return region_name.rsplit("::", 1)[-1]

    def region_class_prefix(region_name: str) -> str:
        local = local_region_name(region_name)
        return local.rsplit(".", 1)[0] if "." in local else ""

    direct_by_function: dict[str, list[SourceUseSite]] = {}
    qualified_qnode_bindings: dict[tuple[str, str, str], set[str]] = {}
    qnode_assignments: list[tuple[ast.Call, set[str], str]] = []
    consumer_bindings: dict[tuple[str, str, str], tuple[str, str, Mapping[str, Any]]] = {}
    object_bindings: dict[tuple[str, str, str], str] = {}
    call_normalization_cache: dict[int, dict[str, Any]] = {}

    def binding_key(name: str, region: str, path: str) -> tuple[str, str, str]:
        scope = local_region_name(region)
        if name.startswith(("self.", "cls.")):
            scope = region_class_prefix(region)
        return path, scope, name

    def binding_lookup(bindings: Mapping, name: str, region: str, path: str) -> Any:
        key = binding_key(name, region, path)
        return bindings.get(key) or bindings.get((path, "<module>", name))

    # Resolve class fields such as self.qnode back to the function passed
    # into QNode, so a wrapper call can inherit the real consumer contract.
    for node in ast.walk(tree):
        value: ast.AST | None = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if not isinstance(value, ast.Call):
            continue
        assignment_region = region_for_node(node)
        assignment_source = _node_source_path(node)
        file_aliases, file_frameworks = api_contexts.get(assignment_source, (aliases, frameworks))
        constructor = _canonical_call(value, file_aliases)
        observed_constructor = _observation_kind(constructor, file_frameworks)
        if observed_constructor is not None and constructor.rsplit(".", 1)[-1] in {
            "MeasureAll", "PQC", "ControlledPQC", "Expectation", "SampledExpectation", "Sample",
            "EstimatorQNN", "SamplerQNN", "TwoLayerQNN", "ExpvalCost",
        }:
            api, consumer_kind = observed_constructor
            binding_facts = _api_facts(api, consumer_kind, value)
            for target in targets:
                target_name = _dotted(target)
                if target_name:
                    consumer_bindings[binding_key(target_name, assignment_region, assignment_source)] = (api, consumer_kind, binding_facts)
        constructor_leaf = constructor.rsplit(".", 1)[-1]
        object_type = None
        if constructor.startswith("qiskit.") and constructor_leaf in {
            "QuantumCircuit", "Statevector", "DensityMatrix", "Estimator", "EstimatorV2",
            "BackendEstimator", "BackendEstimatorV2", "Sampler", "SamplerV2", "BackendSampler",
            "BackendSamplerV2", "AerSimulator", "QuantumInstance", "FastEstimator", "FastReverseGradientEstimator",
        }:
            object_type = constructor
        elif constructor.startswith("qiskit.") and constructor_leaf in {"from_instruction", "from_label"}:
            object_type = constructor.rsplit(".", 1)[0]
        elif constructor.startswith("cirq.") and constructor_leaf in {"Simulator", "DensityMatrixSimulator"}:
            object_type = constructor
        elif constructor in {"strawberryfields.Engine", "tq.QuantumDevice", "pulser.QutipEmulator.from_sequence"}:
            object_type = constructor
        if object_type:
            for target in targets:
                target_name = _dotted(target)
                if target_name:
                    object_bindings[binding_key(target_name, assignment_region, assignment_source)] = object_type
        if constructor not in {"qml.QNode", "qml.qnode"}:
            continue
        func_expr = next((keyword.value for keyword in value.keywords if keyword.arg == "func"), None)
        if func_expr is None and value.args:
            func_expr = value.args[0]
        if not isinstance(func_expr, (ast.Name, ast.Attribute)):
            continue
        target_regions = symbols.targets(func_expr, assignment_region, assignment_source)
        qnode_assignments.append((value, target_regions, assignment_region))
        for target in targets:
            target_name = _dotted(target)
            if target_name:
                qualified_qnode_bindings[binding_key(target_name, assignment_region, assignment_source)] = target_regions

    for region_name, region in regions:
        sites: list[SourceUseSite] = []
        aliases, frameworks = api_contexts.get(_node_source_path(region), (aliases, frameworks))
        custom_return = _custom_return_site(region, region_name)
        if custom_return is not None:
            sites.append(custom_return)
        inherited = module_bindings.get(_node_source_path(region), {})
        region_semantic_facts = _normalization_facts(region, inherited)
        region_semantic_facts.update(_consumer_shape_facts(region, inherited))
        region_has_division = any(
            isinstance(candidate, ast.BinOp) and isinstance(candidate.op, ast.Div)
            for candidate in _region_walk(region)
        )
        for node in _region_walk(region):
            if isinstance(node, ast.Attribute) and node.attr == "data":
                owner = _dotted(node.value).lower()
                if any(token in owner for token in ("backend", "result", "array", "output", "payload")):
                    sites.append(
                        SourceUseSite(
                            qualified_name=region_name,
                            api="opaque:classical_array",
                            consumer_kind=ConsumerKind.CLASSICAL_ANALYSIS.value,
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            source_path=_node_source_path(node),
                            source_location=f"{_node_source_path(node)}:{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}",
                            confidence="dataflow",
                            facts={"classical_array_consumer": True, "opaque_classical_array": True, **region_semantic_facts},
                        )
                    )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and _amplitude_division(node):
                sites.append(
                    SourceUseSite(
                        qualified_name=region_name,
                        api="classical:amplitude_division",
                        consumer_kind=ConsumerKind.CLASSICAL_ANALYSIS.value,
                        line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        source_path=_node_source_path(node),
                        source_location=f"{_node_source_path(node)}:{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}",
                        confidence="direct",
                        facts={
                            "classical_array_consumer": True,
                            "amplitude_division": True,
                            "manual_norm_divide": False,
                            **region_semantic_facts,
                        },
                    )
                )
            if not isinstance(node, ast.Call):
                continue
            name = _canonical_call(node, aliases)
            raw_name = _dotted(node.func)
            if "." in raw_name:
                owner_name, method_name = raw_name.rsplit(".", 1)
                object_type = binding_lookup(object_bindings, owner_name, region_name, _node_source_path(node))
                if object_type is not None:
                    name = f"{object_type}.{method_name}"
            observed = _observation_kind(name, frameworks)
            function_name = raw_name
            owner = function_name.rsplit(".", 1)[0] if "." in function_name else ""
            custom_observed = False
            binding = binding_lookup(consumer_bindings, function_name, region_name, _node_source_path(node))
            if observed is None and binding is None and not symbols.targets(node.func, region_name, _node_source_path(node)):
                observed = _custom_observation_kind(name, region_name)
                custom_observed = observed is not None
            if observed is None and region_semantic_facts.get("controlled_operation"):
                method = raw_name.rsplit(".", 1)[-1].lower()
                if method in {"cp", "cu1", "crz", "cphase", "control"}:
                    sites.append(
                        SourceUseSite(
                            qualified_name=region_name,
                            api=f"controlled:{raw_name}",
                            consumer_kind=ConsumerKind.CONTROLLED_USE.value,
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            source_path=_node_source_path(node),
                            source_location=f"{_node_source_path(node)}:{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}",
                            confidence="direct",
                            facts={"controlled_operation": True, **region_semantic_facts},
                        )
                    )
                    continue
            if observed is None and name.rsplit(".", 1)[-1] not in {
                "float", "int", "len", "list", "tuple", "sum", "abs", "array", "asarray", "tensor", "norm"
            }:
                has_opaque_array_argument = any(
                    isinstance(child, ast.Attribute)
                    and child.attr == "data"
                    and any(token in _dotted(child.value).lower() for token in ("backend", "result", "array", "output", "payload"))
                    for argument in node.args
                    for child in ast.walk(argument)
                )
                if has_opaque_array_argument:
                    sites.append(
                        SourceUseSite(
                            qualified_name=region_name,
                            api=f"unknown_consumer:{raw_name}",
                            consumer_kind=ConsumerKind.CLASSICAL_ANALYSIS.value,
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            source_path=_node_source_path(node),
                            source_location=f"{_node_source_path(node)}:{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}",
                            confidence="dataflow",
                            facts={"contract_unknown": True, "opaque_classical_array": True, **region_semantic_facts},
                        )
                    )
                    continue
            if observed is None:
                binding = binding_lookup(consumer_bindings, function_name, region_name, _node_source_path(node))
                if binding is None:
                    continue
                api, consumer_kind, facts = binding
                sites.append(
                    SourceUseSite(
                        qualified_name=region_name,
                        api=f"bound:{api}",
                        consumer_kind=consumer_kind,
                        line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        source_path=_node_source_path(node),
                        source_location=f"{_node_source_path(node)}:{node.lineno}-{getattr(node, 'end_lineno', node.lineno)}",
                        confidence="wrapper",
                        facts=dict(facts),
                    )
                )
                continue
            api, consumer_kind = observed
            call_end = getattr(node, "end_lineno", node.lineno)
            facts = _api_facts(api, consumer_kind, node)
            facts["manual_norm_divide"] = consumer_kind == ConsumerKind.PREPARE_PURE.value and (
                region_semantic_facts.get("normalization_division", False) or region_has_division
            )
            if custom_observed:
                custom_facts = _custom_observation_facts(name, region_name)
                facts.update(custom_facts)
                facts["boundary_unresolved"] = not bool(custom_facts)
            facts.update(region_semantic_facts)
            if consumer_kind == ConsumerKind.PREPARE_PURE.value or api in {"qiskit.Statevector", "qiskit.DensityMatrix"}:
                value_facts = _normalization_facts(region, inherited, node)
                facts.update(value_facts)
                facts["manual_norm_divide"] = value_facts["normalization_division"]
            sites.append(
                SourceUseSite(
                    qualified_name=region_name,
                    api=api,
                    consumer_kind=consumer_kind,
                    line=node.lineno,
                    end_line=call_end,
                    source_path=_node_source_path(node),
                    source_location=f"{_node_source_path(node)}:{node.lineno}-{call_end}",
                    facts=facts,
                )
            )
        terminal_consumers = [
            site
            for site in sites
            if site.facts.get("probs_return")
            or site.facts.get("expval_observation")
            or site.facts.get("sample_observation")
        ]
        if terminal_consumers:
            sites = [
                site
                for site in sites
                if not (
                    site.facts.get("state_return")
                    and site.api.endswith("Statevector.from_instruction")
                )
            ]
        # A QNode decorator is often the only line retained by a source
        # evidence record.  Expose the decorated function as a boundary at
        # that line while preserving the original call-level candidate.
        decorator_lines = [
            getattr(decorator, "lineno", getattr(region, "lineno", 1))
            for decorator in getattr(region, "decorator_list", ())
        ]
        has_composite_inputs = (
            any(site.facts.get("state_prep") for site in sites)
            and any(
                site.facts.get("state_return")
                or site.facts.get("expval_observation")
                or site.facts.get("sample_observation")
                or site.facts.get("probs_return")
                for site in sites
            )
        )
        if decorator_lines and sites and not has_composite_inputs:
            boundary_line = min(decorator_lines)
            boundary_end = getattr(region, "end_lineno", boundary_line)
            sites.extend(
                SourceUseSite(
                    qualified_name=region_name,
                    api=f"decorator:{site.api}",
                    consumer_kind=site.consumer_kind,
                    line=boundary_line,
                    end_line=boundary_end,
                    source_path=site.source_path,
                    source_location=f"{site.source_path}:{boundary_line}-{boundary_end}",
                    confidence="dataflow",
                    facts=dict(site.facts),
                )
                for site in tuple(sites)
            )
        preparation_sites = [site for site in sites if site.facts.get("state_prep")]
        consumer_sites = [
            site
            for site in sites
            if site.facts.get("state_return")
            or site.facts.get("expval_observation")
            or site.facts.get("sample_observation")
            or site.facts.get("probs_return")
        ]
        consumer_kinds = {
            "state" if site.facts.get("state_return")
            else "expval" if site.facts.get("expval_observation")
            else "sample" if site.facts.get("sample_observation")
            else "probs"
            for site in consumer_sites
        }
        if preparation_sites and len(consumer_kinds) == 1:
            consumer = consumer_sites[0]
            combined_facts: dict[str, Any] = {}
            for site in preparation_sites + consumer_sites:
                combined_facts.update(site.facts)
            combined_facts.update(_merge_normalization_facts(preparation_sites))
            combined_facts["state_prep"] = True
            combined_facts["manual_norm_divide"] = any(site.facts.get("manual_norm_divide", False) for site in preparation_sites + consumer_sites)
            kind = next(iter(consumer_kinds))
            combined_facts["state_return"] = kind == "state"
            combined_facts["expval_observation"] = kind == "expval"
            combined_facts["sample_observation"] = kind == "sample"
            combined_facts["probs_return"] = kind == "probs"
            sites.append(
                SourceUseSite(
                    qualified_name=region_name,
                    api=f"composite:stateprep+{kind}",
                    consumer_kind=consumer.consumer_kind,
                    line=min(decorator_lines or [getattr(region, "lineno", 1)]),
                    end_line=getattr(region, "end_lineno", getattr(region, "lineno", 1)),
                    source_path=_node_source_path(region),
                    source_location=f"{_node_source_path(region)}:{getattr(region, 'lineno', 1)}-{getattr(region, 'end_lineno', getattr(region, 'lineno', 1))}",
                    confidence="dataflow",
                    facts=combined_facts,
                )
            )
        direct_by_function[region_name] = sites

    # Only returned nested callables form factory boundaries. Merely defining
    # an unused inner QNode does not make the enclosing function a consumer.
    factory_targets: dict[str, set[str]] = {}
    for region_name, region in regions:
        for node in _region_walk(region):
            if isinstance(node, ast.Return) and isinstance(node.value, (ast.Name, ast.Attribute)):
                targets = symbols.targets(node.value, region_name, _node_source_path(node))
                nested = {name for name in targets if name.startswith(region_name + ".")}
                if nested:
                    factory_targets.setdefault(region_name, set()).update(nested)

    # Propagate a direct quantum result through local wrappers and QNode calls.
    # A small fixed point handles helper -> wrapper -> QNode chains while
    # keeping the interprocedural proof boundary finite and inspectable.
    propagated: list[SourceUseSite] = [site for sites in direct_by_function.values() for site in sites]
    propagated_keys = {_site_semantic_key(site) for site in propagated}

    def unique_sites(items: Sequence[SourceUseSite]) -> list[SourceUseSite]:
        result: list[SourceUseSite] = []
        seen: set[tuple[object, ...]] = set()
        for item in items:
            key = _site_semantic_key(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    for _ in range(6):
        additions: list[SourceUseSite] = []
        for region_name, region in regions:
            for target_name in sorted(factory_targets.get(region_name, ())):
                for site in tuple(direct_by_function.get(target_name, ())):
                    lifted = SourceUseSite(
                        qualified_name=region_name, api=f"factory:{site.api}",
                        consumer_kind=site.consumer_kind,
                        line=getattr(region, "lineno", site.line),
                        end_line=getattr(region, "end_lineno", site.end_line),
                        source_path=_node_source_path(region), source_location=site.source_location,
                        confidence="wrapper", facts=dict(site.facts),
                    )
                    key = _site_semantic_key(lifted)
                    if key not in propagated_keys:
                        additions.append(lifted)
                        propagated_keys.add(key)
            for node in _region_walk(region):
                if not isinstance(node, ast.Call):
                    continue
                called = _dotted(node.func)
                current_source = _node_source_path(node)
                bound_regions = binding_lookup(qualified_qnode_bindings, called, region_name, current_source)
                target_regions = bound_regions or symbols.targets(node.func, region_name, current_source)
                targets = unique_sites([
                    site
                    for target_region in sorted(target_regions)
                    for site in direct_by_function.get(target_region, ())
                ])
                for target in targets:
                    propagated_facts = dict(target.facts)
                    if target.facts.get("state_prep"):
                        if id(node) not in call_normalization_cache:
                            call_normalization_cache[id(node)] = _normalization_facts(region, module_bindings.get(current_source, {}), node, all_arguments=True)
                        local_facts = call_normalization_cache[id(node)]
                        if local_facts["normalization_division"]:
                            propagated_facts.update(local_facts)
                            propagated_facts["manual_norm_divide"] = True
                        if local_facts["normalization_dependency_unknown"]:
                            propagated_facts["normalization_dependency_unknown"] = True
                    candidate = SourceUseSite(
                        qualified_name=region_name,
                        api=f"wrapper:{target.api}",
                        consumer_kind=target.consumer_kind,
                        line=getattr(node, "lineno", getattr(region, "lineno", 1)),
                        end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                        source_path=_node_source_path(node),
                        source_location=f"{_node_source_path(node)}:{getattr(node, 'lineno', getattr(region, 'lineno', 1))}-{getattr(node, 'end_lineno', getattr(node, 'lineno', 1))}",
                        confidence="wrapper",
                        facts=propagated_facts,
                    )
                    key = _site_semantic_key(candidate)
                    if key not in propagated_keys:
                        additions.append(candidate)
                        propagated_keys.add(key)
        if not additions:
            break
        propagated.extend(additions)
        for site in additions:
            direct_by_function.setdefault(site.qualified_name, []).append(site)

    # Combine propagated preparation and consumer facts at the wrapper/QNode
    # boundary. This recovers a QNode body whose encoding and measurement are
    # implemented in separate helper methods.
    propagated_by_function: dict[str, list[SourceUseSite]] = {}
    for site in propagated:
        propagated_by_function.setdefault(site.qualified_name, []).append(site)
    for region_name, region in regions:
        sites = propagated_by_function.get(region_name, [])
        preparation_sites = [site for site in sites if site.facts.get("state_prep")]
        consumer_sites = [
            site
            for site in sites
            if site.facts.get("state_return")
            or site.facts.get("expval_observation")
            or site.facts.get("sample_observation")
            or site.facts.get("probs_return")
        ]
        consumer_kinds = {
            "state" if site.facts.get("state_return")
            else "expval" if site.facts.get("expval_observation")
            else "sample" if site.facts.get("sample_observation")
            else "probs"
            for site in consumer_sites
        }
        if not preparation_sites or len(consumer_kinds) != 1:
            continue
        kind = next(iter(consumer_kinds))
        combined_facts: dict[str, Any] = {}
        for site in preparation_sites + consumer_sites:
            combined_facts.update(site.facts)
        combined_facts.update(_merge_normalization_facts(preparation_sites))
        combined_facts["state_prep"] = True
        combined_facts["state_return"] = kind == "state"
        combined_facts["expval_observation"] = kind == "expval"
        combined_facts["sample_observation"] = kind == "sample"
        combined_facts["probs_return"] = kind == "probs"
        combined_facts["manual_norm_divide"] = any(site.facts.get("manual_norm_divide", False) for site in preparation_sites + consumer_sites)
        combined = SourceUseSite(
            qualified_name=region_name,
            api=f"composite:stateprep+{kind}",
            consumer_kind=consumer_sites[0].consumer_kind,
            line=min(
                [
                    getattr(decorator, "lineno", getattr(region, "lineno", 1))
                    for decorator in getattr(region, "decorator_list", ())
                ]
                or [getattr(region, "lineno", 1)]
            ),
            end_line=getattr(region, "end_lineno", getattr(region, "lineno", 1)),
            source_path=_node_source_path(region),
            source_location=f"{_node_source_path(region)}:{getattr(region, 'lineno', 1)}-{getattr(region, 'end_lineno', getattr(region, 'lineno', 1))}",
            confidence="dataflow",
            facts=combined_facts,
        )
        key = _site_semantic_key(combined)
        if key not in propagated_keys:
            propagated.append(combined)
            propagated_keys.add(key)

    # Treat a QNode constructor as a boundary use-site when its body is a
    # helper graph. The assignment line is often the only RQ0 evidence anchor
    # for frameworks that build QNodes dynamically.
    for qnode_call, target_regions, assignment_region in qnode_assignments:
        targets = [
            site
            for target_region in sorted(target_regions)
            for site in direct_by_function.get(target_region, ())
            if site.api.startswith("composite:")
        ]
        for target in targets:
            propagated.append(
                SourceUseSite(
                    qualified_name=target.qualified_name,
                    api=f"qnode:{target.api}",
                    consumer_kind=target.consumer_kind,
                    line=qnode_call.lineno,
                    end_line=getattr(qnode_call, "end_lineno", qnode_call.lineno),
                    source_path=_node_source_path(qnode_call),
                    source_location=f"{_node_source_path(qnode_call)}:{qnode_call.lineno}-{getattr(qnode_call, 'end_lineno', qnode_call.lineno)}",
                    confidence="dataflow",
                    facts=dict(target.facts),
                )
            )

    # A normalization in the selected wrapper plus a state-preparation result
    # is the common QNode pattern behind the qdm-style C1 case.
    for site in tuple(propagated):
        region = next((node for name, node in regions if name == site.qualified_name), None)
        if region is None:
            continue
        has_division = bool(site.facts.get("normalization_division"))
        has_state_prep = site.facts.get("state_prep", False)
        has_state_return = site.facts.get("state_return", False)
        if has_division and has_state_prep and has_state_return:
            facts = dict(site.facts)
            facts["manual_norm_divide"] = True
            propagated.append(
                SourceUseSite(
                    qualified_name=site.qualified_name,
                    api=site.api,
                    consumer_kind=site.consumer_kind,
                    line=site.line,
                    end_line=site.end_line,
                    source_path=site.source_path,
                    source_location=site.source_location,
                    confidence="dataflow",
                    facts=facts,
                )
            )
    return tuple(propagated)


def _relevant_nodes(tree: ast.AST, entry_points: Tuple[str, ...] | None) -> list[ast.AST]:
    if not entry_points:
        return [tree]
    selected: list[ast.AST] = []
    wanted = set(entry_points)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            selected.append(node)
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                qualified = f"{node.name}.{child.name}" if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else ""
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (qualified in wanted or child.name in wanted):
                    selected.append(child)
    return selected or [tree]


def _api_aliases(tree: ast.AST) -> tuple[dict[str, str], set[str]]:
    """Canonicalize explicitly imported SDK symbols within one source module."""

    aliases: dict[str, str] = {}
    frameworks: set[str] = set()
    roots = {
        "pennylane": "qml", "qiskit": "qiskit", "cirq": "cirq",
        "tensorflow_quantum": "tfq", "torchquantum": "tq",
        "strawberryfields": "strawberryfields", "qutip": "qutip",
        "pulser": "pulser", "pulser_simulation": "pulser",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".", 1)[0]
                canonical = "qiskit" if root.startswith("qiskit_") else roots.get(root)
                if canonical:
                    frameworks.add("qiskit" if root.startswith("qiskit_") else root)
                    suffix = item.name.partition(".")[2] if item.asname else ""
                    aliases[item.asname or root] = ".".join(p for p in (canonical, suffix) if p)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            canonical = "qiskit" if root.startswith("qiskit_") else roots.get(root)
            if canonical:
                frameworks.add("qiskit" if root.startswith("qiskit_") else root)
                for item in node.names:
                    if item.name != "*":
                        aliases[item.asname or item.name] = f"{canonical}.{item.name}"
    return aliases, frameworks


def _canonical_call(node: ast.Call, aliases: Mapping[str, str]) -> str:
    name = _dotted(node.func)
    if name in aliases:
        return aliases[name]
    if name:
        root, _, suffix = name.partition(".")
        if root in aliases:
            return f"{aliases[root]}.{suffix}" if suffix else aliases[root]
    return name


def _custom_return_site(region: ast.AST, region_name: str) -> SourceUseSite | None:
    """Recover a small set of transparent custom return representations."""

    local = region_name.rsplit("::", 1)[-1]
    if not _quantum_name_context(local):
        return None
    method_name = local.rsplit(".", 1)[-1].lower()
    for node in _region_walk(region):
        if not isinstance(node, ast.Return) or isinstance(node.value, ast.Call):
            continue
        value_name = _dotted(node.value).lower() if node.value is not None else ""
        probability_method = method_name in {
            "born_rule", "probability", "probabilities", "statevector_to_probability",
            "statevector_to_probabilities", "amplitudetoprobability",
        }
        squared_absolute = any(
            isinstance(candidate, ast.BinOp)
            and isinstance(candidate.op, ast.Pow)
            and isinstance(candidate.right, ast.Constant)
            and candidate.right.value == 2
            and isinstance(candidate.left, ast.Call)
            and _dotted(candidate.left.func).rsplit(".", 1)[-1].lower() in {"abs", "absolute"}
            for candidate in ast.walk(node.value)
        ) if node.value is not None else False
        if probability_method or squared_absolute:
            return SourceUseSite(
                qualified_name=region_name,
                api=f"custom:{method_name}",
                consumer_kind=ConsumerKind.MEASURE.value,
                line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                source_path=_node_source_path(node),
                source_location=f"{_node_source_path(node)}:{node.lineno}",
                confidence="custom_body",
                facts={"custom_boundary": True, "probs_return": True, "born_rule": True},
            )
        if method_name not in {"expval", "expectation", "expectation_value", "expval_z"} and value_name.rsplit(".", 1)[-1] not in {
            "state", "statevec", "state_vector", "statevector", "density_matrix",
            "probabilities", "probs", "counts",
        }:
            continue
        return SourceUseSite(
            qualified_name=region_name,
            api=f"custom:{method_name}",
            consumer_kind=ConsumerKind.CLASSICAL_ANALYSIS.value,
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            source_path=_node_source_path(node),
            source_location=f"{_node_source_path(node)}:{node.lineno}",
            confidence="custom_boundary",
            facts={"custom_boundary": True, "boundary_unresolved": True},
        )
    return None


def _custom_observation_facts(name: str, region_name: str) -> dict[str, Any]:
    """Recognize transparent custom observation adapters by their call role."""

    lowered = name.lower()
    local = region_name.rsplit("::", 1)[-1].lower()
    leaf = lowered.rsplit(".", 1)[-1]
    if leaf == "run" and "estimator" in lowered:
        return {"expval_observation": True}
    if leaf in {"measure", "sample", "samples", "get_counts", "sample_counts", "sample_memory"} and (
        "sample" in local or "random" in local or "counts" in lowered
    ):
        return {"sample_observation": True}
    if leaf in {"prob", "probs", "probability", "probabilities"}:
        return {"probs_return": True}
    if leaf in {"expval", "expect", "expectation", "expectation_value"}:
        return {"expval_observation": True}
    return {}


def _amplitude_division(node: ast.BinOp) -> bool:
    """Recognize a classical division by an amplitude-like tensor."""

    denominator = node.right
    names = {
        _dotted(child)
        for child in ast.walk(denominator)
        if isinstance(child, (ast.Name, ast.Attribute))
    }
    lowered = " ".join(name.lower() for name in names)
    has_amplitude_name = any(token in lowered for token in ("amp", "amplitude", "psi", "wavefunction"))
    has_abs = any(
        isinstance(child, ast.Call)
        and _dotted(child.func).split(".")[-1] in {"abs", "absolute"}
        for child in ast.walk(denominator)
    )
    return has_amplitude_name and has_abs


def _facts(tree: ast.AST, entry_points: Tuple[str, ...] | None = None) -> dict[str, Any]:
    facts = {
        "state_return": False,
        "probs_return": False,
        "expval_observation": False,
        "sample_observation": False,
        "state_prep": False,
        "amplitude_embedding": False,
        "explicit_normalize": False,
        "manual_norm_divide": False,
        "torch_consumer": False,
        "jax_consumer": False,
        "qiskit_observation": False,
        "cirq_observation": False,
        "state_observation": False,
    }
    aliases, frameworks = _api_aliases(tree)
    relevant = _relevant_nodes(tree, entry_points)
    for root in relevant:
        for node in ast.walk(root):
            if isinstance(node, ast.ImportFrom) and node.module == "torch":
                facts["torch_consumer"] = True
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("jax"):
                facts["jax_consumer"] = True
            if isinstance(node, ast.Call):
                name = _canonical_call(node, aliases)
                if name == "qml.state":
                    facts["state_return"] = True
                elif name == "qml.probs":
                    facts["probs_return"] = True
                elif name == "qml.expval":
                    facts["expval_observation"] = True
                elif name in {"qml.sample", "qml.counts", "qml.measure"}:
                    facts["sample_observation"] = True
                elif name in {"qml.QubitStateVector", "qml.StatePrep", "qml.AmplitudeEmbedding"}:
                    facts["state_prep"] = True
                    if name.endswith("AmplitudeEmbedding"):
                        facts["amplitude_embedding"] = True
                        normalize = next((keyword.value for keyword in node.keywords if keyword.arg == "normalize"), None)
                        facts["explicit_normalize"] = _literal_bool(normalize, False)
                elif name.endswith(".initialize"):
                    facts["qiskit_observation"] = True
                    facts["state_prep"] = True
                elif name.endswith(".sample_memory") or name.endswith(".sample_counts"):
                    facts["qiskit_observation"] = True
                    facts["sample_observation"] = True
                elif name.endswith(".probabilities"):
                    facts["qiskit_observation"] = True
                    facts["probs_return"] = True
                elif name in {"torch.norm", "torch.linalg.norm", "jnp.linalg.norm", "jax.numpy.linalg.norm"}:
                    facts["manual_norm_divide"] = True
                elif name in {"qiskit.Estimator", "qiskit.BackendEstimator", "qiskit.estimate"} or ("qiskit" in frameworks and name.endswith(".Estimator")):
                    facts["qiskit_observation"] = True
                    facts["expval_observation"] = True
                elif name in {"qiskit.Sampler", "qiskit.SamplerV2", "qiskit.BackendSampler", "qiskit.sample"} or ("qiskit" in frameworks and (name.endswith(".Sampler") or name.endswith(".SamplerV2"))):
                    facts["qiskit_observation"] = True
                    facts["sample_observation"] = True
                elif name in {"qiskit.Statevector", "qiskit.Statevector.from_instruction", "qiskit.Statevector.from_label"}:
                    facts["qiskit_observation"] = True
                    facts["state_return"] = True
                elif "qiskit" in frameworks and (name.endswith(".probabilities") or name.endswith(".get_counts")):
                    facts["qiskit_observation"] = True
                    facts["probs_return"] = True
                elif name in {"cirq.measure", "cirq.sample"}:
                    facts["cirq_observation"] = True
                    facts["sample_observation"] = True
                elif name in {"tfq.layers.Expectation", "tensorflow_quantum.Expectation"} or (name.endswith(".Expectation") and "tensorflow_quantum" in frameworks):
                    facts["cirq_observation"] = True
                    facts["expval_observation"] = True
                elif name.endswith(".Sample") and "tensorflow_quantum" in frameworks:
                    facts["cirq_observation"] = True
                    facts["sample_observation"] = True
                elif name.endswith(".measure") and ("qiskit" in frameworks or "cirq" in frameworks):
                    facts["qiskit_observation" if "qiskit" in frameworks else "cirq_observation"] = True
                    facts["sample_observation"] = True
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
                    facts["manual_norm_divide"] = True
    facts.update(_normalization_facts(tree))
    facts.update(_consumer_shape_facts(tree))
    facts.update(_semantic_safety_facts(tree))
    facts["state_observation"] = facts["state_return"] and not facts["state_prep"]
    return facts


def _evidence(
    source_id: str,
    source_digest: str,
    registry: EvidenceRegistry,
    facts: Mapping[str, Any],
    entry_points: Tuple[str, ...] | None,
    qualified_entry: str = "ast_module",
    derivation_rule: str = "explicit_api_ast_patterns",
) -> EvidenceRef:
    return registry.add(
        SourceEvidence(
            source_id=source_id,
            source_code_digest=source_digest,
            qualified_entry=qualified_entry,
            adapter_id="QTC.RestrictedASTSourceImporter",
        adapter_version="0.4",
        adapter_digest="source-to-core-ast-v4-value-bound-normalization",
            return_spec={"facts": dict(facts), "entry_points": list(entry_points or ())},
            derivation_rule=derivation_rule,
        ),
        "SourceEvidence",
    )


def _qdm_graph(facts: Mapping[str, Any] | None = None) -> tuple[CoreGraph, Mapping[str, ConsumerContract]]:
    facts = facts or {}
    nodes = (
        Node("x", "input", attributes={"shape": (None,)}),
        Node("norm_sq_x", "norm_sq", ("x",)),
        Node("norm_x", "sqrt", ("norm_sq_x",)),
        Node(
            "normalized",
            "divide",
            ("x", "norm_x"),
            attributes={
                "domain_proven": bool(facts.get("domain_proven")),
                "zero_witness": bool(facts.get("zero_norm_witness")),
                "source_location": "AST:normalization-divide",
            },
        ),
        Node("state", "ket", ("normalized",), attributes={"source_id": "qdm_state", "wire_order": (), "shape": (None,), "preparation_recipe": True}),
    )
    contracts = {
        "c_prepare": ConsumerContract(
            "c_prepare",
            ConsumerKind.PREPARE_PURE,
            "stateprep",
            allow_guard=bool(facts.get("nonzero_guard")),
        )
    }
    graph = CoreGraph(nodes, uses=(Use("u_prepare", "normalized", "prepare", "c_prepare", "prepared_state"),), inputs=("x",), outputs=("state",))
    return graph, contracts


def _expval_graph(normalized: bool) -> tuple[CoreGraph, Mapping[str, ConsumerContract]]:
    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node("embedded", "ket", ("features",), attributes={"source_id": "embedded", "wire_order": (), "shape": (None,), "preparation_recipe": True, "normalized": normalized}),
        Node("expval", "expval", ("embedded",), attributes={"source_id": "expval", "wire_order": (), "basis": "pauli_z"}),
    )
    contracts = {"c_expval": ConsumerContract("c_expval", ConsumerKind.MEASURE, "pauli_product", wire_order=())}
    graph = CoreGraph(nodes, uses=(Use("u_expval", "expval", "measure", "c_expval", "expval"),), inputs=("features",), outputs=("expval",))
    return graph, contracts


def _sample_graph(site: SourceUseSite | None = None) -> tuple[CoreGraph, Mapping[str, ConsumerContract | ContractResolution]]:
    facts = site.facts if site is not None else {}
    wire_order = tuple(facts.get("wire_order", ()))
    shots = next((int(value) for value in (facts.get("shots"),) if isinstance(value, int) and value > 0), 1)
    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node("samples", "samples", ("features",), attributes={"source_id": "samples", "wire_order": wire_order, "shot_model": "samples", "shots": shots}),
    )
    contract: ConsumerContract | ContractResolution
    if facts.get("missing_execution_identity"):
        contract = ContractResolution.underspecified(
            "joint sample observation has no execution identity", "execution_identity"
        )
    else:
        contract = ConsumerContract("c_sample", ConsumerKind.MEASURE, "computational_z", wire_order=wire_order, shots=ShotSpec("samples", shots=shots))
    contracts: Mapping[str, ConsumerContract | ContractResolution] = {"c_sample": contract}
    graph = CoreGraph(nodes, uses=(Use("u_sample", "samples", "measure", "c_sample", "sample"),), inputs=("features",), outputs=("samples",))
    return graph, contracts


def _probs_graph(site: SourceUseSite | None = None) -> tuple[CoreGraph, Mapping[str, ConsumerContract | ContractResolution]]:
    facts = site.facts if site is not None else {}
    wire_order = tuple(facts.get("wire_order", ()))
    index_map = tuple(facts.get("index_map", ()))
    output = "probabilities"
    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node("probabilities", "probs", ("features",), attributes={"source_id": "probabilities", "wire_order": wire_order, "shape": (None,), "shot_model": "analytic"}),
    )
    node_list = list(nodes)
    if index_map:
        output = "selected_probabilities"
        node_list.append(Node(output, "gather", ("probabilities",), attributes={"index_map": index_map, "mass_kind": "SELECTED_SUBMASS", "wire_order": wire_order}))
    if facts.get("dynamic_index") and not index_map:
        contract: ConsumerContract | ContractResolution = ContractResolution.underspecified(
            "the selected event depends on a runtime index", "index_map"
        )
    elif facts.get("wire_order_conflict"):
        contract = ConsumerContract(
            "c_probs", ConsumerKind.MEASURE, "computational_z", wire_order=wire_order,
            metadata={"semantic_error": "EventIdentityMismatch"},
        )
    else:
        contract = ConsumerContract("c_probs", ConsumerKind.MEASURE, "computational_z", wire_order=wire_order, selected_events=tuple({"event": int(value)} for _, value in index_map))
    contracts = {"c_probs": contract}
    graph = CoreGraph(tuple(node_list), uses=(Use("u_probs", output, "measure", "c_probs", "probs"),), inputs=("features",), outputs=(output,))
    return graph, contracts


def _state_graph() -> tuple[CoreGraph, Mapping[str, ConsumerContract]]:
    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node("state", "ket", ("features",), attributes={"source_id": "state", "wire_order": (), "shape": (None,), "classical_array": True}),
        Node("state_observation", "observer", ("state",), attributes={"observation": "state_observation"}),
    )
    contracts = {"c_state": ConsumerContract("c_state", ConsumerKind.CLASSICAL_ANALYSIS, "state_observation", requires_classical_array=True)}
    graph = CoreGraph(nodes, uses=(Use("u_state", "state", "observe", "c_state", "state"),), inputs=("features",), outputs=("state",))
    return graph, contracts


def _semantic_error_graph(site: SourceUseSite) -> tuple[CoreGraph, Mapping[str, ConsumerContract]]:
    """Create a resolved graph whose contract is blocked by a source proof."""

    facts = site.facts
    if facts.get("expval_observation"):
        opcode, observation = "expval", "pauli_product"
        kind = ConsumerKind.MEASURE
    elif facts.get("sample_observation"):
        opcode, observation = "samples", "computational_z"
        kind = ConsumerKind.MEASURE
    elif facts.get("probs_return"):
        opcode, observation = "probs", "computational_z"
        kind = ConsumerKind.MEASURE
    else:
        opcode, observation = "observer", "classical_array"
        kind = ConsumerKind.CLASSICAL_ANALYSIS
    error = str(facts.get("semantic_error") or "SourceSemanticViolation")
    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node("unsafe_result", opcode, ("features",), attributes={"source_location": site.source_location}),
    )
    contract = ConsumerContract(
        "c_source_safety",
        kind,
        observation,
        requires_classical_array=kind is ConsumerKind.CLASSICAL_ANALYSIS,
        metadata={"semantic_error": error},
    )
    graph = CoreGraph(
        nodes,
        uses=(Use("u_source_safety", "unsafe_result", "source_safety", "c_source_safety", "output"),),
        inputs=("features",),
        outputs=("unsafe_result",),
    )
    return graph, {"c_source_safety": contract}


def _classical_array_graph() -> tuple[CoreGraph, Mapping[str, ConsumerContract]]:
    """Represent a quantum-derived array consumed by classical tensor code."""

    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node("amplitudes", "ket", ("features",), attributes={"source_id": "amplitudes", "wire_order": (), "shape": (None,), "classical_array": True}),
        Node("classical_observation", "observer", ("amplitudes",), attributes={"observation": "classical_array_consumer"}),
    )
    contracts = {"c_classical": ConsumerContract("c_classical", ConsumerKind.CLASSICAL_ANALYSIS, "classical_array", requires_classical_array=True)}
    graph = CoreGraph(
        nodes,
        uses=(Use("u_classical", "classical_observation", "observe", "c_classical", "output"),),
        inputs=("features",),
        outputs=("classical_observation",),
    )
    return graph, contracts


def _opaque_classical_array_graph() -> tuple[CoreGraph, Mapping[str, ContractResolution]]:
    """Keep an unresolved backend data port explicit at the C1 boundary."""

    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node("backend_state", "ket", ("features",), attributes={"source_id": "opaque_backend", "shape": (None,), "classical_array": True}),
        Node("opaque_observation", "observer", ("backend_state",), attributes={"observation": "classical_array_consumer"}),
    )
    contracts = {
        "c_opaque": ContractResolution.underspecified(
            "the backend result exposes a data field but its quantum data port is not resolved",
            "backend_data_port",
        )
    }
    graph = CoreGraph(
        nodes,
        uses=(Use("u_opaque", "opaque_observation", "observe", "c_opaque", "output"),),
        inputs=("features",),
        outputs=("opaque_observation",),
    )
    return graph, contracts


def _ambiguous_consumer_graph() -> tuple[CoreGraph, Mapping[str, ContractResolution]]:
    """Keep a source ambiguity executable as an explicit RequestInfo result."""

    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node(
            "ambiguous_value",
            "placeholder",
            ("features",),
            attributes={"observation": "ambiguous_source_consumer"},
        ),
    )
    contracts = {
        "c_ambiguous": ContractResolution.underspecified(
            "the evidence selects multiple incompatible quantum consumers",
            "consumer_kind",
            "evidence_range",
        )
    }
    graph = CoreGraph(
        nodes,
        uses=(Use("u_ambiguous", "ambiguous_value", "observe", "c_ambiguous", "output"),),
        inputs=("features",),
        outputs=("ambiguous_value",),
    )
    return graph, contracts


def _prepare_graph(normalized: bool, source_location: str = "", *, allow_guard: bool = False, zero_witness: bool = False) -> tuple[CoreGraph, Mapping[str, ConsumerContract]]:
    nodes = (
        Node("features", "input", attributes={"shape": (None,)}),
        Node(
            "prepared",
            "ket",
            ("features",),
            attributes={"source_id": "prepared", "wire_order": (), "shape": (None,), "preparation_recipe": True, "normalized": normalized, "zero_witness": zero_witness},
            source_location=source_location,
        ),
    )
    contracts = {"c_prepare": ConsumerContract("c_prepare", ConsumerKind.PREPARE_PURE, "stateprep", allow_guard=allow_guard)}
    graph = CoreGraph(nodes, uses=(Use("u_prepare", "prepared", "prepare", "c_prepare", "prepared_state"),), inputs=("features",), outputs=("prepared",))
    return graph, contracts


def _normalized_measurement_graph(site: SourceUseSite) -> tuple[CoreGraph, Mapping[str, ConsumerContract]]:
    facts = site.facts
    wire_order = tuple(facts.get("wire_order", ()))
    domain_proven = bool(facts.get("domain_proven"))
    nodes: list[Node] = [
        Node("features", "input", attributes={"shape": (None,)}),
        Node("norm_sq", "norm_sq", ("features",)),
        Node("norm", "sqrt", ("norm_sq",)),
        Node("normalized", "divide", ("features", "norm"), attributes={"domain_proven": domain_proven, "zero_witness": bool(facts.get("zero_norm_witness"))}),
        Node("prepared", "ket", ("normalized",), attributes={"source_id": "normalized", "wire_order": wire_order, "shape": (None,), "preparation_recipe": True}),
    ]
    if facts.get("expval_observation"):
        output = "expval"
        nodes.append(Node(output, "expval", ("prepared",), attributes={"source_id": output, "wire_order": wire_order, "basis": "pauli_z"}))
        contract = ConsumerContract("c_measure", ConsumerKind.MEASURE, "pauli_product", requires_normalized=True, allow_guard=bool(facts.get("nonzero_guard")), wire_order=wire_order)
    elif facts.get("sample_observation"):
        output = "samples"
        nodes.append(Node(output, "samples", ("prepared",), attributes={"source_id": output, "wire_order": wire_order, "shot_model": "samples", "shots": 1}))
        contract = ConsumerContract("c_measure", ConsumerKind.MEASURE, "computational_z", requires_normalized=True, allow_guard=bool(facts.get("nonzero_guard")), wire_order=wire_order, shots=ShotSpec("samples", shots=1))
    else:
        output = "probabilities"
        nodes.append(Node(output, "probs", ("prepared",), attributes={"source_id": output, "wire_order": wire_order, "shape": (None,), "shot_model": "analytic"}))
        contract = ConsumerContract("c_measure", ConsumerKind.MEASURE, "computational_z", requires_normalized=True, allow_guard=bool(facts.get("nonzero_guard")), wire_order=wire_order)
    graph = CoreGraph(
        tuple(nodes),
        uses=(Use("u_measure", output, "measure", "c_measure", "output"),),
        inputs=("features",),
        outputs=(output,),
    )
    return graph, {"c_measure": contract}


def _graph_for_use_site(site: SourceUseSite) -> tuple[CoreGraph, Mapping[str, ConsumerContract | ContractResolution]] | None:
    facts = site.facts
    if facts.get("semantic_error"):
        return _semantic_error_graph(site)
    if facts.get("normalization_dependency_unknown"):
        # If a state-preparation path is known, retain the normalization
        # obligation as an executable graph.  This lets the compiler reject
        # an unproved nonzero condition instead of hiding it as an opaque
        # source-summary gap.
        if not (facts.get("manual_norm_divide") and facts.get("state_prep")):
            graph = CoreGraph(
                (Node("state_input", "placeholder", attributes={"api": site.api}),),
                uses=(Use("u_domain", "state_input", "observe", "c_domain", "output"),),
                outputs=("state_input",),
            )
            return graph, {"c_domain": ContractResolution.underspecified(
                "state input depends on division without a proved denominator domain or normalization relation",
                "input_domain", "denominator_relation",
            )}
    if facts.get("boundary_unresolved"):
        graph = CoreGraph(
            (Node("boundary", "placeholder", attributes={"api": site.api}),),
            uses=(Use("u_boundary", "boundary", "observe", "c_boundary", "output"),),
            outputs=("boundary",),
        )
        return graph, {"c_boundary": ContractResolution.underspecified(
            "the boundary needs a return representation and consumer summary", "source_summary", "consumer_kind"
        )}
    if facts.get("controlled_operation"):
        nodes = (
            Node("features", "input", attributes={"shape": (None,)}),
            Node("state", "ket", ("features",), attributes={"source_id": "controlled", "wire_order": tuple(facts.get("wire_order", ())), "shape": (None,), "classical_array": True}),
        )
        if facts.get("phase_unknown") and not facts.get("phase_exact_evidence"):
            contract: ConsumerContract | ContractResolution = ContractResolution.underspecified("controlled use requires exact phase evidence", "phase_certificate")
        else:
            contract = ConsumerContract("c_control", ConsumerKind.CONTROLLED_USE, "controlled_kernel", requires_phase_exact=True)
        graph = CoreGraph(nodes, uses=(Use("u_control", "state", "controlled", "c_control", "output"),), inputs=("features",), outputs=("state",))
        return graph, {"c_control": contract}
    if facts.get("opaque_classical_array"):
        return _opaque_classical_array_graph()
    if facts.get("classical_array_consumer"):
        return _classical_array_graph()
    if facts.get("manual_norm_divide") and (facts.get("state_prep") or facts.get("state_return")):
        if facts.get("expval_observation") or facts.get("sample_observation") or facts.get("probs_return"):
            return _normalized_measurement_graph(site)
        if facts.get("state_prep"):
            return _qdm_graph(facts)
        graph, _ = _qdm_graph(facts)
        # Array observation still depends on successful upstream division.
        graph = CoreGraph(graph.nodes, uses=(Use("u_state", "state", "observe", "c_state", "state"),), inputs=graph.inputs, outputs=("state",))
        return graph, {"c_state": ConsumerContract(
            "c_state", ConsumerKind.CLASSICAL_ANALYSIS, "state_observation",
            requires_classical_array=True, requires_normalized=True,
            allow_guard=bool(facts.get("nonzero_guard")),
        )}
    if facts.get("sample_observation"):
        return _sample_graph(site)
    if facts.get("probs_return"):
        return _probs_graph(site)
    if facts.get("expval_observation"):
        return _expval_graph(bool(facts.get("explicit_normalize")))
    if facts.get("state_return"):
        return _state_graph()
    if facts.get("state_prep"):
        return _prepare_graph(
            bool(facts.get("explicit_normalize") or facts.get("domain_proven")),
            site.source_location,
            allow_guard=bool(facts.get("nonzero_guard")),
            zero_witness=bool(facts.get("zero_norm_witness")),
        )
    return None


def _ambiguous_use_site(candidates: Sequence[SourceUseSite], spans: Sequence[SourceSpan]) -> SourceUseSite:
    """Create a provenance-preserving boundary for an unresolved choice."""

    ranked = sorted(
        candidates,
        key=lambda site: (_span_score(site, spans), site.source_path, site.line, site.end_line),
        reverse=True,
    )
    representative = ranked[0]
    candidate_apis = tuple(sorted({site.api for site in candidates}))
    return SourceUseSite(
        qualified_name=representative.qualified_name,
        api="ambiguous:" + "|".join(candidate_apis[:8]),
        consumer_kind=ConsumerKind.CLASSICAL_ANALYSIS.value,
        line=representative.line,
        end_line=representative.end_line,
        source_path=representative.source_path,
        source_location=representative.source_location,
        confidence="dataflow",
        facts={
            "contract_ambiguous": True,
            "candidate_count": len(candidates),
            "candidate_apis": candidate_apis,
        },
    )


def _import_tree(
    tree: ast.AST,
    source_id: str,
    source_digest: str,
    registry: EvidenceRegistry,
    entry_points: Tuple[str, ...] | None = None,
    evidence_spans: Sequence[SourceSpan] = (),
) -> SourceImportResult:
    facts = _facts(tree, entry_points)
    aliases, frameworks = _api_aliases(tree)
    candidates = _candidate_use_sites(tree, aliases, frameworks)
    evidence_spans = _enclosing_evidence_spans(tree, evidence_spans)
    finalists: list[SourceUseSite] = []
    confidence_rank = {"dataflow": 3, "wrapper": 2, "direct": 1}
    selected: SourceUseSite | None = None
    # Prefer a safety boundary anchored by the supplied source evidence.  A
    # complete checkout may contain several independent quantum helpers; a
    # repository-wide fact must not select an unrelated helper merely because
    # it has a more explicit API spelling.
    preferred_paths = {span.path for span in evidence_spans if span.weight >= 1}
    safety_sources = facts.get("semantic_error_sources", {})
    for error, paths in safety_sources.items():
        matching_paths = set(paths) & preferred_paths
        if error == "ProbabilityMassThreshold" and not matching_paths:
            matching_paths = {
                path
                for path in preferred_paths
                if any(
                    isinstance(node, ast.Call)
                    and _dotted(node.func).rsplit(".", 1)[-1].lower() == "random_index"
                    and _node_source_path(node) == path
                    for node in ast.walk(tree)
                )
            }
        if not matching_paths:
            continue
        path = sorted(matching_paths)[0]
        span = next((item for item in evidence_spans if item.path == path and item.weight >= 1), None)
        line = span.start_line if span is not None else 1
        end_line = span.end_line if span is not None else line
        facts_for_error = {
            "semantic_error": error,
            "semantic_errors": (error,),
            "semantic_error_sources": {error: tuple(sorted(paths))},
            "expval_observation": error in {"FiniteShotObservableBasisMismatch", "NonAdjacentWireMapping"},
            "sample_observation": error == "ProbabilityMassThreshold",
            "probs_return": error == "ProbabilityEventAxisMismatch",
        }
        selected = SourceUseSite(
            qualified_name="<source-safety>",
            api=f"source_safety:{error}",
            consumer_kind=ConsumerKind.MEASURE.value,
            line=line,
            end_line=end_line,
            source_path=path,
            source_location=f"{path}:{line}-{end_line}",
            confidence="dataflow",
            facts=facts_for_error,
        )
        break
    if evidence_spans and selected is None:
        # Evidence anchoring is the primary disambiguator.  A high-confidence
        # wrapper candidate elsewhere in a large repository must not hide a
        # directly observed API at the RQ0 line range.
        scored = sorted(
            candidates,
            key=lambda site: (_span_score(site, evidence_spans), confidence_rank.get(site.confidence, 0)),
            reverse=True,
        )
        selected = None
        if scored:
            best_score = _span_score(scored[0], evidence_spans)
            if best_score > 0:
                best = [site for site in scored if _span_score(site, evidence_spans) == best_score]
                best_confidence = max(confidence_rank.get(site.confidence, 0) for site in best)
                finalists = [site for site in best if confidence_rank.get(site.confidence, 0) == best_confidence]
                # A composite boundary carries both the preparation and the
                # actual consumer contract.  Prefer it over a same-location
                # decorator or wrapper fragment when the evidence ties them;
                # retain ambiguity when the tied candidates are different
                # consumers (for example expval versus probs).
                composite_finalists = [
                    site
                    for site in finalists
                    if "composite:" in site.api
                    and site.facts.get("state_prep")
                    and (
                        site.facts.get("state_return")
                        or site.facts.get("expval_observation")
                        or site.facts.get("sample_observation")
                        or site.facts.get("probs_return")
                    )
                ]
                if composite_finalists:
                    finalists = composite_finalists
                # Multiple source locations are safe to merge only when they
                # imply the same Core contract.  This handles repeated helper
                # implementations while keeping expval-vs-probs ambiguity
                # unresolved.
                if (
                    len(finalists) == 1
                    or len({_candidate_semantic_signature(site) for site in finalists}) == 1
                ):
                    selected = sorted(finalists, key=lambda site: (site.source_path, site.line, site.end_line))[0]
    elif not evidence_spans:
        scored = sorted(candidates, key=lambda site: (confidence_rank.get(site.confidence, 0), _span_score(site, evidence_spans)), reverse=True)
        selected = scored[0] if scored else None
    if selected is None and candidates and facts.get("semantic_error"):
        # A proven source safety violation is stronger than an unresolved
        # consumer tie: the compiler must still see a resolved blocked
        # contract instead of allowing an ambiguity to hide the violation.
        ranked = sorted(
            candidates,
            key=lambda site: (_span_score(site, evidence_spans), confidence_rank.get(site.confidence, 0)),
            reverse=True,
        )
        selected = ranked[0]
    if evidence_spans and selected is None and candidates:
        viable = [site for site in finalists if _graph_for_use_site(site) is not None]
        best_score = _span_score(scored[0], evidence_spans) if scored else 0
        if best_score > 0 and len(viable) >= 2:
            selected = _ambiguous_use_site(viable, evidence_spans)
            graph, contracts = _ambiguous_consumer_graph()
            source_ref = _evidence(
                source_id,
                source_digest,
                registry,
                facts,
                (selected.qualified_name,),
                qualified_entry=selected.qualified_name,
                derivation_rule="evidence_scored_ast_ambiguous_use_site",
            )
            return SourceImportResult(
                graph,
                contracts,
                facts,
                source_ref,
                ("EvidenceUseSiteAmbiguous",),
                tuple(candidates),
                selected,
            )
    if selected is not None:
        # File-level safety facts are joined only after evidence selection.
        # This keeps unrelated boundaries out of candidate ranking while still
        # allowing a selected wrapper/QNode to inherit a proven violation from
        # its bounded dependency closure.
        selected_facts = dict(selected.facts)
        if facts.get("semantic_error") and not selected_facts.get("semantic_error"):
            selected_facts["semantic_error"] = facts["semantic_error"]
            selected_facts["semantic_errors"] = tuple(facts.get("semantic_errors", ()))
        if facts.get("manual_norm_divide") and facts.get("state_prep"):
            if selected_facts.get("normalization_dependency_unknown"):
                selected_facts["manual_norm_divide"] = True
                selected_facts["state_prep"] = True
                selected_facts["normalization_division"] = True
        if selected_facts != dict(selected.facts):
            selected = SourceUseSite(
                qualified_name=selected.qualified_name,
                api=selected.api,
                consumer_kind=selected.consumer_kind,
                line=selected.line,
                end_line=selected.end_line,
                source_path=selected.source_path,
                source_location=selected.source_location,
                confidence=selected.confidence,
                facts=selected_facts,
            )
        graph_bundle = _graph_for_use_site(selected)
        source_ref = _evidence(
            source_id,
            source_digest,
            registry,
            facts,
            (selected.qualified_name,),
            qualified_entry=selected.qualified_name,
            derivation_rule="evidence_scored_ast_use_site_dataflow",
        )
        if graph_bundle is not None:
            graph, contracts = graph_bundle
            diagnostics = ("BoundarySummaryRequired",) if selected.facts.get("boundary_unresolved") else ()
            return SourceImportResult(graph, contracts, facts, source_ref, diagnostics, tuple(candidates), selected)
    if evidence_spans:
        source_ref = _evidence(
            source_id,
            source_digest,
            registry,
            facts,
            entry_points,
            derivation_rule="evidence_scored_ast_use_site_unresolved",
        )
        diagnostic = "EvidenceUseSiteNotResolved" if candidates else "NoCandidateUseSite"
        return SourceImportResult(None, {}, facts, source_ref, (diagnostic,), tuple(candidates), None)
    if facts["manual_norm_divide"] and facts["state_prep"] and facts["state_return"]:
        graph, contracts = _qdm_graph()
    elif facts["sample_observation"]:
        graph, contracts = _sample_graph()
    elif facts["probs_return"]:
        graph, contracts = _probs_graph()
    elif facts["expval_observation"] and (facts["amplitude_embedding"] or facts["state_prep"]):
        graph, contracts = _expval_graph(facts["explicit_normalize"])
    elif facts["expval_observation"]:
        graph, contracts = _expval_graph(False)
    elif facts["state_observation"]:
        graph, contracts = _state_graph()
    else:
        source_ref = _evidence(source_id, source_digest, registry, facts, entry_points)
        return SourceImportResult(None, {}, facts, source_ref, ("NoSupportedQuantumBoundary",), tuple(candidates), selected)
    source_ref = _evidence(source_id, source_digest, registry, facts, entry_points)
    return SourceImportResult(graph, contracts, facts, source_ref, (), tuple(candidates), selected)


def import_source(
    path: str | Path,
    registry: EvidenceRegistry | None = None,
    *,
    entry_points: Tuple[str, ...] | None = None,
    evidence_spans: Sequence[SourceSpan] = (),
) -> SourceImportResult:
    source_path = Path(path)
    registry = registry or EvidenceRegistry()
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8", errors="replace"), filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        return SourceImportResult(None, {}, {}, None, (f"SourceParseError:{type(exc).__name__}:{exc}",), (), None, (), (str(source_path),))
    _mark_source(tree, str(source_path))
    result = _import_tree(tree, str(source_path), _source_digest(source_path), registry, entry_points, evidence_spans)
    return SourceImportResult(
        result.graph,
        result.contracts,
        result.facts,
        result.source_ref,
        result.diagnostics,
        result.use_sites,
        result.selected_use_site,
        (str(source_path),),
        (),
    )


def import_sources(
    paths: Tuple[str | Path, ...],
    registry: EvidenceRegistry | None = None,
    *,
    entry_points: Tuple[str, ...] | None = None,
    evidence_spans: Sequence[SourceSpan] = (),
) -> SourceImportResult:
    """Import a small multi-file source unit while retaining file provenance."""

    registry = registry or EvidenceRegistry()
    modules: list[ast.stmt] = []
    digest_input = bytearray()
    source_ids: list[str] = []
    diagnostics: list[str] = []
    parsed_files: list[str] = []
    skipped_files: list[str] = []
    for raw_path in paths:
        source_path = Path(raw_path)
        try:
            text = source_path.read_text(encoding="utf-8", errors="replace")
            module = ast.parse(text, filename=str(source_path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            skipped_files.append(str(source_path))
            diagnostics.append(f"SourceParseError:{source_path}:{type(exc).__name__}:{exc}")
            continue
        _mark_source(module, str(source_path))
        modules.extend(module.body)
        source_ids.append(str(source_path))
        parsed_files.append(str(source_path))
        digest_input.extend(text.encode("utf-8"))
        digest_input.extend(b"\0")
    if not modules:
        return SourceImportResult(None, {}, {}, None, tuple(diagnostics or ("NoParseableSourceFiles",)), (), None, tuple(parsed_files), tuple(skipped_files))
    tree = ast.Module(body=modules, type_ignores=[])
    source_id = "|".join(source_ids)
    source_digest = hashlib.sha256(bytes(digest_input)).hexdigest()
    result = _import_tree(tree, source_id, source_digest, registry, entry_points, evidence_spans)
    return SourceImportResult(
        result.graph,
        result.contracts,
        result.facts,
        result.source_ref,
        tuple(diagnostics) + result.diagnostics,
        result.use_sites,
        result.selected_use_site,
        tuple(parsed_files),
        tuple(skipped_files),
    )
