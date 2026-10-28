"""Bounded lexical and import resolution without executing repository code."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Sequence


def source_path(node: ast.AST) -> str:
    return str(getattr(node, "_qtc_source_path", ""))


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


class SourceSymbols:
    """Resolve only lexical, imported, or statically assigned callable targets.

    Unknown receivers never fall back to methods with the same short name in
    unrelated classes or files. Multiple possible imports remain a target set.
    """

    def __init__(self, regions: Sequence[tuple[str, ast.AST]]) -> None:
        self.definitions = {}
        self.bindings = {}
        self.stars = {}
        self.modules = {}
        self.module_paths = {source_path(node) for _, node in regions if source_path(node)}
        for path in sorted(self.module_paths):
            parts = list(Path(path).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            for start in range(len(parts)):
                self.modules.setdefault(".".join(parts[start:]), set()).add(path)
        for name, region in regions:
            path = source_path(region)
            local = name.rsplit("::", 1)[-1]
            scope = "" if local == "<module>" else local
            if scope:
                self.definitions[(path, scope)] = name
            pending = list(ast.iter_child_nodes(region))
            while pending:
                node = pending.pop()
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                node_path = source_path(node) or path
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        local_name = alias.asname or alias.name.split(".")[0]
                        module = alias.name if alias.asname else local_name
                        self.bindings[(node_path, scope, local_name)] = ("import", module, "")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        package = Path(node_path).parent
                        for _ in range(node.level - 1):
                            package = package.parent
                        base = package / module.replace(".", "/")
                        module = str(base)
                    for alias in node.names:
                        if alias.name == "*":
                            self.stars.setdefault((node_path, scope), []).append(module)
                        else:
                            self.bindings[(node_path, scope, alias.asname or alias.name)] = (
                                "import", module, alias.name
                            )
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                    value = node.value
                    for target in targets:
                        target_name = dotted(target)
                        if not target_name or value is None:
                            continue
                        owner_scope = scope.rsplit(".", 1)[0] if target_name.startswith("self.") else scope
                        target_name = target_name.removeprefix("self.")
                        self.bindings[(node_path, owner_scope, target_name)] = ("value", value, scope)
                pending.extend(ast.iter_child_nodes(node))

    def _paths(self, module: str) -> set[str]:
        if module.startswith("/"):
            base = Path(module)
            return {str(path) for path in (base.with_suffix(".py"), base / "__init__.py")
                    if str(path) in self.module_paths}
        return self.modules.get(module, set())

    def targets(self, expression: ast.AST, region_name: str, path: str) -> set[str]:
        scope = region_name.rsplit("::", 1)[-1]
        if scope == "<module>":
            scope = ""
        return self._resolve(expression, path, scope, frozenset())

    def _resolve(self, expression: ast.AST, path: str, scope: str, seen: frozenset) -> set[str]:
        # A call used as a receiver represents a factory result or an instance.
        if isinstance(expression, ast.Call):
            return self._resolve(expression.func, path, scope, seen)
        name = dotted(expression)
        if not name:
            return set()
        key = (path, scope, name)
        if key in seen or len(seen) >= 16:
            return set()
        seen = seen | {key}
        if name.startswith(("self.", "cls.")):
            owner = scope.rsplit(".", 1)[0]
            return self._resolve_name(name.split(".", 1)[1], path, owner, seen)
        return self._resolve_name(name, path, scope, seen)

    def _resolve_name(self, name: str, path: str, scope: str, seen: frozenset) -> set[str]:
        root, _, suffix = name.partition(".")
        scopes = [scope]
        while scopes[-1]:
            scopes.append(scopes[-1].rsplit(".", 1)[0] if "." in scopes[-1] else "")
        for candidate_scope in scopes:
            qualified = f"{candidate_scope}.{name}" if candidate_scope else name
            exact = self.definitions.get((path, qualified))
            if exact:
                return {exact}
            instance_methods = {
                self.definitions[(path, qualified + "." + method)]
                for method in ("__call__", "forward")
                if (path, qualified + "." + method) in self.definitions
            }
            if instance_methods:
                return instance_methods
            binding = self.bindings.get((path, candidate_scope, root))
            if binding:
                kind, value, original = binding
                if kind == "import":
                    exported = ".".join(part for part in (original, suffix) if part)
                    result = set()
                    for target_path in self._paths(value):
                        result.update(self._resolve(ast.Name(id=exported), target_path, "", seen))
                    # ``from package import module`` also names a submodule.
                    if original:
                        submodule = value + ("/" if value.startswith("/") else ".") + original
                        for target_path in self._paths(submodule):
                            result.update(self._resolve(ast.Name(id=suffix), target_path, "", seen))
                    return result
                expr = value.func if isinstance(value, ast.Call) else value
                if suffix:
                    base = dotted(expr)
                    expr = ast.Name(id=f"{base}.{suffix}") if base else expr
                return self._resolve(expr, path, original, seen)
        result = set()
        for module in self.stars.get((path, ""), ()):
            for target_path in self._paths(module):
                result.update(self._resolve(ast.Name(id=name), target_path, "", seen))
        return result
