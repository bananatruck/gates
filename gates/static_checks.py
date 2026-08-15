"""Pre-execution source analysis.

These checks run before a single line of the experiment executes, so rejecting a
broken program costs no compute. They are deliberately conservative: a false
positive here blocks a valid experiment, so every rule below only fires on facts
the language guarantees.
"""

from __future__ import annotations

import ast
import builtins
import symtable
from dataclasses import dataclass

#: Builtins that fold a constant into another constant. ``float(0.816)`` is a
#: literal wearing a hat.
_CONSTANT_CONVERTERS = frozenset(
    {"float", "int", "round", "abs", "str", "bool", "complex", "len"}
)

_BANNED_CALLS = {
    ("exit", None): "exit()",
    ("quit", None): "quit()",
    ("exit", "sys"): "sys.exit()",
    ("_exit", "os"): "os._exit()",
    ("abort", "os"): "os.abort()",
}

_MODULE_DUNDERS = frozenset(
    {
        "__name__",
        "__file__",
        "__doc__",
        "__builtins__",
        "__package__",
        "__loader__",
        "__spec__",
        "__debug__",
    }
)


@dataclass
class UnboundName:
    name: str
    lineno: int
    scope: str
    source_line: str = ""

    def where(self) -> str:
        return f"line {self.lineno} in {self.scope}"


@dataclass
class BannedCall:
    call: str
    lineno: int
    source_line: str = ""


def parse(source: str, filename: str = "<experiment>") -> ast.Module:
    """Parse source, letting SyntaxError propagate to the caller."""
    return ast.parse(source, filename=filename)


def find_unbound_names(
    source: str,
    filename: str = "<experiment>",
    extra_bound: frozenset[str] | set[str] = frozenset(),
) -> list[UnboundName]:
    """Names that are read but can never resolve at runtime.

    Uses :mod:`symtable` rather than a hand-rolled AST walk so that
    comprehensions, ``global``/``nonlocal``, closures, walrus bindings and class
    scopes are handled by the same machinery the interpreter uses.

    A name is reported only when it is *referenced* in some scope, resolves to
    the module namespace (not local, not a parameter, not a closure cell, not
    imported), and is never bound anywhere at module level nor by a ``global``
    assignment in a nested scope. That is the ``hidden_dim`` case: read inside
    ``forward``, assigned nowhere.
    """
    table = symtable.symtable(source, filename, "exec")
    lines = source.splitlines()

    bound: set[str] = set(dir(builtins))
    bound |= set(_MODULE_DUNDERS)
    bound |= set(extra_bound)

    scopes: list[tuple[str, symtable.SymbolTable]] = []
    _collect_scopes(table, "<module>", scopes)

    # Pass 1 — everything that ends up bound in the module namespace.
    for scope_name, scope in scopes:
        is_module = scope.get_type() == "module"
        for sym in scope.get_symbols():
            binds = sym.is_assigned() or sym.is_imported()
            if not binds:
                continue
            # Module-level bindings, and nested `global x; x = ...` bindings.
            if is_module or sym.is_global():
                bound.add(sym.get_name())

    # Pass 2 — references that cannot resolve against that namespace.
    unbound: list[UnboundName] = []
    seen: set[str] = set()
    for scope_name, scope in scopes:
        if scope.get_type() == "module":
            # Use-before-assignment at module level is a runtime ordering
            # question, not a resolution question. Out of scope: we would
            # produce false positives on conditional definitions.
            continue
        for sym in scope.get_symbols():
            name = sym.get_name()
            if not sym.is_referenced() or name in bound or name in seen:
                continue
            if sym.is_assigned() or sym.is_parameter() or sym.is_imported():
                continue
            if sym.is_free() or sym.is_local():
                continue
            seen.add(name)
            lineno = _first_reference_line(source, filename, name, scope_name)
            unbound.append(
                UnboundName(
                    name=name,
                    lineno=lineno,
                    scope=scope_name,
                    source_line=_line_at(lines, lineno),
                )
            )

    unbound.sort(key=lambda u: (u.lineno or 0, u.name))
    return unbound


def find_banned_calls(source: str, filename: str = "<experiment>") -> list[BannedCall]:
    """Calls that forge a clean exit code and would defeat the exit-code check."""
    tree = parse(source, filename)
    lines = source.splitlines()
    found: list[BannedCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        key = _call_key(node.func)
        if key in _BANNED_CALLS:
            found.append(
                BannedCall(
                    call=_BANNED_CALLS[key],
                    lineno=node.lineno,
                    source_line=_line_at(lines, node.lineno),
                )
            )
    return found


@dataclass(frozen=True)
class ShadowedName:
    """A harness-injected name the experiment defined for itself."""

    name: str
    lineno: int
    kind: str  # "function" | "assignment" | "import"
    source_line: str


def find_shadowed_harness_names(
    source: str,
    names: frozenset[str] = frozenset({"record_result", "record_metadata"}),
    filename: str = "<experiment>",
) -> list[ShadowedName]:
    """Definitions that shadow the API the harness injected.

    Observed live, and it is the reason this check exists. A code model wrote:

        def record_result(key, value, unit=None):
            print(f"{key}: {value}")

    then called it four times and exited 0. Every value went to its own stub,
    the harness recorded nothing, and the run was rejected on
    ``results.contract_present`` -- "the experiment never called
    record_result()". True of the harness's function, and misleading about what
    the agent did: it called one, its own, and believed it was recording.

    Caught statically, so the rejection costs no execution and the message can
    name the actual mistake.
    """
    tree = parse(source, filename)
    lines = source.splitlines()
    found: list[ShadowedName] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in names:
                found.append(ShadowedName(node.name, node.lineno, "function",
                                          _line_at(lines, node.lineno)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found.append(ShadowedName(target.id, node.lineno, "assignment",
                                              _line_at(lines, node.lineno)))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in names:
                    found.append(ShadowedName(bound, node.lineno, "import",
                                              _line_at(lines, node.lineno)))
    return sorted(found, key=lambda f: f.lineno)


def classify_record_calls(
    source: str, filename: str = "<experiment>", func_name: str = "record_result"
) -> dict[int, str]:
    """Map each ``record_result`` call line to ``"computed"`` or ``"literal"``.

    This is the mechanical form of "are the variables real". A recorded value
    whose expression contains no name, attribute, subscript or non-folding call
    was typed into the source rather than measured by the run.
    """
    tree = parse(source, filename)
    kinds: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_name(node.func) != func_name:
            continue
        value = _value_argument(node)
        if value is None:
            continue
        kind = "literal" if _is_constant_expr(value) else "computed"
        # A line with any computed record call counts as computed; a line with
        # only literal ones is a fabrication.
        if kinds.get(node.lineno) != "computed":
            kinds[node.lineno] = kind
    return kinds


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _collect_scopes(
    table: symtable.SymbolTable, name: str, out: list[tuple[str, symtable.SymbolTable]]
) -> None:
    out.append((name, table))
    for child in table.get_children():
        child_name = child.get_name()
        qualified = child_name if name == "<module>" else f"{name}.{child_name}"
        _collect_scopes(child, qualified, out)


def _first_reference_line(
    source: str, filename: str, name: str, scope_name: str
) -> int:
    """Locate the first read of ``name`` so the feedback report can point at it."""
    try:
        tree = parse(source, filename)
    except SyntaxError:
        return 0
    best = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
            if best == 0 or node.lineno < best:
                best = node.lineno
    return best


def _value_argument(node: ast.Call) -> ast.expr | None:
    for kw in node.keywords:
        if kw.arg == "value":
            return kw.value
    if len(node.args) >= 2:
        return node.args[1]
    return None


def _is_constant_expr(node: ast.expr) -> bool:
    """True when the expression can be evaluated without reading any binding."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_constant_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_constant_expr(node.left) and _is_constant_expr(node.right)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_constant_expr(e) for e in node.elts)
    if isinstance(node, ast.Call):
        name = _called_name(node.func)
        if name in _CONSTANT_CONVERTERS and not node.keywords:
            return all(_is_constant_expr(a) for a in node.args)
        return False
    return False


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_key(func: ast.expr) -> tuple[str | None, str | None]:
    if isinstance(func, ast.Name):
        return (func.id, None)
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else None
        return (func.attr, base)
    return (None, None)


def _line_at(lines: list[str], lineno: int | None) -> str:
    if not lineno or lineno < 1 or lineno > len(lines):
        return ""
    return lines[lineno - 1].rstrip()
