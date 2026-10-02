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

#: Calls whose result is fixed by their arguments, so all-constant arguments
#: give a constant: ``sum([0.95])`` is as typed as ``0.95``. Called bare, on a
#: module (``np.mean``, ``statistics.mean``) or on a constant receiver.
_PURE_CALLS = frozenset(
    {
        "sum", "min", "max", "sorted", "mean", "median", "fmean", "average",
        "std", "var", "stdev", "pstdev", "array", "asarray", "tensor",
    }
)

#: Methods that read a container without changing it. Any other method called
#: on a container may change it, so its arguments join what the container holds.
_READ_METHODS = frozenset(
    {"get", "copy", "keys", "values", "items", "index", "count", "item", "tolist"}
) | _PURE_CALLS

#: Calls that read the containers passed to them and never change them:
#: printing and formatting, JSON, a logger's level methods, and the harness's
#: own record calls. A container passed to any other call may be filled by it
#: (B2).
_READ_CALLS = frozenset(
    {
        "print", "format", "repr", "pprint", "pformat",
        "dump", "dumps",
        "debug", "info", "warning", "error", "critical", "exception",
        "record_result", "record_setting", "record_metadata",
    }
)

#: How measured a call site is. A line holding several ``record_result``
#: calls takes the highest, so one real measurement is never hidden behind a
#: constant recorded on the same line.
_KIND_RANK = {"literal": 0, "constant": 1, "computed": 2}

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
    for _scope_name, scope in scopes:
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
    names: frozenset[str] = frozenset({"record_result", "record_setting", "record_metadata"}),
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
    """Map each ``record_result`` call line to how its value was obtained.

    This is the mechanical form of "are the variables real". Three answers:

    ``literal``
        The call site is a constant expression — ``record_result("k", 0.816)``.
        Typed, not measured, and a blocking failure.
    ``constant``
        The call site reads names, but every binding of every name it reads is
        itself constant — ``acc = 0.816`` then ``record_result("k", acc)``. The
        indirection is the only difference from ``literal``, so the call-site
        check alone cannot see it. A configured value belongs in
        ``record_setting``, so as a result this fails too (D75).
    ``computed``
        Something the run produced reaches the value.

    The name resolution is deliberately shallow. A name that is ever bound by
    anything other than a plain assignment is treated as computed, so the pass
    under-reports rather than accusing a real measurement. A literal kept in a
    dict or list, or passed through ``sum``, ``max``, ``np.mean`` and the other
    :data:`_PURE_CALLS`, is followed; a container the run fills after binding
    it is not constant (:func:`_container_contents`).
    """
    # limit: a function the program defines is followed only when its returns
    # are constant whatever it is passed, so def m(n): return n then m(0.95)
    # reads as computed; binding arguments to parameters would follow it.
    tree = parse(source, filename)
    names = _record_names(tree, func_name)
    bindings = _constant_bindings(tree, func_name)
    kinds: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _called_name(node.func)
        if called != func_name and not (
            isinstance(node.func, ast.Name) and called in names
        ):
            continue
        value = _value_argument(node)
        if value is None:
            continue
        if _is_constant_expr(value):
            kind = "literal"
        elif _is_literal_derived(value, bindings, frozenset()):
            kind = "constant"
        else:
            kind = "computed"
        # One line can hold several calls. The most-measured one wins, so a line
        # is only reported as literal when every value on it is.
        if _KIND_RANK[kind] > _KIND_RANK.get(kinds.get(node.lineno, ""), -1):
            kinds[node.lineno] = kind
    return kinds


def find_unused_record_values(
    source: str, filename: str = "<experiment>", func_name: str = "record_result"
) -> set[int]:
    """Lines whose recorded value is read by nothing but a record or print call.

    The decoy (B8): ``lr = 0.001``, the optimizer built with ``0.01``, then
    ``record_result("config.lr", lr)``. The recorded value matches the plan and
    the computation never saw it. A line is returned only when no name its value
    reads is read anywhere else, so one real use clears it. A call-site literal
    reads no name and is never returned; ``classify_record_calls`` owns that.
    """
    # limit: names only. cfg.lr recorded while cfg.batch is used reads as
    # used, and so does a value passed to a logger; follow attributes and
    # subscripts if a decoy is ever caught hiding behind one.
    tree = parse(source, filename)
    ignored: set[int] = set()
    recorded: dict[int, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _called_name(node.func)
        if called not in {func_name, "print"}:
            continue
        ignored.update(id(n) for n in ast.walk(node) if isinstance(n, ast.Name))
        value = _value_argument(node) if called == func_name else None
        if value is not None:
            recorded.setdefault(node.lineno, set()).update(
                n.id
                for n in ast.walk(value)
                if isinstance(n, ast.Name) and not hasattr(builtins, n.id)
            )
    read = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in ignored
    }
    return {line for line, names in recorded.items() if names and not names & read}


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
        if (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
                and (best == 0 or node.lineno < best)):
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
    return _is_literal_derived(node, {}, frozenset())


def _is_literal_derived(
    node: ast.expr, bindings: dict[str, list[ast.expr]], seen: frozenset[str]
) -> bool:
    """True when every leaf of ``node`` is a literal, following ``bindings``.

    With an empty ``bindings`` this is the call-site test: nothing but constants
    and constant-folding conversions. With bindings supplied it follows each
    name back to its assignments, which is what catches a literal laundered
    through a variable. ``seen`` breaks the cycle a self-referential binding
    would otherwise create.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        if node.id in seen or node.id not in bindings:
            return False
        deeper = seen | {node.id}
        return all(_is_literal_derived(v, bindings, deeper) for v in bindings[node.id])
    if isinstance(node, ast.UnaryOp):
        return _is_literal_derived(node.operand, bindings, seen)
    if isinstance(node, ast.BinOp):
        return _is_literal_derived(node.left, bindings, seen) and _is_literal_derived(
            node.right, bindings, seen
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_derived(e, bindings, seen) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is None or _is_literal_derived(k, bindings, seen) for k in node.keys
        ) and all(_is_literal_derived(v, bindings, seen) for v in node.values)
    if isinstance(node, ast.Subscript):
        return _is_literal_derived(node.value, bindings, seen) and _is_literal_derived(
            node.slice, bindings, seen
        )
    if isinstance(node, ast.Slice):
        parts = (node.lower, node.upper, node.step)
        return all(p is None or _is_literal_derived(p, bindings, seen) for p in parts)
    if isinstance(node, ast.Call):
        return _is_constant_call(node, bindings, seen)
    return False


def _is_constant_call(
    node: ast.Call, bindings: dict[str, list[ast.expr]], seen: frozenset[str]
) -> bool:
    """True when the call's result is fixed by arguments that are all constant."""
    name = _called_name(node.func)
    if isinstance(node.func, ast.Name) and _returns_key(name) in bindings:
        # A function the program defines whose every return is constant,
        # whatever it was passed (S14). ``seen`` stops a recursive one.
        key = _returns_key(name)
        return key not in seen and all(
            _is_literal_derived(r, bindings, seen | {key}) for r in bindings[key]
        )
    args = all(_is_literal_derived(a, bindings, seen) for a in node.args)
    keywords = all(_is_literal_derived(k.value, bindings, seen) for k in node.keywords)
    if isinstance(node.func, ast.Name):
        if name in _CONSTANT_CONVERTERS:
            return args and keywords
        return name in _PURE_CALLS and bool(node.args) and args and keywords
    if not isinstance(node.func, ast.Attribute) or name not in _READ_METHODS:
        return False
    receiver = node.func.value
    if _is_literal_derived(receiver, bindings, seen):
        return args and keywords
    # A module, such as np or statistics: the arguments alone decide.
    namespace = _is_dotted_name(receiver) and _root_name(receiver) not in bindings
    return namespace and name in _PURE_CALLS and bool(node.args) and args and keywords


def _is_dotted_name(node: ast.expr) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _root_name(node: ast.expr) -> str | None:
    """The name at the bottom of ``a.b[c].d``, or ``None`` when there is none."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _constant_bindings(
    tree: ast.AST, record_name: str = "record_result"
) -> dict[str, list[ast.expr]]:
    """Every name bound only by plain assignment, mapped to what it was assigned.

    A name bound by anything this pass cannot evaluate — a loop target, a
    parameter, an augmented assignment, an import, a ``with`` or ``except``
    alias, a ``global`` declaration — is dropped entirely rather than partially
    resolved. Dropping it means the name reads as computed, which is the safe
    direction: a constant result fails the run (D75), so a false one costs an
    honest engineer a rejection.
    """
    values: dict[str, list[ast.expr]] = {}
    opaque: set[str] = set()
    functions: dict[str, list[ast.FunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values.setdefault(target.id, []).append(node.value)
                elif _pairs(target, node.value):
                    # acc, _ = 0.95, 0: one name per element of a display of
                    # the same length, so each value reaches its name (S13).
                    for name, element in zip(target.elts, node.value.elts, strict=True):
                        values.setdefault(name.id, []).append(element)
                else:
                    # Starred or nested unpacking, an unpacked call result,
                    # attributes, subscripts: the binding is real but which
                    # value reaches which name is not this pass's business.
                    opaque.update(_stored_names(target))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                if node.value is None:
                    opaque.add(node.target.id)
                else:
                    values.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            values.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor, ast.comprehension)):
            opaque.update(_stored_names(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                opaque.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, []).append(node)
            opaque.update(_argument_names(node.args))
        elif isinstance(node, ast.Lambda):
            opaque.update(_argument_names(node.args))
        elif isinstance(node, ast.ClassDef):
            opaque.add(node.name)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            opaque.update(_stored_names(node.optional_vars))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            opaque.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            opaque.update(node.names)
    filled, contents = _container_contents(tree, values, _record_names(tree, record_name))
    opaque |= filled
    for name, exprs in contents.items():
        values[name] = [*values[name], *exprs]
    returns = {
        _returns_key(name): _own_returns(defs[0])
        for name, defs in functions.items()
        if _followable(defs, name, values, opaque)
    }
    # A function's name is a binding like any other, never a constant value.
    opaque.update(functions)
    kept = {name: exprs for name, exprs in values.items() if name not in opaque}
    return {**kept, **returns}


def _pairs(target: ast.expr, value: ast.expr) -> bool:
    """A flat unpacking of names from a display of exactly as many elements."""
    return (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and all(isinstance(e, ast.Name) for e in target.elts)
    )


def _returns_key(name: str | None) -> str:
    """Where a defined function's returns are kept among the bindings.

    The parentheses keep it apart from every name a program can bind.
    """
    return f"{name}()"


def _own_nodes(func: ast.AST) -> list[ast.AST]:
    """Every node of a function's body, not descending into a nested scope."""
    out: list[ast.AST] = []
    stack = list(getattr(func, "body", []))
    while stack:
        node = stack.pop()
        out.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


def _own_returns(func: ast.FunctionDef) -> list[ast.expr]:
    """What the function can return; falling off its end returns None."""
    returns = [
        node.value if node.value is not None else ast.Constant(None)
        for node in _own_nodes(func)
        if isinstance(node, ast.Return)
    ]
    return returns or [ast.Constant(None)]


def _followable(
    defs: list[ast.AST], name: str, values: dict[str, list[ast.expr]], opaque: set[str]
) -> bool:
    """A function whose returns stand for its calls: defined once, plainly.

    Defined twice, rebound, decorated, async or a generator, a call to it may
    not return what its body says, so it stays opaque, which reads as computed.
    """
    if len(defs) != 1 or not isinstance(defs[0], ast.FunctionDef):
        return False
    func = defs[0]
    if func.decorator_list or name in values or name in opaque:
        return False
    return not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in _own_nodes(func))


def _record_names(tree: ast.AST, func_name: str) -> set[str]:
    """``func_name`` and every name bound only to it, such as ``rr = record_result``.

    A name with any other binding, or bound to anything but the call or one of
    its aliases, is no alias: it may not be the recording call when it runs.
    """
    assigned: dict[str, list[ast.expr]] = {}
    plain: dict[str, int] = {}
    stores: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            name = node.targets[0].id
            assigned.setdefault(name, []).append(node.value)
            plain[name] = plain.get(name, 0) + 1
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            stores[node.id] = stores.get(node.id, 0) + 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores[node.name] = stores.get(node.name, 0) + 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                stores[bound] = stores.get(bound, 0) + 1
        elif isinstance(node, ast.arg):
            stores[node.arg] = stores.get(node.arg, 0) + 1
    names = {func_name}
    grew = True
    while grew:
        grew = False
        for name, exprs in assigned.items():
            if (
                name not in names
                and stores.get(name) == plain[name]
                and all(isinstance(e, ast.Name) and e.id in names for e in exprs)
            ):
                names.add(name)
                grew = True
    return names


_CONTAINER_DISPLAYS = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)

#: A change inside one of these may run any number of times.
_LOOPS = (
    ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp
)


def _container_contents(
    tree: ast.AST, values: dict[str, list[ast.expr]], record_names: set[str]
) -> tuple[set[str], dict[str, list[ast.expr]]]:
    """What the run puts into each container after binding it.

    ``results = {}`` then ``results["acc"] = evaluate()`` is a measurement,
    though every assignment to the name itself is a literal. So a container
    reads as constant only when everything put into it is constant too: every
    key and value stored into it, and every argument of a method on it that is
    not a read (B3). Those expressions are returned as extra bindings of the
    name, and the ordinary constant test follows them, exactly as it follows a
    second plain assignment to a name. ``results["acc"] = 0.95`` and
    ``accs.sort()`` put in nothing measured; ``accs.append(acc)`` puts in
    whatever ``acc`` is.

    Everything else that changes a container makes the name opaque, as ``+=``
    makes a name opaque: an augmented store, a store the pass cannot pair with
    one value (tuple unpacking, a loop target), any change made inside a loop,
    and being passed to a call, which may fill it. The loop rule is what keeps
    a count honest: ``hits.append(1)`` once per correct prediction puts in
    nothing but a constant, and ``len(hits)`` is still a measurement. A call in
    :data:`_READ_CALLS` only reads, so printing, logging or dumping a container
    before recording from it changes nothing (B2).

    An alias shares its container, so ``r = results; r["acc"] = x`` fills
    ``results``. Only a name bound to a container display takes part: a float
    cannot change, and an alias reads the container through its binding.
    """
    # limit: a container passed to any call outside _READ_CALLS is assumed
    # filled, so wandb.log(results) or a helper the program defines still
    # launders a literal dict as computed; knowing what an arbitrary callee
    # does with its argument would need interprocedural analysis.
    # limit: any change inside a loop is opaque, so accs.append(0.95) in a
    # loop launders as computed; a loop that repeats a constant cannot be told
    # from one that counts without knowing what decides its iterations.
    # limit: a change in a function body is followed as if it ran once, so a
    # helper that appends a constant and is called from a loop reads as
    # constant, as a constant assigned in a branch does for a plain name.
    looped = {
        id(inner)
        for loop in ast.walk(tree)
        if isinstance(loop, _LOOPS)
        for inner in ast.walk(loop)
        if inner is not loop
    }
    contents: dict[str, list[ast.expr]] = {}
    unpaired: set[str] = set()
    paired: set[int] = set()

    def put(node: ast.AST, container: ast.expr, exprs: list[ast.expr]) -> None:
        if id(node) in looped:
            unpaired.add(_root_name(container))
        else:
            contents.setdefault(_root_name(container), []).extend(exprs)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript):
                    paired.add(id(target))
                    put(node, target.value, [node.value, target.slice])
                elif isinstance(target, ast.Attribute):
                    paired.add(id(target))
                    put(node, target.value, [node.value])
        elif isinstance(node, ast.Call):
            func = node.func
            args = [*node.args, *(k.value for k in node.keywords)]
            if isinstance(func, ast.Attribute) and func.attr not in _READ_METHODS:
                put(node, func.value, args)
            name = _called_name(func)
            if (
                name in record_names
                or name in _READ_CALLS
                or name in _PURE_CALLS
                or name in _CONSTANT_CONVERTERS
            ):
                continue
            unpaired.update(_root_name(arg) for arg in args)
    # An augmented store, tuple unpacking, a loop or with target: nothing
    # above paired it with the one value it stores.
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Subscript, ast.Attribute))
            and isinstance(node.ctx, ast.Store)
            and id(node) not in paired
        ):
            unpaired.add(_root_name(node.value))

    shared = {name: list(exprs) for name, exprs in contents.items()}
    filled = set(unpaired)
    for name in set(contents) | unpaired:
        for other in _aliases(name, values):
            shared.setdefault(other, []).extend(contents.get(name, []))
            if name in unpaired:
                filled.add(other)
    containers = {
        name
        for name, exprs in values.items()
        if any(isinstance(e, _CONTAINER_DISPLAYS) for e in exprs)
    }
    return filled & containers, {
        name: exprs for name, exprs in shared.items() if name in containers and exprs
    }


def _aliases(name: str | None, values: dict[str, list[ast.expr]]) -> set[str]:
    """Every name ``name`` is bound to through plain ``a = b`` assignments."""
    found: set[str] = set()
    stack = [name]
    while stack:
        for expr in values.get(stack.pop(), []):
            if isinstance(expr, ast.Name) and expr.id != name and expr.id not in found:
                found.add(expr.id)
                stack.append(expr.id)
    return found


def _stored_names(target: ast.expr) -> set[str]:
    return {
        n.id
        for n in ast.walk(target)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }


def _argument_names(args: ast.arguments) -> set[str]:
    every = [
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *([args.vararg] if args.vararg else []),
        *([args.kwarg] if args.kwarg else []),
    ]
    return {a.arg for a in every}


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
