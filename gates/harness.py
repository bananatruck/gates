"""Child-process execution harness. Runs the experiment; records what happened.

This module is executed as a standalone script in a fresh interpreter:

    python -m gates.harness <code_path> <artifact_dir>

It therefore imports **stdlib only** — no ``gates`` imports, no third-party
packages — so it can be dropped into any scaffold's environment. It emits JSON
matching ``gates.schema.SCHEMA_VERSION``; the parent parses it back.

Two properties matter and are provided by construction:

1. The experiment runs in an empty ``__main__`` namespace. Nothing an earlier
   attempt bound is visible, so a name that only existed because a previous run
   defined it now raises ``NameError`` instead of silently resolving.
2. ``record_result`` captures its own call site, which lets the parent decide
   statically whether the recorded value was computed or typed.
"""

from __future__ import annotations

import builtins
import json
import linecache
import os
import platform
import sys
import traceback

SCHEMA_VERSION = "1.0"

_METRICS: "dict[str, dict]" = {}
_METADATA: "dict[str, object]" = {}
_CODE_PATH = ""


def _record_result(key, value, unit=None):
    """Declare a result. The parent gate verifies it was computed, not typed."""
    if not isinstance(key, str) or not key:
        raise ValueError("record_result: key must be a non-empty string")
    frame = sys._getframe(1)
    lineno = frame.f_lineno
    filename = frame.f_code.co_filename
    source_line = linecache.getline(filename, lineno).rstrip()

    existing = _METRICS.get(key)
    _METRICS[key] = {
        "key": key,
        "value": _jsonable(value),
        "unit": unit,
        "lineno": lineno,
        "source_line": source_line,
        "call_count": (existing["call_count"] + 1) if existing else 1,
    }
    return value


def _record_metadata(key, value):
    """Attach free-form provenance (seed, split sizes, device) to the run."""
    _METADATA[str(key)] = _jsonable(value)
    return value


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    # numpy scalars, torch scalars and anything else with .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _environment():
    env = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    for name in ("torch", "numpy", "sklearn", "scipy", "pandas", "matplotlib"):
        module = sys.modules.get(name)
        if module is not None:
            env[f"{name}_version"] = getattr(module, "__version__", "unknown")
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["torch_initial_seed"] = int(torch.initial_seed())
            env["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            pass
    return env


def _exception_record(exc, code_path):
    tb = exc.__traceback__
    frames = traceback.extract_tb(tb)
    # The deepest frame that belongs to the experiment itself, so the report
    # points at the agent's line rather than somewhere inside torch.
    own = [f for f in frames if f.filename == code_path]
    frame = own[-1] if own else (frames[-1] if frames else None)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, tb)),
        "filename": frame.filename if frame else None,
        "lineno": frame.lineno if frame else None,
        "function": frame.name if frame else None,
        "source_line": (frame.line or "") if frame else "",
    }


def main(argv):
    global _CODE_PATH
    if len(argv) != 3:
        print("usage: harness.py <code_path> <artifact_dir>", file=sys.stderr)
        return 2

    _CODE_PATH = os.path.abspath(argv[1])
    artifact_dir = argv[2]
    os.makedirs(artifact_dir, exist_ok=True)
    results_path = os.path.join(artifact_dir, "results.json")

    with open(_CODE_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    # A fresh namespace containing only what we put in it. This is the check
    # that `exec_globals = globals()` made impossible upstream.
    namespace = {
        "__name__": "__main__",
        "__file__": _CODE_PATH,
        "__doc__": None,
        "__package__": None,
        "__loader__": None,
        "__spec__": None,
        "__builtins__": builtins,
        "record_result": _record_result,
        "record_metadata": _record_metadata,
    }
    initial_namespace = sorted(namespace)

    exception = None
    harness_error = None
    exit_code = 0
    try:
        code = compile(source, _CODE_PATH, "exec")
        exec(code, namespace)
    except SystemExit as exc:  # noqa: PERF203 - distinct handling per type
        # A forged clean exit. The static gate rejects these calls, but if one
        # reaches here it must not be mistaken for a successful run.
        exception = {
            "type": "SystemExit",
            "message": f"experiment called exit({exc.code!r})",
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "filename": None,
            "lineno": None,
            "function": None,
            "source_line": "",
        }
        exit_code = 1
    except BaseException as exc:  # noqa: BLE001 - the gate needs every failure
        exception = _exception_record(exc, _CODE_PATH)
        exit_code = 1

    payload = {
        "schema_version": SCHEMA_VERSION,
        "code_path": _CODE_PATH,
        "metrics": _METRICS,
        "metadata": _METADATA,
        "initial_namespace": initial_namespace,
        "environment": _environment(),
        "exception": exception,
    }
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=repr)
    except OSError as exc:
        harness_error = f"could not write results.json: {exc}"
        print(harness_error, file=sys.stderr)
        return 3

    if harness_error:
        return 3
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
