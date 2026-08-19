"""Process-isolated execution of experiment code.

Replaces the ``exec(code, globals())``-in-a-thread pattern that the gate exists
to fix. Three differences matter:

* **Fresh process, fresh namespace.** No state survives from a previous attempt,
  so "the variables are real" becomes a property of the runtime rather than a
  hope.
* **Untruncated capture, stdout and stderr separately, straight to files.** The
  1000-character ceiling is gone, and with it the coupling that let truncation
  hide a crash.
* **Timeout kills the process group.** ``ThreadPoolExecutor.result(timeout=)``
  returns to the caller but leaves the runaway thread burning CPU for the rest
  of the run; ``killpg`` actually stops it, including dataloader workers.

This is not an operating-system security sandbox: the child still runs as the
calling user and may have filesystem or network access.  Provider credentials
are nevertheless removed from its environment so ordinary agent code cannot
obtain the key that the parent scaffold needs for model calls.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .errors import HarnessError
from .schema import ExceptionRecord, ExecutionRecord, MetricRecord

HARNESS_PATH = str(Path(__file__).with_name("harness.py").resolve())

DEFAULT_TIMEOUT_S = 900

_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "AUTH",
    "COOKIE",
)
_SENSITIVE_ENV_NAMES = {
    "DATABASE_URL",
    "REDIS_URL",
    "MONGODB_URI",
    "SENTRY_DSN",
    "SSH_AUTH_SOCK",
    "GPG_AGENT_INFO",
    "KUBECONFIG",
    "NETRC",
}

_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4
_DUMPABLE_LOCK = threading.Lock()
_DUMPABLE_USERS = 0
_DUMPABLE_ORIGINAL: int | None = None


def code_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def make_run_id(code_hash: str, started_at: str, artifact_dir: str) -> str:
    """A short identifier for one execution.

    Derived from what makes the run unique rather than randomly generated, so a
    third party holding the artifacts can recompute it and confirm the values in
    the registry belong to this run and not another one.
    """
    material = f"{code_hash}|{started_at}|{artifact_dir}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def make_trace_id(run_id: str, key: str, lineno: int | None) -> str:
    """Binds one recorded value to one run and one call site."""
    return hashlib.sha256(f"{run_id}|{key}|{lineno}".encode("utf-8")).hexdigest()[:16]


def run_experiment(
    source: str,
    artifact_dir: str | os.PathLike[str],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    python: str | None = None,
) -> ExecutionRecord:
    """Execute ``source`` in a fresh interpreter and report what happened.

    Never raises on experiment failure — a crash is a result, and the gate is
    what decides what it means. ``HarnessError`` is reserved for the harness
    itself breaking.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    code_path = artifact_dir / "experiment.py"
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"
    results_path = artifact_dir / "results.json"
    code_path.write_text(source, encoding="utf-8")

    child_env = os.environ.copy()
    child_env.update(
        {
            "MPLBACKEND": "Agg",  # no display, no plotting crash
            "PYTHONUNBUFFERED": "1",  # partial output survives a kill
            "JOBLIB_VERBOSITY": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if env:
        child_env.update(env)
    child_env = _without_credentials(child_env)

    argv = [python or sys.executable, HARNESS_PATH, str(code_path), str(artifact_dir)]

    timed_out = False
    started_at = _utc_now()
    run_id = make_run_id(code_sha256(source), started_at, str(artifact_dir))
    started = time.monotonic()
    # Writing to files rather than PIPE: an experiment that outproduces the pipe
    # buffer would otherwise deadlock, which is exactly what a long run does.
    with _hide_parent_process_from_child():
        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            popen_kwargs: dict[str, object] = {
                "stdout": out,
                "stderr": err,
                "stdin": subprocess.DEVNULL,
                "cwd": cwd,
                "env": child_env,
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            try:
                proc = subprocess.Popen(argv, **popen_kwargs)  # type: ignore[arg-type]
            except OSError as exc:
                raise HarnessError(f"could not start execution harness: {exc}") from exc

            try:
                exit_code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_tree(proc)
                exit_code = proc.wait()

    duration = time.monotonic() - started

    record = ExecutionRecord(
        exit_code=exit_code,
        timed_out=timed_out,
        duration_s=duration,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        stdout_bytes=_size(stdout_path),
        stderr_bytes=_size(stderr_path),
        truncated=False,  # by construction: nothing here slices the output
        results_json_path=str(results_path) if results_path.exists() else None,
        run_id=run_id,
        argv=list(argv),
        started_at=started_at,
        finished_at=_utc_now(),
    )

    _load_results(results_path, record)
    for key, metric in record.metrics.items():
        metric.trace_id = make_trace_id(run_id, key, metric.lineno)
    return record


def _without_credentials(environment: dict[str, str]) -> dict[str, str]:
    """Return a child environment with likely credentials removed.

    Scrubbing happens after caller overrides, so ``env=`` cannot accidentally
    re-introduce a provider key.  This is defense in depth, not a replacement
    for a container or a separate unprivileged execution account.
    """
    clean: dict[str, str] = {}
    for name, value in environment.items():
        upper = name.upper()
        if upper in _SENSITIVE_ENV_NAMES:
            continue
        if any(marker in upper for marker in _SENSITIVE_ENV_MARKERS):
            continue
        clean[name] = value
    return clean


@contextlib.contextmanager
def _hide_parent_process_from_child():
    """Block Linux children from reading the parent's memory and `/proc` env.

    Environment scrubbing protects the child's own environment.  It does not,
    by itself, stop a same-UID child from opening ``/proc/<ppid>/environ`` or
    attaching with ptrace.  Linux's non-dumpable process flag closes those
    paths while the untrusted experiment is alive.  A reference count keeps
    the parent protected if two experiment threads overlap.

    Other operating systems retain environment scrubbing but need a real
    sandbox or separate account for the equivalent parent-process boundary.
    """
    global _DUMPABLE_USERS, _DUMPABLE_ORIGINAL
    joined = False
    if sys.platform.startswith("linux"):
        with _DUMPABLE_LOCK:
            if _DUMPABLE_USERS == 0:
                original = _prctl(_PR_GET_DUMPABLE)
                if original >= 0 and _prctl(_PR_SET_DUMPABLE, 0) == 0:
                    _DUMPABLE_ORIGINAL = original
            if _DUMPABLE_ORIGINAL is not None:
                _DUMPABLE_USERS += 1
                joined = True
    try:
        yield
    finally:
        if joined:
            with _DUMPABLE_LOCK:
                _DUMPABLE_USERS -= 1
                if _DUMPABLE_USERS == 0:
                    original = _DUMPABLE_ORIGINAL
                    _DUMPABLE_ORIGINAL = None
                    if original is not None:
                        _prctl(_PR_SET_DUMPABLE, original)


def _prctl(option: int, argument: int = 0) -> int:
    """Call Linux prctl without adding a third-party dependency."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return int(libc.prctl(option, argument, 0, 0, 0))
    except (AttributeError, OSError):
        return -1


def _load_results(results_path: Path, record: ExecutionRecord) -> None:
    """Fold the harness's JSON into the execution record.

    A missing or unparseable file is not an error here — ``results.contract_present``
    is the check that has an opinion about it.
    """
    if not results_path.exists():
        return
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        record.harness_error = f"results.json is unreadable: {exc}"
        return

    for key, raw in (payload.get("metrics") or {}).items():
        record.metrics[key] = MetricRecord(
            key=raw.get("key", key),
            value=raw.get("value"),
            unit=raw.get("unit"),
            lineno=raw.get("lineno"),
            source_line=raw.get("source_line", ""),
            call_count=raw.get("call_count", 1),
            observations=list(raw.get("observations") or []),
            observations_truncated=bool(raw.get("observations_truncated")),
        )

    record.code_sha256 = payload.get("code_sha256")
    record.metadata = payload.get("metadata") or {}
    record.initial_namespace = payload.get("initial_namespace") or []
    record.environment = payload.get("environment") or {}

    exc_payload = payload.get("exception")
    if exc_payload:
        record.exception = ExceptionRecord(
            type=exc_payload.get("type", "Exception"),
            message=exc_payload.get("message", ""),
            traceback=exc_payload.get("traceback", ""),
            filename=exc_payload.get("filename"),
            lineno=exc_payload.get("lineno"),
            function=exc_payload.get("function"),
            source_line=exc_payload.get("source_line", ""),
        )


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child and everything it spawned."""
    if os.name != "posix":
        proc.kill()
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
