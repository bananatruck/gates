"""Exceptions raised by the G.A.T.E.S. validity layer."""


class GateError(Exception):
    """Base class for all gate errors."""


class GateFailure(GateError):
    """A gate rejected every attempt and its retry budget is exhausted.

    Raised only when there is no earlier passing artifact to fall back on. The
    host scaffold is expected to let this propagate and exit non-zero: a run
    that never produced a valid experiment must not produce a paper.
    """

    def __init__(self, gate: str, attempts: int, report=None):
        self.gate = gate
        self.attempts = attempts
        self.report = report
        detail = ""
        if report is not None:
            failed = ", ".join(c.id for c in report.failed_checks())
            if failed:
                detail = f" Last failure: {failed}."
        super().__init__(
            f"{gate} rejected all {attempts} attempt(s) and no earlier attempt "
            f"passed.{detail}"
        )


class HarnessError(GateError):
    """The execution harness itself failed, independently of the agent's code."""
