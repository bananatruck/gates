"""Turn an ablation run's JSON into the tables the report and deck show.

Everything here reads `ablation.json` produced by `tools_ablation.py`. No value
is written by hand — if a number appears in the report, this module took it out
of a run record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


# --------------------------------------------------------------------------- #
# what each arm produced, and whether it survives re-execution
# --------------------------------------------------------------------------- #


def reported_numbers_table(run: dict) -> str:
    """The numbers each arm would hand to the writing agent."""
    g = run["arms"]["gated"]["verification"]
    u = run["arms"]["ungated"]["verification"]
    keys = sorted(set(g["reported"]) | set(u["reported"]))

    if not keys:
        return ("<p class='small'>Neither arm recorded a value through the "
                "results contract.</p>")

    rows = []
    for k in keys:
        gv = g["reported"].get(k)
        gr = g["reproduced"].get(k)
        uv = u["reported"].get(k)
        status = ("<span class='ok'>reproduced</span>" if k in g["matched"]
                  else "<span class='bad'>did not reproduce</span>"
                  if k in g["mismatched"] else "<span class='small'>—</span>")
        rows.append(
            f"<tr><td class='mono'>{k}</td>"
            f"<td>{_fmt(gv) if gv is not None else '—'}</td>"
            f"<td>{_fmt(gr) if gr is not None else '—'}</td>"
            f"<td>{status}</td>"
            f"<td>{_fmt(uv) if uv is not None else '<span class=small>not recorded</span>'}</td>"
            f"</tr>"
        )
    return (
        "<table><tr>"
        "<th style='width:26%'>metric</th>"
        "<th>with Gate 1<br><span class='small'>reported</span></th>"
        "<th>re-executed<br><span class='small'>same code, fresh process</span></th>"
        "<th>verdict</th>"
        "<th>without Gate 1<br><span class='small'>reported</span></th>"
        "</tr>" + "".join(rows) + "</table>"
    )


def accuracy_summary(run: dict) -> str:
    """The headline: how many of each arm's numbers can be checked at all."""
    out = []
    for arm in ("gated", "ungated"):
        a = run["arms"][arm]
        v = a["verification"]
        label = "with Gate 1" if arm == "gated" else "without Gate 1"
        reported = len(v["reported"])
        matched, mismatched = len(v["matched"]), len(v["mismatched"])
        checkable = matched + mismatched
        rate = f"{100 * v['rate']:.0f}%" if v["rate"] is not None else "—"
        out.append(
            f"<tr><td><b>{label}</b></td>"
            f"<td>{'yes' if a['accepted'] else '<span class=bad>no</span>'}</td>"
            f"<td>{a['turns']}</td>"
            f"<td>{reported}</td>"
            f"<td>{checkable}</td>"
            f"<td>{matched}</td>"
            f"<td>{rate}</td></tr>"
        )
    note = ""
    u = run["arms"]["ungated"]["verification"]
    if not u["reported"]:
        note = (f"<p class='small'>{u['reason'].capitalize()}.</p>")
    return (
        "<table><tr><th style='width:20%'>arm</th><th>accepted a run</th>"
        "<th>turns used</th><th>numbers reported</th><th>numbers checkable</th>"
        "<th>reproduced</th><th>reproduction rate</th></tr>"
        + "".join(out) + "</table>" + note
    )


# --------------------------------------------------------------------------- #
# every check that ran
# --------------------------------------------------------------------------- #


def checks_fired_table(run: dict, arm: str = "gated") -> str:
    """Per attempt, every check and its outcome — what the gate actually did."""
    attempts = run["arms"][arm]["attempts"]
    if not attempts or not attempts[0].get("all_checks"):
        return "<p class='small'>No per-check record in this run.</p>"

    ids: list[str] = []
    for a in attempts:
        for c in a["all_checks"]:
            if c["id"] not in ids:
                ids.append(c["id"])

    head = "".join(f"<th>turn {a['turn']}</th>" for a in attempts)
    rows = []
    for cid in ids:
        cells = []
        sev = ""
        for a in attempts:
            match = next((c for c in a["all_checks"] if c["id"] == cid), None)
            if match is None:
                cells.append("<td class='small'>not run</td>")
                continue
            sev = match["severity"]
            if match["passed"]:
                cells.append("<td class='ok'>pass</td>")
            elif sev == "FAIL":
                cells.append("<td class='bad'>FAIL</td>")
            else:
                cells.append(f"<td class='warn'>{sev}</td>")
        rows.append(
            f"<tr><td class='mono'>{cid}</td><td>{sev}</td>{''.join(cells)}</tr>"
        )
    return (
        f"<table><tr><th style='width:30%'>check</th><th>severity</th>{head}</tr>"
        + "".join(rows) + "</table>"
    )


def attempts_table(run: dict) -> str:
    """Turn by turn, both arms: what was decided and on what evidence."""
    rows = []
    for arm in ("gated", "ungated"):
        a = run["arms"][arm]
        label = "with Gate 1" if arm == "gated" else "without Gate 1"
        for att in a["attempts"]:
            verdict = ("<span class='ok'>accepted</span>" if att["accepted"]
                       else "<span class='bad'>rejected</span>")
            other = att.get("counterpart_would_accept")
            other_s = ("—" if other is None
                       else "would accept" if other else "would reject")
            failed = ", ".join(att["failed_checks"]) or "—"
            rows.append(
                f"<tr><td>{label}</td><td>{att['turn']}</td><td>{verdict}</td>"
                f"<td class='small'>{other_s}</td>"
                f"<td>{att['exit_code'] if att['exit_code'] is not None else 'not run'}</td>"
                f"<td>{att['stdout_bytes']:,}</td>"
                f"<td>{len(att['recorded'])}</td>"
                f"<td class='mono small'>{failed}</td></tr>"
            )
    return (
        "<table><tr><th>arm</th><th>turn</th><th>verdict</th>"
        "<th>other arm</th><th>exit</th><th>stdout bytes</th>"
        "<th>values recorded</th><th>failing checks</th></tr>"
        + "".join(rows) + "</table>"
    )


# --------------------------------------------------------------------------- #
# logs
# --------------------------------------------------------------------------- #


def log_locations(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    return f"""<table>
<tr><th style="width:22%">arm</th><th>path</th><th>files per attempt</th></tr>
<tr><td><b>with Gate 1</b></td>
    <td class="mono">{run_dir}/gated/gate1/attempt_NN/</td>
    <td class="mono small">experiment.py · stdout.txt · stderr.txt ·
        results.json · registry.json · gate1_report.json</td></tr>
<tr><td><b>without Gate 1</b></td>
    <td class="mono">{run_dir}/ungated/attempt_NN/</td>
    <td class="mono small">experiment.py · stdout.txt · stderr.txt</td></tr>
<tr><td>shadow verdicts</td>
    <td class="mono">{run_dir}/ungated/shadow_gate/gate1/attempt_NN/</td>
    <td class="mono small">what Gate 1 would have said about the ungated arm's
        executions</td></tr>
<tr><td>re-execution</td>
    <td class="mono">{run_dir}/&lt;arm&gt;/verify/</td>
    <td class="mono small">the accepted code, run again</td></tr>
<tr><td>summary</td>
    <td class="mono">{run_dir}/ablation.txt · ablation.json</td>
    <td class="mono small">the tables in this report</td></tr>
</table>"""


def log_snippet(path: str | Path, *, head: int = 0, tail: int = 0,
                grep: str | None = None, limit: int = 14) -> str:
    """A real excerpt from a real file, or an honest note that it is absent."""
    p = Path(path)
    if not p.exists():
        return f"<p class='small'>[not present: {p}]</p>"
    lines = p.read_text(errors="replace").splitlines()
    if grep:
        picked = [ln for ln in lines if grep.lower() in ln.lower()][:limit]
    elif head:
        picked = lines[:head]
    elif tail:
        picked = lines[-tail:]
    else:
        picked = lines[:limit]
    body = "\n".join(ln[:150] for ln in picked)
    total = len(lines)
    return (f"<pre>{_escape(body)}</pre>"
            f"<p class='small'>{p} — {total:,} lines total</p>")


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------- #
# our run, scored against MLR-Bench's published taxonomy
# --------------------------------------------------------------------------- #


def taxonomy_table(run: dict) -> str:
    """Score both arms against MLR-Bench's four hallucination types.

    This is *our* run scored with *their* categories. It is not a run of
    MLR-Bench — that needs their harness and their task set, and this report
    does not claim otherwise. What it does claim is that the first category is
    measurable on any run, and here it is measured.
    """
    g, u = run["arms"]["gated"], run["arms"]["ungated"]
    gv, uv = g["verification"], u["verification"]

    def faked(arm, ver):
        """Numbers the arm would hand the writer that no execution backs."""
        if not arm["accepted"]:
            return "n/a — nothing reached the writer"
        unbacked = len(ver["reported"]) - (len(ver["matched"]) + len(ver["mismatched"]))
        if not ver["reported"]:
            return ("<span class='bad'>every number it reports is unbacked</span> — "
                    "it recorded none through any contract, so the writer receives "
                    "prose to copy from")
        return (f"{unbacked} of {len(ver['reported'])} unbacked; "
                f"{len(ver['matched'])} reproduced on re-execution")

    rows = [
        ("Faked experimental results",
         "Data, metrics or outcomes fabricated or never performed",
         faked(g, gv), faked(u, uv),
         "<b>measured</b> — this is the class Gate 1 addresses"),
        ("Hallucinated methodology",
         "Techniques claimed but not implemented",
         "not measured here", "not measured here",
         "Gate 2 — needs the plan compared to the code"),
        ("Incorrect citations",
         "References wrong or nonexistent",
         "not measured here", "not measured here",
         "Gate 3 — the manuscript is out of Gate 1's scope"),
        ("Mathematical errors",
         "Wrong equations or derivations",
         "not measured here", "not measured here",
         "outside all three gates as specified"),
    ]
    body = "".join(
        f"<tr><td><b>{name}</b><br><span class='small'>{desc}</span></td>"
        f"<td>{withg}</td><td>{without}</td><td class='small'>{who}</td></tr>"
        for name, desc, withg, without, who in rows
    )
    return (
        "<table><tr><th style='width:24%'>MLR-Bench class</th>"
        "<th style='width:24%'>with Gate 1</th>"
        "<th style='width:24%'>without Gate 1</th>"
        "<th>whose job</th></tr>" + body + "</table>"
    )


def feedback_example(run_dir: str | Path, run: dict) -> str:
    """The actual feedback the engineer received on its first rejection.

    The point of the LLM layer, shown rather than described: a deterministic
    finding, and a fix written against that run's own code.
    """
    run_dir = Path(run_dir)
    for att in run["arms"]["gated"]["attempts"]:
        if att["accepted"]:
            continue
        report_path = (run_dir / "gated" / "gate1"
                       / f"attempt_{att['turn']:02d}" / "gate1_report.json")
        if not report_path.exists():
            continue
        data = json.loads(report_path.read_text())
        exc = (data.get("execution") or {}).get("exception") or {}
        fixes = data.get("generated_fixes")
        if not fixes:
            continue
        diagnosis = ""
        if exc:
            diagnosis = (
                f"<b>{exc.get('type')}: {exc.get('message')}</b><br>"
                f"<span class='mono small'>line {exc.get('lineno')} in "
                f"{exc.get('function') or '&lt;module&gt;'} &nbsp;|&nbsp; "
                f"{_escape((exc.get('source_line') or '').strip())}</span>"
            )
        return (
            f"<p>Turn {att['turn']}, rejected on "
            f"<code>{', '.join(att['failed_checks'])}</code>.</p>"
            f"<div class='callout'>{diagnosis}</div>"
            f"<p class='small'>The deterministic finding above. Below, what the "
            f"LLM layer wrote for the engineer from it — every name in it "
            f"checked against the report before it was shown:</p>"
            f"<pre>{_escape(fixes)}</pre>"
        )
    return "<p class='small'>No rejection with generated fixes in this run.</p>"


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #


def usage_table(run: dict) -> str:
    """Tokens by role, and where the money actually goes.

    The engineer is a paid reasoning model; the gate runs locally. Reasoning
    tokens are billed and invisible in the output, which is why they are broken
    out rather than folded into completion.
    """
    usage = run.get("usage") or {}
    if not usage:
        return "<p class='small'>No token accounting in this run record.</p>"

    paid_roles = {"engineer"}
    rows, totals = [], {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0}
    for role, u in sorted(usage.items()):
        for k in totals:
            totals[k] += u.get(k, 0)
        billed = "paid API" if role in paid_roles else "local — no marginal cost"
        rows.append(
            f"<tr><td><b>{role}</b></td><td>{u['calls']}</td>"
            f"<td>{u['prompt']:,}</td><td>{u['completion']:,}</td>"
            f"<td>{u['reasoning']:,}</td>"
            f"<td class='small'>{billed}</td></tr>"
        )
    per_call = (totals["completion"] / totals["calls"]) if totals["calls"] else 0
    return (
        "<table><tr><th style='width:18%'>role</th><th>calls</th>"
        "<th>prompt tokens</th><th>completion tokens</th>"
        "<th>of which reasoning</th><th>billed</th></tr>"
        + "".join(rows)
        + f"<tr><td><b>total</b></td><td>{totals['calls']}</td>"
        f"<td>{totals['prompt']:,}</td><td>{totals['completion']:,}</td>"
        f"<td>{totals['reasoning']:,}</td>"
        f"<td class='small'>{per_call:,.0f} completion tokens per call</td></tr>"
        "</table>"
    )


def cost_note(run: dict) -> str:
    """What Gate 1 adds to the bill, and what it saves."""
    usage = run.get("usage") or {}
    eng = usage.get("engineer", {})
    gate = usage.get("gate", {})
    if not eng:
        return ""
    eng_per = eng["completion"] / eng["calls"] if eng["calls"] else 0
    gate_per = gate.get("completion", 0) / gate["calls"] if gate.get("calls") else 0
    ratio = (eng_per / gate_per) if gate_per else None

    # Executions the static tier rejected before they ran.
    saved = sum(
        1
        for arm in ("gated",)
        for a in run["arms"][arm]["attempts"]
        if a["exit_code"] is None
    )
    lines = [
        f"<p><b>Gate 1's marginal cost on the paid API is zero.</b> Its two jobs "
        f"ran locally: {gate.get('calls', 0)} calls, "
        f"{gate.get('completion', 0):,} completion tokens, nothing billed.</p>"
    ]
    if ratio:
        lines.append(
            f"<p>For scale, one engineer call costs {eng_per:,.0f} completion "
            f"tokens against {gate_per:,.0f} for a gate call — "
            f"<b>{ratio:.0f}×</b>. Almost all of the engineer's is reasoning, "
            f"which is billed and never appears in the output.</p>"
        )
    lines.append(
        f"<p>What Gate 1 <i>does</i> change on the bill is rewrites: each "
        f"rejection buys another engineer call. In this run "
        f"<b>{saved}</b> attempt(s) were rejected by the static tier before "
        f"execution, costing no compute at all, and every rejection that names "
        f"the real fault is a rewrite the agent does not waste — which is the "
        f"whole reason the shadowed-contract check was added.</p>"
    )
    return "".join(lines)
