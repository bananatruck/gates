"""Build the Gate 1 presentation (PPTX).

Same numbers as GATE1_REPORT.pdf, same palette, one claim per slide.

    python reports/make_deck.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
OUT = HERE / "GATE1_DECK.pptx"

RUN_JSON = Path(
    "/home/kesh/AgentLaboratory-Gemini/ablation_runs/data_efficiency/ablation.json"
)
RUN_DIR = RUN_JSON.parent

BENCH = Path(
    "/tmp/claude-1000/-home-kesh/976cef29-0afd-479c-a716-6c557e07b6cb"
    "/scratchpad/bench2.json"
)

BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x8A, 0x88, 0x80)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
PANEL = RGBColor(0xF4, 0xF3, 0xEE)

W, H = Inches(13.333), Inches(7.5)


def deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = SURFACE
    return s


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, spacing=1.15):
    """runs: list of (text, size, bold, colour) -> one paragraph each."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, (t, size, bold, colour) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = t
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = colour
        r.font.name = "DejaVu Sans"
    return box


def panel(slide, x, y, w, h, colour=PANEL):
    from pptx.enum.shapes import MSO_SHAPE

    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = colour
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def header(slide, kicker, title):
    text(slide, Inches(0.6), Inches(0.35), Inches(12), Inches(0.4),
         [(kicker.upper(), 11, True, BLUE)])
    text(slide, Inches(0.6), Inches(0.72), Inches(12), Inches(0.7),
         [(title, 26, True, INK)])


def bullets(slide, x, y, w, items, size=14, gap=1.5):
    runs = []
    for it in items:
        if isinstance(it, tuple):
            runs.append(it)
        else:
            runs.append((f"·  {it}", size, False, INK2))
    text(slide, x, y, w, Inches(4), runs, spacing=gap)


def stat(slide, x, y, w, value, label, colour=BLUE):
    panel(slide, x, y, w, Inches(1.35))
    text(slide, x + Inches(0.2), y + Inches(0.12), w - Inches(0.4), Inches(0.6),
         [(value, 30, True, colour)])
    text(slide, x + Inches(0.2), y + Inches(0.82), w - Inches(0.4), Inches(0.45),
         [(label, 10, False, INK2)])


def picture(slide, name, x, y, w, max_h=None):
    """Place a figure, shrinking it if it would not fit the space given.

    matplotlib saves with bbox_inches="tight", so a figure's on-disk aspect is
    not the figsize it was declared with -- placing by width alone overflowed
    the slide edge and clipped the axis labels off the scanner chart. The real
    aspect is read from the file.
    """
    p = ASSETS / name
    if not p.exists():
        return
    from PIL import Image

    pw, ph = Image.open(p).size
    height = Emu(int(w * ph / pw))
    if max_h is not None and height > max_h:
        height = max_h
        w = Emu(int(max_h * pw / ph))
    slide.shapes.add_picture(str(p), x, y, width=w, height=height)




def _table(slide, x, y, w, rows, col_w, header=True, size=11, rh=Inches(0.34)):
    """A real table, because a claim about numbers should show the numbers."""
    from pptx.util import Inches as I

    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, x, y, w, rh * n_rows)
    tbl = shape.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = cw
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(cell_text)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            para.font.name = "DejaVu Sans"
            para.font.bold = header and i == 0
            para.font.color.rgb = INK if (header and i == 0) else INK2
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if (header and i == 0) else SURFACE
    return shape


def _slide_run(prs, run):
    s = blank(prs)
    header(s, "the run", "Data efficiency, with and without Gate 1")
    text(s, Inches(0.6), Inches(1.55), Inches(12.1), Inches(0.9),
         [("Question given to the agent: how much labelled data does a classifier "
           "actually need? Report test accuracy at 5/10/25/50/100% of the training "
           "pool, the smallest fraction reaching 95% of full-data accuracy, and the "
           "training wallclock — seven values.", 13, False, INK2),
          ("Engineer and gate both on a local qwen3:8b. The arms differ only in what "
           "decides that an experiment succeeded.", 11, False, MUTED)])

    g, u = run["arms"]["gated"], run["arms"]["ungated"]
    gv, uv = g["verification"], u["verification"]
    rows = [["", "with Gate 1", "without Gate 1"],
            ["accepted a run", "yes" if g["accepted"] else "no",
             "yes" if u["accepted"] else "no"],
            ["turns used", g["turns"], u["turns"]],
            ["numbers reported", len(gv["reported"]), len(uv["reported"])],
            ["numbers checkable against an execution",
             len(gv["matched"]) + len(gv["mismatched"]),
             len(uv["matched"]) + len(uv["mismatched"])],
            ["reproduced on re-execution", len(gv["matched"]), len(uv["matched"])]]
    _table(s, Inches(0.6), Inches(3.0), Inches(12.1), rows,
           [Inches(6.1), Inches(3.0), Inches(3.0)], size=13, rh=Inches(0.46))

    rate = f"{100 * gv['rate']:.0f}%" if gv["rate"] is not None else "n/a"
    stat(s, Inches(0.6), Inches(6.0), Inches(4.0), rate,
         "of the Gate 1 arm's numbers reproduced when its code was re-run", BLUE)
    stat(s, Inches(4.9), Inches(6.0), Inches(4.0), str(len(uv["reported"])),
         "numbers the ungated arm recorded through any contract", ORANGE)
    stat(s, Inches(9.2), Inches(6.0), Inches(3.5),
         str(len(gv["matched"]) + len(gv["mismatched"])),
         "of the gated arm's numbers are checkable at all", BLUE)


def _slide_accuracy(prs, run):
    s = blank(prs)
    header(s, "the two reports", "What each arm would hand the writing agent")
    gv = run["arms"]["gated"]["verification"]
    uv = run["arms"]["ungated"]["verification"]
    keys = sorted(set(gv["reported"]) | set(uv["reported"]))[:9]
    rows = [["metric", "Gate 1: reported", "re-executed", "verdict",
             "no gate: reported"]]
    for k in keys:
        rep = gv["reported"].get(k)
        again = gv["reproduced"].get(k)
        verdict = ("reproduced" if k in gv["matched"]
                   else "MISMATCH" if k in gv["mismatched"] else "—")
        rows.append([
            k,
            f"{rep:.4g}" if isinstance(rep, float) else (rep if rep is not None else "—"),
            f"{again:.4g}" if isinstance(again, float) else (again if again is not None else "—"),
            verdict,
            uv["reported"].get(k, "not recorded"),
        ])
    if len(rows) == 1:
        rows.append(["(neither arm recorded a value)", "", "", "", ""])
    _table(s, Inches(0.6), Inches(1.7), Inches(12.1), rows,
           [Inches(3.4), Inches(2.2), Inches(2.2), Inches(2.1), Inches(2.2)],
           size=11, rh=Inches(0.36))
    text(s, Inches(0.6), Inches(6.4), Inches(12.1), Inches(0.9),
         [(uv["reason"].capitalize() + "." if uv["reason"] else
           "Both arms recorded values.", 13, True, ORANGE),
          ("Accuracy here means one thing only: run the accepted code again and "
           "see whether the numbers it reported are the numbers it produces.",
           11, False, INK2)])


def _slide_checks_fired(prs, run):
    s = blank(prs)
    header(s, "checks", "Every check that ran, turn by turn")
    attempts = run["arms"]["gated"]["attempts"]
    ids = []
    for a in attempts:
        for c in a.get("all_checks", []):
            if c["id"] not in ids:
                ids.append(c["id"])
    ids = ids[:14]
    rows = [["check", "sev"] + [f"turn {a['turn']}" for a in attempts]]
    for cid in ids:
        row, sev = [cid], ""
        for a in attempts:
            m = next((c for c in a.get("all_checks", []) if c["id"] == cid), None)
            if m is None:
                row.append("—")
            else:
                sev = m["severity"]
                row.append("pass" if m["passed"] else sev)
        rows.append([row[0], sev] + row[1:])
    if len(rows) == 1:
        rows.append(["(no per-check record)", "", ""])
    ncols = len(rows[0])
    _table(s, Inches(0.6), Inches(1.7), Inches(12.1), rows,
           [Inches(4.6), Inches(1.1)] + [Inches(6.4 / max(ncols - 2, 1))] * (ncols - 2),
           size=10, rh=Inches(0.33))


def _slide_taxonomy(prs, run):
    """Our run, scored with MLR-Bench's published categories."""
    s = blank(prs)
    header(s, "taxonomy", "This run, scored against MLR-Bench's four classes")
    g, u = run["arms"]["gated"], run["arms"]["ungated"]
    gv, uv = g["verification"], u["verification"]

    def cell(arm, ver):
        if not arm["accepted"]:
            return "nothing reached the writer"
        if not ver["reported"]:
            return "every number unbacked — nothing recorded"
        unbacked = len(ver["reported"]) - (len(ver["matched"]) + len(ver["mismatched"]))
        return f"{unbacked} of {len(ver['reported'])} unbacked, {len(ver['matched'])} reproduced"

    rows = [["MLR-Bench class", "with Gate 1", "without Gate 1", "whose job"],
            ["Faked experimental results", cell(g, gv), cell(u, uv), "Gate 1 — measured here"],
            ["Hallucinated methodology", "not measured", "not measured", "Gate 2"],
            ["Incorrect citations", "not measured", "not measured", "Gate 3"],
            ["Mathematical errors", "not measured", "not measured", "outside all three"]]
    _table(s, Inches(0.6), Inches(1.8), Inches(12.1), rows,
           [Inches(3.4), Inches(3.3), Inches(3.3), Inches(2.1)], size=12,
           rh=Inches(0.62))
    text(s, Inches(0.6), Inches(5.6), Inches(12.1), Inches(1.2),
         [("This is our run scored with their categories — not a run of MLR-Bench, "
           "which needs their harness and task set.", 13, True, ORANGE),
          ("For scale, MLR-Bench reports faked results and hallucinated methodology "
           "in more than half of 10 tasks, and almost all AI Scientist V2 papers "
           "contain both. BadScientist reports fabricated papers accepted at up to "
           "82.0% with detection barely above chance.", 11, False, INK2)])


def _slide_cost(prs, run):
    """Where the money goes, and what the gate adds to it."""
    s = blank(prs)
    header(s, "cost", "Gate 1's marginal cost on the paid API is zero")
    usage = run.get("usage") or {}
    rows = [["role", "calls", "prompt tok", "completion tok", "of which reasoning", "billed"]]
    for role, u in sorted(usage.items()):
        rows.append([role, u["calls"], f"{u['prompt']:,}", f"{u['completion']:,}",
                     f"{u['reasoning']:,}",
                     "paid API" if role == "engineer" else "local — free"])
    if len(rows) == 1:
        rows.append(["(no accounting)", "", "", "", "", ""])
    _table(s, Inches(0.6), Inches(1.8), Inches(12.1), rows,
           [Inches(2.0), Inches(1.3), Inches(2.2), Inches(2.4), Inches(2.4), Inches(1.8)],
           size=12, rh=Inches(0.5))
    eng, gate = usage.get("engineer", {}), usage.get("gate", {})
    if eng.get("calls") and gate.get("calls"):
        ep = eng["completion"] / eng["calls"]
        gp = gate["completion"] / gate["calls"]
        stat(s, Inches(0.6), Inches(4.6), Inches(3.8), f"{ep/gp:.0f}x",
             "one engineer call vs one gate call, in completion tokens", ORANGE)
        stat(s, Inches(4.7), Inches(4.6), Inches(3.8), f"{gate['completion']:,}",
             "gate tokens — all local, none billed", BLUE)
        stat(s, Inches(8.8), Inches(4.6), Inches(3.9), f"{eng['reasoning']:,}",
             "engineer reasoning tokens: billed, never seen in the output", ORANGE)
    text(s, Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.9),
         [("What Gate 1 changes on the bill is rewrites: each rejection buys "
           "another engineer call. A rejection that names the real fault is a "
           "rewrite the agent does not waste — which is why the shadowed-contract "
           "check was worth adding.", 13, False, INK2)])


def _slide_logs(prs, run):
    s = blank(prs)
    header(s, "logs", "Where the evidence is")
    d = RUN_DIR
    rows = [["arm", "path", "files per attempt"],
            ["with Gate 1", f"{d}/gated/gate1/attempt_NN/",
             "experiment.py, stdout.txt, stderr.txt, results.json, registry.json, gate1_report.json"],
            ["without Gate 1", f"{d}/ungated/attempt_NN/",
             "experiment.py, stdout.txt, stderr.txt"],
            ["shadow verdicts", f"{d}/ungated/shadow_gate/gate1/attempt_NN/",
             "what Gate 1 would have said about the same executions"],
            ["re-execution", f"{d}/<arm>/verify/", "the accepted code, run again"],
            ["summary", f"{d}/ablation.txt · ablation.json", "the tables in the report"]]
    _table(s, Inches(0.6), Inches(1.7), Inches(12.1), rows,
           [Inches(2.2), Inches(5.2), Inches(4.7)], size=10, rh=Inches(0.52))
    # a real excerpt
    acc = run["arms"]["gated"].get("accepted_at")
    snippet = ""
    if acc:
        f = RUN_DIR / "gated" / "gate1" / f"attempt_{acc:02d}" / "stdout.txt"
        if f.exists():
            snippet = "\n".join(f.read_text(errors="replace").splitlines()[:6])
    if snippet:
        panel(s, Inches(0.6), Inches(4.9), Inches(12.1), Inches(2.0))
        text(s, Inches(0.85), Inches(5.05), Inches(11.6), Inches(1.8),
             [("stdout.txt from the accepted run, captured in full and never truncated",
               11, True, INK)] +
             [(ln[:120], 10, False, INK2) for ln in snippet.splitlines()])


# --------------------------------------------------------------------------- #


def build(bench, run):
    prs = deck()

    # 1 — title
    s = blank(prs)
    text(s, Inches(0.9), Inches(2.3), Inches(11.5), Inches(1.2),
         [("G.A.T.E.S. — Gate 1", 46, True, INK)])
    text(s, Inches(0.9), Inches(3.5), Inches(11.5), Inches(1.0),
         [("Execution validity for autonomous research agents", 20, False, INK2)])
    text(s, Inches(0.9), Inches(4.15), Inches(11.5), Inches(1.0),
         [("Checks · metrics · runs · measured impact on Agent Laboratory",
           15, False, MUTED)])
    text(s, Inches(0.9), Inches(6.3), Inches(11.5), Inches(0.5),
         [(f"{date.today().isoformat()}   ·   247 gate tests, 63 integration tests"
           "   ·   every figure a recorded measurement or a cited number",
           11, False, MUTED)])

    # 2 — the run and what each arm produced
    if run:
        _slide_run(prs, run)
        _slide_accuracy(prs, run)
        _slide_checks_fired(prs, run)
        _slide_logs(prs, run)
        _slide_taxonomy(prs, run)
        _slide_cost(prs, run)

    # 3 — the defect
    s = blank(prs)
    header(s, "the problem", "One line produced both the silent failure and the fabrication")
    panel(s, Inches(0.6), Inches(1.7), Inches(12.1), Inches(1.5))
    text(s, Inches(0.9), Inches(1.85), Inches(11.5), Inches(1.2),
         [("return output_capture.getvalue()[:MAX_LEN]      # MAX_LEN = 1000",
           15, True, INK),
          ("the crash marker is appended AFTER the program's own output — so it "
           "falls off the end of the same slice", 12, False, INK2)])
    bullets(s, Inches(0.6), Inches(3.5), Inches(12),
            ["The writing agent received 1,000 characters as the entire experimental record",
             "The only crash test was a substring search for a marker that was no longer there",
             "Execution ran in a never-cleared namespace, in a thread the timeout could not kill"])
    stat(s, Inches(0.6), Inches(5.5), Inches(3.0), "1.0",
         "reward score on a run raising NameError every attempt", ORANGE)
    stat(s, Inches(3.9), Inches(5.5), Inches(3.0), "81.60%",
         "test accuracy reported — never measured", ORANGE)
    stat(s, Inches(7.2), Inches(5.5), Inches(3.0), "13.61×",
         "speedup reported — never measured", ORANGE)
    stat(s, Inches(10.5), Inches(5.5), Inches(2.2), "n=1",
         "archived runs usable of 7", MUTED)

    # 3 — checks
    s = blank(prs)
    header(s, "what gate 1 checks", "20 deterministic checks in six families")
    picture(s, "checks.png", Inches(0.6), Inches(1.65), Inches(7.6), max_h=Inches(4.4))
    panel(s, Inches(8.5), Inches(1.65), Inches(4.2), Inches(4.6))
    text(s, Inches(8.8), Inches(1.85), Inches(3.7), Inches(4.2),
         [("The severity split is the design", 14, True, INK),
          ("FAIL blocks the run. WARN and INFO are reported, propagate into the "
           "evidence bundle, and can never change a verdict.", 11, False, INK2),
          ("", 8, False, INK2),
          ("That is what lets a REQUIRED LLM layer coexist with a model-free "
           "verdict: model findings are WARN by construction, and the feedback "
           "report is written after decide() has already fixed the outcome.",
           11, False, INK2),
          ("", 8, False, INK2),
          ("A test parses the module's AST to assert Severity.FAIL never appears "
           "in an expression there.", 10, False, MUTED)])
    text(s, Inches(0.6), Inches(6.5), Inches(12), Inches(0.5),
         [("results.values_computed is the strongest claim: record_result(\"k\", acc) "
           "passes, record_result(\"k\", 0.816) fails — and round(0.8160, 3) fails too, "
           "because constant folding does not launder a typed number.",
           11, False, INK2)])

    # 4 — logs
    s = blank(prs)
    header(s, "logs", "Collapse the log, then read what the patterns cannot reach")
    picture(s, "compression.png", Inches(0.6), Inches(1.5), Inches(6.6),
            max_h=Inches(2.2))
    picture(s, "scanner.png", Inches(0.6), Inches(3.9), Inches(6.6),
            max_h=Inches(3.2))
    panel(s, Inches(7.5), Inches(1.5), Inches(5.2), Inches(5.4))
    prec = rec = None
    if bench and "qwen_fewshot_3" in bench:
        prec = bench["qwen_fewshot_3"]["precision"]
        rec = bench["qwen_fewshot_3"]["recall"]
    lines = [
        ("Lossless in distinct content", 14, True, INK),
        ("203 non-blank lines → 4 distinct shapes. Every shape survives with its "
         "first real line number, so nothing a scanner could flag disappears — "
         "recall is not traded for the saving.", 11, False, INK2),
        ("", 8, False, INK2),
        ("nan and inf are not digits, so 'loss nan' survives as its own shape "
         "rather than being absorbed into the epoch group.", 11, False, INK2),
        ("", 8, False, INK2),
        ("Precision is the floor", 14, True, INK),
        ("These findings are WARN and cannot force a rewrite. What a false "
         "positive does is put a non-issue into the section the writer is told "
         "it must disclose — so the paper reports a problem that never happened. "
         "The corpus includes TensorFlow's cuFFT notice and oneDNN's banner: "
         "E-level, CUDA-adjacent, and on every healthy run.", 11, False, INK2),
    ]
    text(s, Inches(7.8), Inches(1.7), Inches(4.6), Inches(5.0), lines)

    # 5 — runs
    s = blank(prs)
    header(s, "the runs", "Five runs, three models, nine defects found")
    rows = [
        ("Loop rig · 5 scenarios", "scripted",
         "All five behave as documented; 6 turns rejected, upstream would have taken 3"),
        ("End-to-end solver", "scripted",
         "Full gated path exercised — 2 defects found"),
        ("Live MLE-solver phase", "gemini-3.5-flash",
         "415s, 5 executions; caught a real NameError, engineer recovered — 2 defects"),
        ("Live ablation", "qwen3:8b local",
         "658s; gate on converged in 2 turns, gate off never converged in 3 — 2 defects"),
        ("Corpus benchmark", "qwen3:8b local",
         "68 labelled lines × 2 prompt variants"),
    ]
    y = Inches(1.75)
    for name, model, result in rows:
        panel(s, Inches(0.6), y, Inches(12.1), Inches(0.82))
        text(s, Inches(0.85), y + Inches(0.10), Inches(3.0), Inches(0.6),
             [(name, 13, True, INK)])
        text(s, Inches(3.9), y + Inches(0.13), Inches(2.4), Inches(0.5),
             [(model, 11, False, BLUE)])
        text(s, Inches(6.4), y + Inches(0.13), Inches(6.1), Inches(0.6),
             [(result, 11, False, INK2)])
        y += Inches(0.95)
    text(s, Inches(0.6), Inches(6.7), Inches(12.1), Inches(0.6),
         [("Four defects appeared only against a real model — and two of those only "
           "against the weaker one, which is the argument for testing on both.",
           12, True, ORANGE)])

    # 6 — impact
    s = blank(prs)
    header(s, "impact", "What the writing agent receives instead")
    panel(s, Inches(0.6), Inches(1.7), Inches(5.9), Inches(4.3))
    text(s, Inches(0.9), Inches(1.9), Inches(5.4), Inches(3.9),
         [("BEFORE", 13, True, ORANGE),
          ("1,000 characters of stdout, prefix-sliced, with the crash marker "
           "already fallen off the end.", 12, False, INK2),
          ("", 8, False, INK2),
          ("No way to tell a measured number from an invented one.", 12, False, INK2),
          ("", 8, False, INK2),
          ("A crashed run reaching the writer scored 1.0.", 12, False, INK2)])
    panel(s, Inches(6.8), Inches(1.7), Inches(5.9), Inches(4.3))
    text(s, Inches(7.1), Inches(1.9), Inches(5.4), Inches(3.9),
         [("AFTER", 13, True, BLUE),
          ("A verified registry: typed values, trace ids, provenance chains, run "
           "metadata, and the full untruncated capture on disk.", 12, False, INK2),
          ("", 8, False, INK2),
          ("results.values_computed rejects any value that is a source literal at "
           "its call site.", 12, False, INK2),
          ("", 8, False, INK2),
          ("A run that never passed cannot reach the writer at all — GateFailure "
           "ends the phase.", 12, False, INK2)])
    text(s, Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.8),
         [("Every WARN travels with the evidence under a heading saying it must be "
           "stated rather than omitted — and carries its line and reason, not just a count.",
           12, False, INK2)])

    # 7 — ablation
    s = blank(prs)
    header(s, "ablation", "Gate on versus gate off — same task, model and engineer")
    picture(s, "ablation.png", Inches(0.6), Inches(1.7), Inches(7.3), max_h=Inches(4.2))
    panel(s, Inches(8.2), Inches(1.7), Inches(4.5), Inches(4.4))
    text(s, Inches(8.5), Inches(1.9), Inches(4.0), Inches(4.0),
         [("false_success = 0", 16, True, INK),
          ("The divergence did not reproduce here. Every ungated failure crashed "
           "with zero bytes of stdout, so the marker stayed inside the slice and "
           "upstream's rule caught them all.", 11, False, INK2),
          ("", 8, False, INK2),
          ("That is the honest shape of the claim: the channel defect needs an "
           "experiment that prints past the ceiling before it dies.", 11, False, INK2),
          ("", 8, False, INK2),
          ("What did differ: the gated engineer got a report naming the failing "
           "check and fixed it; the ungated one got 1,000 raw characters and "
           "failed three times.", 11, False, INK2),
          ("", 8, False, INK2),
          ("n = 1, temperature 0. Suggestive, not a rate.", 11, True, ORANGE)])

    # 8 — cost
    s = blank(prs)
    header(s, "cost", "A third of the calls, a sixth of the tokens")
    picture(s, "callsplit.png", Inches(0.6), Inches(1.7), Inches(7.6), max_h=Inches(4.2))
    panel(s, Inches(8.5), Inches(1.7), Inches(4.2), Inches(4.3))
    text(s, Inches(8.8), Inches(1.9), Inches(3.7), Inches(3.9),
         [("Which is why it can run locally", 14, True, INK),
          ("qwen3:8b on an 8 GB laptop GPU — 5.5 GB resident, ~16s a call, zero "
           "marginal cost, no quota.", 11, False, INK2),
          ("", 8, False, INK2),
          ("The gate's prompts stay small because the digest collapses repetition; "
           "the engineer's do not, because they carry code, history and the "
           "evidence bundle.", 11, False, INK2),
          ("", 8, False, INK2),
          ("Putting the ENGINEER on a weak model is the tempting economy and the "
           "wrong one: it fails the gate more often, and each rejection costs a "
           "fixes call, a shadow-reward call and another turn.", 11, False, ORANGE)])

    # 9 — literature
    s = blank(prs)
    header(s, "the landscape", "What the published benchmarks measure")
    picture(s, "literature.png", Inches(0.6), Inches(1.65), Inches(7.5), max_h=Inches(4.4))
    panel(s, Inches(8.4), Inches(1.65), Inches(4.3), Inches(4.6))
    text(s, Inches(8.7), Inches(1.85), Inches(3.8), Inches(4.2),
         [("Why the verdict consults no model", 14, True, INK),
          ("BadScientist: fabricated papers accepted at up to 82.0%, with "
           "detection accuracy barely above random chance.", 11, False, INK2),
          ("", 8, False, INK2),
          ("If LLM reviewers detect fabrication at chance, a validity layer built "
           "on model judgement inherits that ceiling. Gate 1's verdict is "
           "therefore entirely deterministic.", 11, False, INK2),
          ("", 8, False, INK2),
          ("MLR-Bench: faked experimental results is the most prevalent "
           "hallucination class — present in almost every AI Scientist V2 paper. "
           "That is the class Gate 1 addresses, upstream of every reviewed "
           "approach.", 11, False, INK2)])

    # 10 — limits
    s = blank(prs)
    header(s, "limits", "What Gate 1 does not claim")
    bullets(s, Inches(0.6), Inches(1.8), Inches(12.1),
            ["It does not check whether a result is plausible — that is Gate 2",
             "It does not read the manuscript — that is Gate 3",
             "The static tier is conservative on purpose; the runtime tier catches "
             "what it declines to claim, one execution later",
             "The log scanner's recall is bounded by what it was shown, and the "
             "number of lines examined is recorded so the claim cannot exceed the evidence",
             "A mean computed inside the experiment over a silently shortened list "
             "arrives as one value Gate 1 cannot see behind (Gate 2's to close)",
             "n is still 1 for the archive comparison — the re-run has not been performed"],
            size=14, gap=2.0)
    panel(s, Inches(0.6), Inches(5.9), Inches(12.1), Inches(1.1), RGBColor(0xFB, 0xF5, 0xEC))
    text(s, Inches(0.9), Inches(6.05), Inches(11.5), Inches(0.9),
         [("The design documents paraphrase MLR-Bench's taxonomy. Its published names are "
           "faked experimental results, hallucinated methodology, incorrect citations, "
           "mathematical errors — any \"eliminates N of four classes\" claim should use "
           "those names or state the mapping.", 12, False, INK2)])

    # 11 — next
    s = blank(prs)
    header(s, "next", "Gate 1 is complete as an implementation")
    bullets(s, Inches(0.6), Inches(1.9), Inches(12.1),
            [("Measurement, not code, is what remains", 16, True, INK),
             ("·  Re-run the archive for a real n — both arms, needs a billed key "
              "and hand annotation", 14, False, INK2),
             ("·  Channel-fidelity writer arm at MAX_LEN ∈ {1000, 4000, 16000, ∞}",
              14, False, INK2),
             ("·  Capture integrity: a missing log must fail loudly, not read as clean",
              14, False, INK2),
             ("·  Re-execution verification — turn P10 from possible into performed",
              14, False, INK2),
             ("·  A second adapter, to make portability a demonstration rather than "
              "an argument", 14, False, INK2),
             ("", 10, False, INK2),
             ("Then Gate 2 — source ↔ result coherence, on the same model seam",
              16, True, BLUE)])
    return prs


def main():
    bench = json.loads(BENCH.read_text()) if BENCH.exists() else None
    run = json.loads(RUN_JSON.read_text()) if RUN_JSON.exists() else None
    if run is None:
        print(f'  WARNING: no {RUN_JSON} — run slides omitted')
    prs = build(bench, run)
    prs.save(OUT)
    print(f"  wrote {OUT.name} ({OUT.stat().st_size:,} bytes, "
          f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
