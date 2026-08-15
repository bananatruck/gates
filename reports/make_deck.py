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


# --------------------------------------------------------------------------- #


def build(bench):
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

    # 2 — the defect
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
    prs = build(bench)
    prs.save(OUT)
    print(f"  wrote {OUT.name} ({OUT.stat().st_size:,} bytes, "
          f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
