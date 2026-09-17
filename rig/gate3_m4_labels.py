"""Hand labels for G3-M4: which numbers in the archived manuscripts are claims.

D37: Kesh reviews this file before the G3-M4 figure is recorded, because the
paper quotes that figure and this is the one Gate 3 number with a human judgement
behind it rather than a deterministic check. Everything downstream in
``rig/gate3_scanner_miss.py`` is derived from these rows.

**What counts as a claim.** A number the manuscript asserts as an observed
quantity from *this run*. That includes a loss value quoted out of the captured
log, because the writer typed it into a findings section and the gate's job is to
demand it be bound to a recorded value.

**What does not.** In every case the test is whether the run could have measured
it differently:

* Hyperparameters and design constants the paper itself sets: sample counts,
  step counts, feature and class counts, the seed, the test-set size.
* Subscript identifiers. ``a_{25}`` and ``\\texttt{eff.acc\\_at\\_100}`` contain
  digits that name a quantity rather than state one.
* Thresholds fixed before the run. The ungated paper's ``r >= 0.99`` decision
  rule was chosen in advance and is not a result.
* Mathematical constants. ``\\ln 3 \\approx 1.0986`` is arithmetic, not a
  measurement.
* Step indices in a loss table, figure numbers, and dates inside file paths.
* Properties of the harness the paper describes rather than measures, such as the
  1,000-character evidence budget.

Labelling in the permissive direction would inflate the miss count and flatter
the argument, so each exclusion above errs the other way. The consequence is that
the reported miss rate is a floor, and the readout says so.

Labels are pinned to a content hash. Editing a manuscript invalidates the line
numbers, and a stale label silently attaching to a different number is exactly
the failure this project exists to catch.
"""

from __future__ import annotations

#: sha256 of each manuscript these labels were read against. The papers live in
#: the frozen, checksum-signed submission package and must not change.
MANUSCRIPT_SHA256 = {
    "gated": "e0554f56b0f300d7341df1e1075850a58f7854191a8379dd04ec865d99644ff8",
    "ungated": "9fb824a23b15723ed45c22f661f1f916daff5a4d5c6033335022ef63ecc7a195",
}

#: Every number I judged an empirical claim of the run, as ``(line, tokens)``.
#: Tokens are listed in the order they appear on the line, repeats included, and
#: exactly as the manuscript writes them.
CLAIMS: dict[str, dict[int, tuple[str, ...]]] = {
    # The gated arm reports four recorded metrics and restates them throughout.
    "gated": {
        # Abstract: the two accuracies, the ratio, its percentage form, both
        # error rates, the power-law exponent, the wall-clock time.
        16: ("0.89", "0.97", "0.9175257731958764", "91.75", "0.11", "0.03",
             "0.94", "0.017149006998806726"),
        189: ("0.89", "0.97"),
        192: ("0.89", "0.97"),          # the ratio written as a fraction
        193: ("0.9175257731958764",),
        195: ("91.75", "0.017149006998806726"),
        205: ("0.89",),                 # the recorded-metrics table
        206: ("0.97",),
        207: ("0.9175257731958764",),
        208: ("0.017149006998806726",),
        213: ("0.2277", "0.2707"),      # final training loss, both arms
        222: ("10.5", "0.11", "0.03", "3.7"),
        224: ("0.11", "0.03", "0.94"),  # the exponent, derived in an equation
        235: ("0.08", "16"),            # accuracy gap, and it as sample count
        238: ("91.75",),
        240: ("0.94", "0.94"),          # stated twice on one line
        242: ("0.9175", "0.94"),
        248: ("0.89", "0.97"),
        254: ("0.9175257731958764", "91"),
    },
    # The ungated run's metrics never reached the evidence window, so this paper
    # states no accuracy and no efficiency ratio. Its only claims are the loss
    # values it quotes out of the partial log.
    "ungated": {
        276: ("1.098612",),             # the log line the paper reproduces
        280: ("0.432672", "60.6", "1.098612", "0.553408"),
        290: ("1.098612",),             # the observed-loss table
        291: ("0.553408",),
        292: ("0.481476",),
        293: ("0.450088",),
        294: ("0.432672",),
    },
}

#: Numbers the scanner reports as claims that are not claims, with the reason.
#: Declared so the harness can tell a false positive from a labelling gap: any
#: scanner finding that is neither a labelled claim nor listed here fails the
#: cross-check rather than being quietly absorbed.
FALSE_POSITIVES: dict[str, dict[int, tuple[tuple[str, str], ...]]] = {
    "gated": {
        # A value from the cited literature, in the Discussion. The scanner
        # cannot tell whose number it is, and Discussion is a findings section.
        240: (("0.5", "predicted exponent quoted from arXiv 2412.07942v1"),),
    },
    "ungated": {
        # "0000" is the step index in "[arm=25][step=0000]", read as 0.0.
        276: (("0000", "step index in a quoted log line, parsed as 0.0"),),
        305: (
            ("0.99", "decision threshold fixed before the run"),
            ("0.95", "decision threshold fixed before the run"),
        ),
        322: (("1500", "step count the paper specifies"),),
        326: (("1500", "step count the paper specifies"),),
    },
}

#: Why each labelled claim the scanner does not report was missed. Keyed by
#: ``(arm, line)`` because within a line the cause is the same for every token.
#: The three names are the readout's classes (`docs/research/
#: gate2-gate3-literature-readout.md` §6), plus ``duplicate_context`` which that
#: probe did not reach.
MISS_CAUSE: dict[tuple[str, int], str] = {
    ("gated", 195): "skipped_line",      # "listed in Table~\ref{tab:metrics}"
    ("gated", 213): "skipped_line",      # "shown in Figure~\ref{fig:loss}"
    ("gated", 222): "skipped_line",      # "shown in Figure~\ref{fig:acc}"
    ("gated", 235): "small_integer",     # "16 correctly classified"
    ("gated", 240): "duplicate_context", # 0.94 twice, one context, deduped
    ("gated", 254): "small_integer",     # "more than 91\%"
    ("ungated", 280): "skipped_line",    # "Table~\ref{tab:observed-loss}"
}

#: What each cause is, for the readout. ``skipped_line`` and ``small_integer``
#: are the readout's; ``duplicate_context`` is not, and is the one this
#: measurement found.
CAUSES = {
    "skipped_line": (
        "The line carries a \\ref, which SKIP_LINE matches, so every number on "
        "it is dropped and the line reports nothing. The readout found this "
        "with a \\cite probe; in real manuscripts \\ref is the common trigger, "
        "because a findings sentence points at the table or figure it discusses."
    ),
    "small_integer": (
        "NUMBER requires a decimal point or four digits, so a two- or "
        "three-digit integer result is invisible by construction."
    ),
    "duplicate_context": (
        "context_of() locates a token with line.find(), which always returns "
        "the first occurrence, so a value stated twice on one line yields one "
        "context and the second is deduplicated away. This understates the "
        "literal count; it does not let a line through, since the first "
        "occurrence still reports."
    ),
    "invisible_section": (
        "The claim sits in a \\begin{abstract} environment, which _heading does "
        "not match, so the scanner never enters the section and reads none of "
        "its lines. D40 leaves this open deliberately: teaching the scanner to "
        "read the environment would restate the published Gate 1 number."
    ),
}
