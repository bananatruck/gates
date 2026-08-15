"""Few-shot exemplars for the log scanner — deliberately disjoint from the corpus.

`tests/fixtures/log_corpus.jsonl` is the **evaluation** set. Retrieving few-shot
examples from it would leak the answers into the prompt and inflate every number
the model tier is claimed on, so nothing here appears there. The two sets cover
the same decision boundary in different words: the bank teaches it, the corpus
tests it.

The bank is small and hand-written on purpose. Retrieval picks a handful per
scan, so a large bank would mostly be dead weight, and every entry here has to
earn its place by teaching a distinction the model gets wrong without it.

Negatives matter more than positives. Precision is the floor — a false positive
costs the ML engineer a rewrite for nothing — and the failure mode a model falls
into unprompted is flagging anything containing the word "error". Roughly half
the bank is therefore noise that looks alarming.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Exemplar:
    line: str
    #: True when this is trouble the run continued past.
    signal: bool
    #: The reason, in the terms the scanner is asked to reason in. Shown to the
    #: model, so it teaches the distinction rather than just the label.
    why: str


EXEMPLARS: tuple[Exemplar, ...] = (
    # ---- signals: substituted or degraded inputs --------------------------- #
    Exemplar(
        "Could not open shard 4 of 8; continuing with the shards that loaded",
        True,
        "the training set is not the one described",
    ),
    Exemplar(
        "checksum mismatch on cifar-10 batch 3, re-downloading failed, using cache",
        True,
        "the data may not be what the run says it is",
    ),
    Exemplar(
        "vocab file absent -- falling back to whitespace tokenisation",
        True,
        "the method is not the one described",
    ),
    # ---- signals: the metric itself is compromised -------------------------- #
    Exemplar(
        "val_loss=-0.0000  (clipped from -inf)",
        True,
        "a non-finite value was hidden by clipping rather than fixed",
    ),
    Exemplar(
        "warning: 0 samples in class 6, precision set to 0 for that class",
        True,
        "the macro average is over a class that was never evaluated",
    ),
    Exemplar(
        "evaluating on train split (val split not found)",
        True,
        "the reported accuracy is not held out",
    ),
    # ---- signals: silently dropped work ------------------------------------ #
    Exemplar(
        "2 of 5 seeds crashed; reporting the mean of the 3 that finished",
        True,
        "an average over a silently shortened set",
    ),
    Exemplar(
        "batch 17 raised and was skipped",
        True,
        "part of the epoch did not happen",
    ),
    # ---- signals: the run changed its own configuration -------------------- #
    Exemplar(
        "requested fp16 unavailable on this device, continuing in fp32",
        True,
        "precision differs from the configuration, which matters for a timing claim",
    ),
    Exemplar(
        "num_workers reduced 8 -> 0 after a worker died",
        True,
        "wallclock is no longer comparable to the configured run",
    ),
    # ---- signals: swallowed failure ---------------------------------------- #
    Exemplar(
        "except: pass  # noqa  -- caught TypeError during eval, continuing",
        True,
        "a failure during evaluation was suppressed",
    ),
    Exemplar(
        "retry 3/3 failed, returning partial results",
        True,
        "the results are partial and are not labelled as such",
    ),
    # ---- noise: alarming words, no bearing on any number ------------------- #
    Exemplar(
        "successfully registered error handler for SIGTERM",
        False,
        "the word error, no error",
    ),
    Exemplar(
        "computing per-class error bars over 5 seeds",
        False,
        "error as a statistical term",
    ),
    Exemplar(
        "loaded error_analysis.py",
        False,
        "a filename",
    ),
    Exemplar(
        "W0815 03:11:02.000001 12345 plugin_registry.cc:88] duplicate registration ignored",
        False,
        "framework startup noise, present on every healthy run",
    ),
    Exemplar(
        "NCCL WARN Bootstrap: no socket interface found, using eth0",
        False,
        "resolved during setup, before anything was measured",
    ),
    Exemplar(
        "DeprecationWarning: pkg_resources is deprecated as an API",
        False,
        "a deprecation changes nothing about the numbers",
    ),
    Exemplar(
        "Downloading builder script: 100%|##########| 5.60k/5.60k [00:00<00:00]",
        False,
        "a progress bar",
    ),
    Exemplar(
        "early stopping: no val improvement for 20 epochs, best was epoch 63",
        False,
        "early stopping on a converged fit is correct behaviour",
    ),
    Exemplar(
        "eval loss 0.3128 | eval accuracy 0.8842 | eval runtime 4.02s",
        False,
        "an ordinary result line",
    ),
    Exemplar(
        "Trainer is attempting to log a value of 0.0 for key train/grad_norm",
        False,
        "a logging note about a legitimate zero",
    ),
)

SIGNALS = tuple(e for e in EXEMPLARS if e.signal)
NOISE = tuple(e for e in EXEMPLARS if not e.signal)
