"""Retrieval-augmented few-shot for the log scanner.

Three things need holding: that the exemplar bank does not leak into the
evaluation corpus, that retrieval returns balanced classes rather than whichever
happens to score highest, and that it surfaces the analogue a given log actually
needs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gates import retrieval  # noqa: E402
from gates.exemplars import EXEMPLARS, NOISE, SIGNALS  # noqa: E402
from gates.llm import ModelLayer  # noqa: E402
from gates.llm_scan import scan_with_model  # noqa: E402
from rig.corpus import load_corpus  # noqa: E402


# --------------------------------------------------------------------------- #
# no leakage
# --------------------------------------------------------------------------- #


def test_the_exemplar_bank_is_disjoint_from_the_evaluation_corpus():
    """Retrieving few-shot examples from the eval set would leak the answers
    into the prompt and inflate every number the model tier is claimed on."""
    corpus = {e.line.strip() for e in load_corpus()}
    bank = {e.line.strip() for e in EXEMPLARS}
    assert corpus & bank == set()


def test_the_bank_carries_both_classes_in_useful_proportion():
    """Negatives matter more: the failure mode a model falls into unprompted is
    flagging anything containing the word 'error'."""
    assert len(SIGNALS) >= 8
    assert len(NOISE) >= 8


def test_every_exemplar_explains_itself():
    for exemplar in EXEMPLARS:
        assert exemplar.why.strip()
        assert len(exemplar.why) < 90


# --------------------------------------------------------------------------- #
# retrieval behaviour
# --------------------------------------------------------------------------- #


def test_selection_is_balanced_across_classes():
    """Unbalanced neighbours would bias the prompt toward whichever class the
    log happens to resemble."""
    picked = retrieval.select("Skipping 3 of 10 folds that raised", k_each=3)
    assert sum(e.signal for e in picked) == 3
    assert sum(not e.signal for e in picked) == 3


def test_neither_class_always_leads():
    """Models weight the first example most; which class leads should not be an
    accident of ordering."""
    picked = retrieval.select("anything at all", k_each=2)
    assert picked[0].signal != picked[1].signal


def test_a_dropped_replicate_retrieves_the_dropped_replicate_exemplar():
    picked = retrieval.select("Skipping 3 of 10 folds that raised during fitting")
    signals = [e.line for e in picked if e.signal]
    assert any("seeds crashed" in line or "skipped" in line for line in signals)


def test_framework_noise_retrieves_the_framework_noise_exemplar():
    """The hard negative is the more valuable neighbour here, since precision
    is the floor."""
    picked = retrieval.select("E0000 cuda_dnn.cc:8310] Unable to register cuDNN factory")
    noise = [e.line for e in picked if not e.signal]
    assert any("registration" in line or "registry" in line for line in noise)


def test_a_metric_named_error_retrieves_the_error_as_a_word_exemplars():
    picked = retrieval.select("Mean Squared Error: 0.0431")
    noise = [e.line.lower() for e in picked if not e.signal]
    assert any("error" in line for line in noise)


def test_selection_never_returns_nothing():
    """An empty few-shot block teaches nothing; a fixed one still beats none."""
    assert retrieval.select("", k_each=2)
    assert retrieval.select("zzzz qqqq xxxx", k_each=2)


def test_scoring_is_deterministic():
    a = [e.line for e in retrieval.select("loss nan at epoch 12")]
    b = [e.line for e in retrieval.select("loss nan at epoch 12")]
    assert a == b


def test_tokenizer_folds_case_and_splits_on_punctuation():
    assert retrieval.tokenize("Falling-back TO CPU!") == ["falling", "back", "to", "cpu"]


# --------------------------------------------------------------------------- #
# the rendered block, and the switch
# --------------------------------------------------------------------------- #


def test_the_block_labels_each_example_with_the_action_not_the_class():
    text = retrieval.render(list(retrieval.select("nan loss", k_each=1)))
    assert "REPORT" in text
    assert "IGNORE" in text


def test_few_shot_reaches_the_system_prompt():
    seen = {}

    def capture(prompt, system):
        seen["system"] = system
        return "[]"

    outcome = scan_with_model(
        ModelLayer(capture), "Skipping 3 of 10 folds that raised", "", few_shot=3
    )
    assert "Worked examples" in seen["system"]
    assert outcome.exemplars_used == 6


def test_few_shot_zero_sends_the_bare_prompt():
    """The arm the bench compares against."""
    seen = {}

    def capture(prompt, system):
        seen["system"] = system
        return "[]"

    outcome = scan_with_model(ModelLayer(capture), "some log line", "", few_shot=0)
    assert "Worked examples" not in seen["system"]
    assert outcome.exemplars_used == 0


def test_few_shot_costs_a_bounded_amount_of_prompt():
    """Retrieval is meant to sharpen the judgement, not to spend the saving the
    digest just made."""
    sizes = {}

    def capture(prompt, system):
        sizes[len(system)] = True
        return "[]"

    scan_with_model(ModelLayer(capture), "a line", "", few_shot=0)
    bare = max(sizes)
    sizes.clear()
    scan_with_model(ModelLayer(capture), "a line", "", few_shot=3)
    with_shots = max(sizes)
    assert with_shots - bare < 1500
