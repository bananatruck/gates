# Gate 1 validation report

One package: the finalized submission and everything it cites.

Start at [`GATE1_FINAL_REPORT.pdf`](finalized-report-and-results/GATE1_FINAL_REPORT.pdf) in [`finalized-report-and-results/`](finalized-report-and-results/).

| Path | Contents |
|---|---|
| `GATE1_FINAL_REPORT.pdf` / `.html` | the combined report |
| `GATE1_FINAL_PRESENTATION.pdf` / `.pptx` | the 16-slide deck |
| `SUBMISSION_MANIFEST.json` | headline results, canonical run paths, config hash |
| `SHA256SUMS.txt` | checksum for every file in the package |
| `verification/evidence/` | campaign analysis, feedback-loop repeats, deterministic scenario repeats, shadow audit, log-scanner benchmark |
| `verification/execution-artifacts/` | per-attempt gate reports, registries, captured stdout/stderr |
| `verification/papers/` | the generated manuscript from each arm |
| `verification/logs/`, `verification/run-metadata/` | workflow logs and run manifests |
| `verification/source-reports/` | the two independent validation reports the combined one is built from |
| `verification/benchmark/` | the MLR-Bench PDF the comparison cites |

The [top-level README](../README.md#results-so-far) summarises the headline numbers and the boundary they do not support.

## Vintage

This package is frozen as submitted on 2026-08-16, and its `SHA256SUMS.txt` is byte-identical, so it still verifies.
Know two things before quoting from it.

- `SUBMISSION_MANIFEST.json` records `regression_tests: 362` and `legacy_blind_rejected_turns: 9/18`.
  Both moved after submission, so the current tree reports other numbers: `progress.md` holds the live test count, and the [top-level README](../README.md#gate-1-on-a-live-scaffold) the live rejection count, 6 turns of which the host would have accepted 3.
  The manifest is not wrong about the run it describes.
- `SHA256SUMS.txt` lists a package-local `README.md` that commit `6ee1f3e` removed when the three-folder report tree merged into this one.
  512 of the 513 listed files verify, and that one entry dangles.
  It stays, because a checksum file edited after the fact is worth less than one with a documented gap.
- 31 `gate1_report.json` files under `verification/evidence/gate1_loop_repeat_*` come from runs with no model, and they record `"model": {"calls": 1, "failures": 1, "degraded": true}` with the error `no model was supplied`.
  That call never happened.
  Since 2026-09-16 a run with no model records `"model": null`, and its feedback no longer says the model could not be reached.
  None of the frozen files carries that sentence, since the change touched only the JSON.

## Provenance

`build_final_submission.py` stays as the build record for this package, since it made the figures, tables and copied evidence.
It does not run from a fresh clone.
It read a report tree of three folders, `claude-research/` and `codex-research/` among them, that has since been merged into this one package.
The outputs it produced, and every input it copied, are in `verification/`.

The campaign itself reproduces from the host scaffold.
From the host checkout, with the key read through a prompt that does not echo, so it never reaches argv, a config file or a log:

```bash
.venv/bin/python tools_full_gate1_ablation.py --prompt-key \
  --config experiment_configs/gate1_common_deepseek.yaml \
  --outdir full_ablation_runs/<new-run>
```

No file in this tree contains a provider credential; both run manifests record `credential_persisted: false`.
