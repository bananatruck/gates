# Gate 1 validation report

One package: the finalized submission and everything it cites.

[`finalized-report-and-results/`](finalized-report-and-results/) — start at
[`GATE1_FINAL_REPORT.pdf`](finalized-report-and-results/GATE1_FINAL_REPORT.pdf).

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

The headline numbers, and the boundary they do **not** support, are summarised in
the [top-level README](../README.md#measured-results).

## Provenance

`build_final_submission.py` is retained as the build record for this package: it
is how the figures, tables and copied evidence got here. It is **not runnable
from a fresh clone** — it read a three-folder report tree (`claude-research/`,
`codex-research/`) that has since been collapsed into this single package. The
outputs it produced, and every input it copied, are in `verification/`.

The campaign itself is reproducible from the host scaffold. From
`/home/kesh/AgentLaboratory-Gemini`, reading the key through a non-echoing
prompt so it never reaches argv, a config file or a log:

```bash
.venv/bin/python tools_full_gate1_ablation.py --prompt-key \
  --config experiment_configs/gate1_common_deepseek.yaml \
  --outdir full_ablation_runs/<new-run>
```

No file in this tree contains a provider credential; both run manifests record
`credential_persisted: false`.
