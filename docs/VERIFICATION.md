# Verification Record

Core pipeline verification was refreshed on 2026-09-01 after adding semantic image screening and evaluation-focused double annotation.

## Passed

- `pytest`: 34 tests passed.
- Ruff lint: passed.
- Ruff format check: passed for `src/`, `scripts/`, and `tests/`.
- Python byte compilation: passed.
- Generated notebook validation: both notebooks are valid nbformat 4 JSON, every code cell parses, and no stale outputs/execution counts remain.
- Result publication consistency: `reports/RESULTS.md` exactly matches the renderer output from `artifacts/metrics.json`.
- Git whitespace validation: passed.
- Quality UI regression: a rejected collage advances to the next reserve candidate from the same class; accepted counts cannot exceed the target; decisions save atomically.
- Annotation UI regression: 43 mutually exclusive three-state controls, category grouping, save round-trip, and special-status exclusivity passed.
- Synthetic benchmark integration: created 21 images for each of the 13 frozen classes, rejected one semantic-quality candidate, replaced it within the same class, sealed a 260-row 12/4/4 packet, and verified the manifest, quality-screen, candidate-pool, config, ontology, prompt, and protocol digests.
- Annotation coverage integration: the synthetic packet produced 260 primary rows and exactly 104 secondary validation/test rows.

## Intentionally not claimed

- No real annotation sheet has been fabricated.
- No final CNN or VLM performance score exists yet.
- The full Kaggle GPU training/inference run cannot be validly executed until semantic screening, the 260-row primary pass, the 104-row secondary evaluation pass, and evaluation adjudication are complete.
- Dataset/model downloads and GPU memory/runtime remain environment-dependent checks for the first Kaggle execution; exact source versions and the Qwen revision are pinned, and the run manifest records the resolved environment.
