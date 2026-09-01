# Verification Record

Core pipeline verification was completed on 2026-08-13. The portfolio-language refresh was reverified on 2026-09-01.

## Passed

- `pytest`: 28 tests passed.
- Ruff lint: passed.
- Ruff format check: passed for `src/`, `scripts/`, and `tests/`.
- Python byte compilation: passed.
- Generated notebook validation: both notebooks are valid nbformat 4 JSON, every code cell parses, and no stale outputs/execution counts remain.
- Result publication consistency: `reports/RESULTS.md` exactly matches the renderer output from `artifacts/metrics.json`.
- Portfolio-language check: current README, documentation, source files, and generated notebooks contain no audit severity labels, weakness IDs, rebuild framing, or dash-heavy headings.
- Git whitespace validation: passed.
- Synthetic benchmark integration: created 21 images for each of the 13 frozen classes, produced a 260-row packet, verified 12/4/4 per-class splits and every manifest/config/ontology/prompt/protocol digest, validated two complete annotation sheets, and produced one required adjudication row.

## Intentionally not claimed

- No real annotation sheet has been fabricated.
- No final CNN or VLM performance score exists yet.
- The full Kaggle GPU training/inference run cannot be validly executed until the two 260-image annotation passes and adjudication are complete.
- Dataset/model downloads and GPU memory/runtime remain environment-dependent checks for the first Kaggle execution; exact source versions and the Qwen revision are pinned, and the run manifest records the resolved environment.
