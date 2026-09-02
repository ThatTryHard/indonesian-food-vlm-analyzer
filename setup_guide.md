# Setup Guide

## Recommended environment

Use Kaggle with one NVIDIA GPU, internet enabled for the first dependency/model download, and persistent output or manual artifact downloads. The CNN can run without a GPU but Qwen2-VL evaluation is explicitly blocked on CPU.

## Notebook 01

1. Import `notebooks/01_build_benchmark.ipynb` from GitHub.
2. Enable internet. GPU is optional for this notebook.
3. Run cells from the top. The notebook clones the repository and uses Kaggle's compatible base environment without replacing NumPy or pandas inside the live kernel.
4. The target dataset resolves from the exact slug in `configs/project.json`. To use an attached copy, set:

   ```python
   os.environ["INDONESIAN_FOOD_DATASET_DIR"] = "/kaggle/input/your-exact-folder"
   ```

5. Review the semantic quality interface. Accept one clear food photograph; reject collages, heavy overlays, wrong classes, unrelated multi-dish scenes, non-food, unreadable, or sensitive images. Each rejection automatically advances to a same-class reserve candidate.
6. Rerun the quality finalization cell after every class reaches 20 accepted images. This is the first point at which the 12/4/4 split and manifest are sealed.
7. Read `docs/ANNOTATION_GUIDE.md`.
8. Set `ANNOTATOR_ID="annotator_a"`, label all 260 images, and download the packet.
9. Give the packet to a different person in a separate session. Set `ANNOTATOR_ID="annotator_b"`; this sheet contains only the 104 validation/test images. Do not inspect A's answers.
10. Download `benchmark_packet.zip` after every work session.

The interface saves every confirmed row immediately, but Kaggle storage is temporary; keep external copies.

## Notebook 02

1. Upload the completed packet as a private Kaggle dataset and attach it.
2. Import `notebooks/02_train_evaluate.ipynb`.
3. Enable GPU and internet.
4. If more than one manifest exists in attached inputs, explicitly set:

   ```python
   os.environ["FOOD_VLM_BENCHMARK_PACKET"] = "/kaggle/input/your-packet/benchmark_packet.zip"
   ```

5. Run annotation validation.
6. Resolve every row in the adjudication interface and rerun finalization.
7. Run weak pretraining and target training. Set `RUN_WEAK_RECIPE_PRETRAINING=False` only if compute is unavailable; the ImageNet linear probe remains valid.
8. Inspect validation outputs. Do not use test data to change prompts, aliases, thresholds, or model selection.
9. When every choice is frozen, change `OPEN_SEALED_TEST=True` once and run final cells.
10. Preserve `/kaggle/working/indonesian-food-vlm-artifacts/run/`.
11. Download `github-publication-overlay.zip`, unzip it over the repository, review the generated report, and commit the lightweight metrics/report files. Do not commit raw images, checkpoints, VLM responses, or per-image predictions.

## Environment variables

| Variable | Meaning |
|---|---|
| `FOOD_VLM_PROJECT_ROOT` | Existing repository location |
| `FOOD_VLM_ARTIFACT_DIR` | Writable artifact root |
| `INDONESIAN_FOOD_DATASET_DIR` | Exact target dataset root |
| `FOOD_RECIPE_DATASET_DIR` | Exact weak-pretraining dataset root |
| `INDONESIAN_RECIPE_CORPUS_DIR` | Optional recipe corpus root |
| `FOOD_VLM_BENCHMARK_PACKET` | Completed packet directory, manifest path, or ZIP |
| `FOOD_VLM_REVISION` | Git branch/tag to clone; defaults to `main` |

## Local testing

Python 3.11 or 3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ci.txt
ruff check src scripts tests
pytest -q
```

For a local GPU experiment, install a matching PyTorch/torchvision build for your CUDA environment, then install the remaining packages from `requirements-kaggle.txt`. The pinned Kaggle pair is `torch==2.10.0` and `torchvision==0.25.0`.

## Common blockers

### Manifest already exists

This is intentional protection. Reuse the locked manifest. Do not alter quality decisions or replace images after ingredient annotation starts. A deliberately new benchmark version must use a fresh output directory and new lock.

### Quality screen is incomplete

The manifest does not exist yet. Continue the quality interface until every class has 20 accepted images, rerun the finalization cell, and download the resumable packet. Blank reserve rows are expected and do not need review after a class reaches 20 acceptances.

### Older packet is rejected

Packets created before semantic screening do not contain `quality_candidate_pool.csv` or `quality_screen.csv` and cannot be upgraded into the new locked protocol. Start Notebook 01 in a fresh session. GitHub history, source code, and the original Kaggle dataset remain unchanged; only the old benchmark packet is replaced.

### An unsuitable image appears during annotation

Do not mark a collage as `Not food`, and do not combine labels across panels. Stop the pass and review whether the wrong packet was attached. Semantic failures belong in the pre-split quality screen.

### Annotation validation fails

The error identifies blank rows, unknown ontology IDs, visible/uncertain overlap, invalid booleans, or sample mismatch. Correct the sheet rather than weakening validation.

### Adjudication remains blocked

Every disagreement requires a non-empty rationale, even when the correct resolution is an empty visible set.

### Qwen cannot load

Confirm CUDA, internet/model access, and the exact revision in `configs/project.json`. Do not silently switch models mid-experiment; update the config and start a new run.

### Test gate stops execution

Expected until `OPEN_SEALED_TEST=True`. Open it only after all validation decisions are final.
