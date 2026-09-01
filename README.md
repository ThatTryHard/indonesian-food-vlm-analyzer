# Indonesian Food Visible-Ingredient Benchmark

A reproducible Kaggle and GitHub portfolio project comparing CNN and Vision-Language Model predictions on **visible food components in individual images**.

> **Current status:** benchmark construction is ready and human annotation is the next stage. Final scores remain unpublished until two independent annotation passes are complete and every disagreement has been adjudicated. See [Reproducible Results](reports/RESULTS.md).

## Project design

The project asks one focused question: how well can a conventional CNN and a zero-shot VLM recognize visually supported food components under the same label ontology and evaluation protocol?

The benchmark includes:

- one primary task, visible-component recognition in a specific image;
- a frozen 43-label phrase ontology with visual-evidence rules;
- 260 corruption-checked and globally deduplicated images across 13 source classes;
- sealed 12/4/4 train, validation, and test membership per class;
- two independent annotation passes followed by human adjudication;
- prevalence, frozen-backbone, adapted CNN, and zero-shot VLM baselines;
- validation-only checkpoint, threshold, and prompt decisions;
- grouped bootstrap confidence intervals and a paired permutation test;
- one machine-readable metrics artifact that generates the result report;
- a clear boundary that excludes image-only nutrition claims.

The rationale for these choices is documented in [Design Rationale](docs/DESIGN_RATIONALE.md). Pre-delivery checks are recorded in [Verification](docs/VERIFICATION.md). The original exploratory notebook is preserved under [`legacy/`](legacy/) for provenance, but its outputs are not used as current results.

## Run order

### 1. Build and annotate the benchmark

Open [`notebooks/01_build_benchmark.ipynb`](notebooks/01_build_benchmark.ipynb) in Kaggle.

It will:

1. download or resolve the exact target dataset;
2. verify images and detect exact/near duplicates;
3. create and hash the immutable 260-row manifest;
4. create two independently ordered annotation sheets;
5. launch an autosaving image annotation interface;
6. export a benchmark packet for the next notebook.

Read [Annotation Guide](docs/ANNOTATION_GUIDE.md) before labeling. Annotator A and B must not inspect each other's sheets.
The design uses 260 unique images but requires 520 independent judgments: 260 in each pass before adjudication.
For a second Kaggle session, attach the first exported packet; the notebook restores it into writable storage before resuming.

### 2. Adjudicate, train, and evaluate

Attach the completed benchmark packet and open [`notebooks/02_train_evaluate.ipynb`](notebooks/02_train_evaluate.ipynb).

It will:

1. verify the manifest lock and both completed sheets;
2. report inter-annotator agreement;
3. launch an adjudication interface for every disagreement;
4. optionally pretrain canonical labels on weak recipe-presence data;
5. train the CNN baselines using target train/validation only;
6. run the pinned zero-shot Qwen2-VL prompt on validation;
7. enforce an explicit sealed-test gate;
8. save raw predictions, confidence intervals, statistical tests, metrics, and a run manifest.

Detailed setup is in [Setup Guide](setup_guide.md); the frozen methodology is in [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md).

## Primary task and label policy

The task is not “recover the recipe from a photo.” The positive target is a component visibly supported in one image. Plausible but ambiguous components are stored separately as uncertain and excluded from the primary positive ground truth.

| Image evidence | Primary label behavior |
|---|---|
| Distinct rice grains | `rice` |
| Visible egg or omelette | `egg` |
| Brown meat with unclear species | uncertain meat label, not a confident positive |
| Garlic normally used in the dish but not visible | no `garlic` label |
| Dark glaze that might be sweet soy sauce | `dark_sauce` or uncertain, not hidden recipe inference |

The ontology is versioned at [`data/ontology/visible_ingredients.json`](data/ontology/visible_ingredients.json).

## Evaluation

All systems predict the same ontology. The final report includes:

- micro-F1;
- macro-F1 over labels with test support;
- sample-F1;
- micro-precision and recall;
- micro and supported macro average precision where continuous scores exist;
- precision@5 and exact set match;
- per-label support;
- food-class-grouped 95% bootstrap intervals;
- paired CNN vs. VLM permutation test;
- VLM food-name accuracy as a separate task;
- parse-failure, unknown-label, and abstention rates.

Web evidence may support development analysis but is never test truth. Nutrition requires recipes, ingredient mass, and cooked yield and is not inferred from an image.

## Repository structure

```text
├── notebooks/
│   ├── 01_build_benchmark.ipynb
│   └── 02_train_evaluate.ipynb
├── src/                         # modular pipeline
├── scripts/                     # benchmark, validation, publication utilities
├── data/ontology/               # frozen canonical ontology
├── docs/                        # protocol, cards, annotation and failure guides
├── tests/                       # deterministic unit and integrity tests
├── reports/RESULTS.md           # generated only from metrics.json
├── artifacts/metrics.json       # current honest project status
└── legacy/                      # original exploratory notebook for provenance
```

## Local quality checks

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-ci.txt
ruff check src scripts tests
pytest -q
python scripts/generate_notebooks.py
python scripts/validate_notebooks.py
python scripts/publish_results.py --metrics artifacts/metrics.json --output /tmp/RESULTS.md
diff -u reports/RESULTS.md /tmp/RESULTS.md
```

The full GPU experiment is designed for Kaggle and uses exact direct pins in [`requirements-kaggle.txt`](requirements-kaggle.txt); the completed run manifest records the resolved environment.

## Scope and responsible use

This is a portfolio research benchmark, not a production dietary tool. Do not use it for nutrition, allergies, medical advice, religious compliance, or hidden-ingredient detection. It represents 13 source-directory classes and does not represent all Indonesian cuisine. Review the [Model Card](docs/MODEL_CARD.md), [Data Card](docs/DATA_CARD.md), and [Failure Modes](docs/FAILURE_MODES.md) before interpreting results.

## Reproducibility and data access

Raw datasets and checkpoints are not redistributed. Dataset slugs, roles, model revision, prompt version, seeds, split counts, and evaluation settings are frozen in [`configs/project.json`](configs/project.json). Every completed run records the Git revision, package versions, dataset/annotation digests, manifest lock, raw VLM responses, and selected thresholds.
