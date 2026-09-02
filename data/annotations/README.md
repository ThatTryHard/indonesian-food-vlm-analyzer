# Annotation files

Notebook `01_build_benchmark.ipynb` creates:

- `quality_candidate_pool.csv`
- `quality_screen.csv`
- `benchmark_manifest.csv`
- `manifest_lock.json`
- `annotations_annotator_a.csv`
- `annotations_annotator_b.csv`

Annotator A covers all 260 images. Annotator B covers the 104 validation/test images independently. Notebook `02_train_evaluate.ipynb` validates both files and creates an evaluation-only `adjudication_queue.csv`. Final metrics remain blocked until every evaluation disagreement is resolved.

The repository intentionally does not ship fabricated labels or reuse the old web-mined class mappings as ground truth.
