# Annotation files

Notebook `01_build_benchmark.ipynb` creates:

- `benchmark_manifest.csv`
- `manifest_lock.json`
- `annotations_annotator_a.csv`
- `annotations_annotator_b.csv`

After two independent passes, notebook `02_train_evaluate.ipynb` validates both files and creates `adjudication_queue.csv`. Final metrics remain blocked until every disagreement is resolved.

The repository intentionally does not ship fabricated labels or reuse the old web-mined class mappings as ground truth.
