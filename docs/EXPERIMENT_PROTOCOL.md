# Experiment Protocol

## Primary claim

The only primary modeling claim is performance on **per-image visible-ingredient recognition**. Full-recipe ingredients, likely hidden ingredients, web-mined associations, dish recognition, and nutrition are separate outcomes and must not be combined into one F1 score.

## Frozen benchmark

1. Inventory and verify every target image.
2. Group exact duplicates and near duplicates globally, including cross-class copies.
3. Select 20 unique images per class using seed 42.
4. Seal 12/4/4 train/validation/test images per class before annotation.
5. Lock the ontology, zero-shot prompt, config, and protocol-code digest at the same time.
6. Complete two independent annotation passes.
7. Report agreement and adjudicate every disagreement.

The test split is never used to build aliases, select prompts, tune thresholds, choose checkpoints, or select qualitative examples.

## Trade-off: balance versus multi-label stratification

With only 20 annotated images per class, using the finished labels to rearrange the test set would leak label information into benchmark design. The target benchmark therefore prioritizes pre-annotation class balance and a sealed test. Within larger recipe-pretraining data, grouping and iterative multi-label stratification are used because there is sufficient scale and no human benchmark is being constructed.

## Training stages

### Stage A: weak recipe-presence pretraining

- Labels are canonical multiword ontology phrases extracted from recipe ingredient lists.
- These labels mean “listed in recipe,” not “visible.”
- Quarantine exact and near image matches against all 260 sealed target images.
- Build connected split groups from normalized recipe titles and duplicate-image groups before iterative multi-label stratification.
- This stage initializes visual features only and is never reported as target benchmark performance.

### Stage B: visible-ingredient fine-tuning

- Fine-tune on adjudicated benchmark train images.
- Compare a frozen ResNet18 linear probe with a last-block-plus-head model.
- Select checkpoint, threshold, and prompt solely on validation.
- Include a prevalence/top-*k* baseline.

The delivery target is a Kaggle/GitHub research portfolio, not an operating decision system, so there is no legitimate false-positive/false-negative cost matrix to invent. Checkpoint/model selection uses validation micro-average-precision and the global decision threshold uses validation micro-F1; precision and recall are reported separately. Any later product use must replace that threshold rule with an explicit cost or utility function supplied by the owner.

### Stage C: frozen VLM comparison

- Use `visible-zero-shot-v1`; no evaluated dish examples appear in the prompt.
- Schema and ontology are frozen before validation comparison.
- Unknown labels, malformed JSON, and abstentions are retained and reported.
- Run once on sealed test after all choices are frozen.

## Primary metrics

- Micro-F1.
- Macro-F1 over labels with test support.
- Sample-F1.
- Micro-precision and micro-recall.
- Micro and supported macro average precision.
- Precision@5 and exact set match.
- Per-label support and performance.
- Food-name accuracy for the VLM as a separate task.

Report 95% food-class-grouped bootstrap intervals and a paired permutation test for the CNN vs. VLM sample-F1 difference.

For sample-F1, a correctly predicted all-negative image receives 1.0; an all-negative ground-truth image with any false positive receives 0.0. This convention is required because `no_visible_ontology_label` is a legitimate evaluable state.

## Test gate

Final evaluation is allowed only if all of these are present:

- completed and validated annotation sheets;
- zero unresolved adjudication rows;
- manifest digest matches `manifest_lock.json`;
- frozen ontology and prompt versions match the config;
- model checkpoint and validation-selected threshold are saved;
- VLM raw responses and parser status are saved;
- run manifest records package, model, dataset, config, and Git revisions.

If the gate fails, publish no final score.

## Web mining

Web evidence is optional development material. Each record stores query, UTC timestamp, URL, domain, rank, and content hash. Labels require votes from at least two independent domains. Web evidence may guide ontology-gap review on train/validation but is never test ground truth.

## Nutrition

Image-only nutrition estimation is out of scope. A separate future task may use standardized recipes with ingredient mass and cooked yield. Per-100g records must pass macro mass and Atwater-energy consistency checks before display.

## LoRA

LoRA is not part of the primary experiment. If revisited, training labels must mask all tokens before the assistant response and use the attention mask rather than token value to exclude padding. Train, validation, and test images must not overlap, and the adapter must outperform the frozen VLM on validation before sealed-test evaluation.
