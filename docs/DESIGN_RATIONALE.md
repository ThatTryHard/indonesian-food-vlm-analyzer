# Design Rationale

This document explains the main methodological choices behind the benchmark. It is written as part of the current study design, not as a repair log.

## Research target

The primary task is per-image visible-component recognition. Recipe ingredients, likely hidden ingredients, dish naming, web associations, and nutrition are treated as separate outcomes because they require different evidence.

The target ontology contains 43 canonical phrase labels. Each label has a visual-evidence rule, supported aliases, and a stable identifier. Preparation words, quantities, and recipe-writing fragments are not labels. Annotators can also record uncertain components, unreadable images, non-food images, and valid cases where no supported ontology label is visible.

## Benchmark construction

The benchmark contains 20 images from each of 13 source-directory classes. Image files are validated before sampling, and exact or perceptual duplicates are grouped across the complete target dataset. Each class contributes 12 training images, 4 validation images, and 4 test images.

Split membership is sealed before annotation. This keeps finished labels from influencing test composition. Every sampled image, configuration file, ontology file, VLM prompt, and protocol implementation is recorded by hash so later runs can verify that they use the same benchmark.

Two annotators label all 260 images independently. Exact agreements are accepted, while disagreements require a documented human decision. This avoids treating one class-level recipe list as ground truth for every image and avoids automatically resolving difficult cases by union or intersection.

## Model comparison

All evaluated systems predict the same ontology. The comparison begins with a prevalence baseline, followed by a frozen ResNet18 linear probe, a ResNet18 model with its final block adapted, and a pinned zero-shot VLM.

Optional recipe-presence pretraining is kept separate from visible-component evaluation. Recipe images that match any sealed benchmark image are quarantined, and recipe-presence scores are never reported as benchmark performance.

The CNN checkpoint and decision threshold are selected using validation data only. The VLM model revision, prompt, parser schema, and aliases are also frozen before the test split is opened. The sealed test is evaluated once after those choices have been recorded.

## Evaluation and reporting

The final report includes micro-F1, supported macro-F1, sample-F1, micro-precision, micro-recall, average precision, precision at 5, exact set match, and per-label support. Confidence intervals use food-class-grouped bootstrap resampling. The selected CNN and VLM are compared with a clustered paired permutation test.

All report values come from one machine-readable metrics artifact. If annotation, adjudication, or the sealed-test gate is incomplete, the publication script reports a blocked status instead of generating provisional headline scores.

## Scope boundaries

Web evidence is allowed only for development analysis and cannot become final ground truth. Image-only nutrition estimation is outside the benchmark because a photograph does not provide recipe quantities, cooked yield, or portion mass. LoRA adaptation is also outside the primary comparison unless a future experiment uses clean assistant-only targets and demonstrates a validation improvement without data overlap.

The deliverable is a Kaggle and GitHub research portfolio project. It is not a dietary, allergy, religious-compliance, or medical decision tool.

## Chosen trade-offs

- Label validity takes priority over vocabulary size.
- A sealed pre-annotation test takes priority over rearranging samples after labels are known.
- Duplicate quarantine takes priority over retaining every weak-pretraining image.
- Simple baselines take priority over a more complex adaptation experiment without sufficient evidence.
- Valid human annotation takes priority over publishing immediate scores.
