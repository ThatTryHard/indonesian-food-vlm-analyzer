# Model Card: Indonesian Food Visible-Ingredient Benchmark

## Status

Semantic screening and benchmark construction are complete in code; human screening and annotation are pending. No final performance claim is currently valid.

## Intended use

- Educational and portfolio analysis of visible components in the 13 represented food classes.
- Comparison of a CNN and a frozen VLM under one fixed ontology and sealed test protocol.

## Out-of-scope use

- Medical, dietary, allergy, religious-compliance, or nutrition decisions.
- Identifying hidden ingredients or trace allergens.
- Generalizing to all Indonesian cuisines, regions, preparation styles, or restaurant conditions.
- Automated decisions about people.

## Data

- Weak recipe-image data for optional feature pretraining.
- A 260-image target benchmark selected from 13 directory classes after human semantic screening.
- Annotator A covers all 260 images; Annotator B independently covers the 104 validation/test images, followed by evaluation-set adjudication.

Burger, Pizza, and French Fries are included in the source dataset; therefore the benchmark is a mixed food-class collection rather than a comprehensive Indonesian-cuisine census.

## Known risks

- Regional recipe variation and culturally stereotyped ingredient priors.
- Ambiguous meat types and visually hidden components.
- Domain shift in lighting, plating, camera, restaurant, and household images.
- VLM outputs can be fluent but unsupported.
- Small per-class test counts produce wide uncertainty intervals.
- Training labels come from one annotator, so training-label noise is not captured by evaluation-set agreement statistics.
- Grouped intervals use only 13 food-class clusters and should be treated as exploratory, not population-wide inference.
- The pre-registered headline run uses one deterministic training seed; its intervals do not include between-seed optimization variance.

## Guardrails

- Visible-only ontology.
- Explicit uncertainty and abstention.
- Schema validation and unknown-label accounting.
- Human benchmark annotations.
- No image-only nutrition output.
- Final scores blocked until the complete test gate passes.

## Monitoring if later deployed

Deployment is not currently targeted. A future deployment would require input-quality monitoring, unknown-dish detection, abstention-rate tracking, label/prevalence drift, latency/memory budgets, human corrections, privacy review, and periodic re-annotation.
