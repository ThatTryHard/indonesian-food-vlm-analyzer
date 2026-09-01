# Visible-Ingredient Annotation Guide

## Purpose

The original project evaluated recipe-derived and web-derived class priors as if they were per-image ingredient truth. This guide replaces that invalid target with a human-annotated benchmark: **mark only components supported by the specific image**.

## Benchmark design

- 13 food classes.
- 20 de-duplicated images per class.
- 260 images total.
- 520 independent image judgments in total (260 per pass), plus adjudication of disagreements.
- Sealed before annotation: 12 train, 4 validation, and 4 test images per class.
- Two independent annotation sheets followed by explicit adjudication.
- The test split must never be opened for prompt design, alias editing, threshold selection, or qualitative cherry-picking.

## Labels

Use canonical IDs from `data/ontology/visible_ingredients.json`. The annotation interface shows the corresponding hints.

The interface groups ingredients into collapsible categories. Each ingredient has one mutually exclusive state:

- **None**: the default; the image does not support the label.
- **Visible**: the component is visually identifiable.
- **Uncertain**: a component is present, but its identity is visually ambiguous.

Open only the categories relevant to the image, choose the supported states, then click **Save + Next**. The current row is written to the annotation CSV immediately. Do not use Ctrl-click or try to select the same ingredient in two lists.

### `visible_ingredients`

Select a label only when visual evidence supports it in this image. Examples:

- Distinct rice grains → `rice`.
- A visible egg or omelette → `egg`.
- A lime wedge → `lime`.
- Clearly visible fried-shallot garnish → `fried_shallot`.

### `uncertain_ingredients`

Use this when a component is plausible but visual evidence is insufficient. Examples:

- Brown meat that might be beef or chicken.
- A dark glaze that might be sweet soy sauce.
- A pale sauce that might contain coconut milk.

Uncertain labels are retained for error analysis but are **not positive ground truth in the primary metric**.

## Forbidden inference

Do not label an ingredient solely because:

- it normally appears in the named dish;
- it appeared in a recipe or search result;
- it was present in a prompt example;
- it is probably hidden in a sauce or spice paste.

For example, do not label garlic, salt, turmeric, oil, or coconut milk unless the chosen ontology label has direct visual support under its stated rule.

## Image-level exclusions

- `unreadable=True`: the image cannot be assessed because of corruption, extreme blur, or obstruction.
- `non_food=True`: the image does not contain an assessable food dish.
- `no_visible_ontology_label=True`: the food is assessable, but none of the frozen ontology labels is visibly supported. This is a valid all-negative target row, not an exclusion.
- Use `notes` for ambiguous meat types, mixed dishes, or ontology gaps.

The three flags are mutually exclusive and cannot be combined with visible or uncertain labels. Never force a label merely to complete a row.

## Independent passes

Annotator A and Annotator B must work independently. They must not inspect each other's labels before both sheets are complete. If only one human annotator is available, perform the second pass after a washout period and disclose this limitation; it is not equivalent to two independent annotators.

The notebook interface hides the directory class during annotation to reduce recipe-prior bias. Do not inspect the manifest's `food_class` column while labeling.

## Adjudication

The pipeline automatically accepts exact agreement. Every disagreement appears in `adjudication_queue.csv`; a human must fill `resolved_visible_ingredients`, `resolved_uncertain_ingredients`, and `resolution_notes`. The evaluation notebook refuses to continue while any disagreement is unresolved.
