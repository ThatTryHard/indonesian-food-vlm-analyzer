# Visible-Ingredient Annotation Guide

## Purpose

This guide defines a human-annotated benchmark in which only components supported by the specific image are marked.

## Benchmark design

- 13 food classes.
- 20 de-duplicated and human-screened images per class.
- 260 images total.
- 364 image judgments in total: 260 from Annotator A and 104 validation/test judgments from Annotator B.
- Sealed after semantic quality screening and before ingredient annotation: 12 train, 4 validation, and 4 test images per class.
- Independent evaluation annotation followed by explicit adjudication.
- The test split must never be opened for prompt design, alias editing, threshold selection, or qualitative cherry-picking.

## Semantic quality screen

Before split assignment, accept only one assessable food photograph with a clear primary subject. A mixed meal on one plate is valid. Reject the candidate and let the interface supply a same-class replacement when it contains:

- a collage or multi-panel montage;
- a heavy cartoon, text, or graphic overlay;
- multiple unrelated dishes with no primary subject;
- the wrong source class;
- no assessable food;
- extreme blur, obstruction, or other unreadable content;
- visible personal or sensitive information.

Quality screening shows the expected source class so class mismatches can be detected. Ingredient annotation hides source class and split. Quality decisions must not include ingredient judgments.

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

Annotator A labels all 260 images. Annotator B independently labels only the 104 validation and test images. They must not inspect each other's labels before the secondary sheet is complete. If only one human annotator is available, perform the secondary pass after a washout period and disclose this limitation; it is not equivalent to two independent annotators.

The 156 training rows are single-annotator labels. The 104 model-selection and final-evaluation rows are double-annotated and adjudicated.

The notebook interface hides the directory class during annotation to reduce recipe-prior bias. Do not inspect the manifest's `food_class` column while labeling.

## Adjudication

The pipeline automatically accepts exact agreement on the 104 overlapping images. Every evaluation-set disagreement appears in `adjudication_queue.csv`; a human must fill `resolved_visible_ingredients`, `resolved_uncertain_ingredients`, and `resolution_notes`. The evaluation notebook refuses to continue while any disagreement is unresolved.
