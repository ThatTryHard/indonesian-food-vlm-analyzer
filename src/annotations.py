"""Annotation validation, overlap agreement, and explicit adjudication.

The primary annotator covers all benchmark rows. The secondary annotator covers
validation and test independently. Their evaluation-set disagreements remain
unresolved until the adjudication file is completed by a human.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ontology import IngredientOntology

ANNOTATION_COLUMNS = [
    "sample_id",
    "annotator_id",
    "visible_ingredients",
    "uncertain_ingredients",
    "unreadable",
    "non_food",
    "no_visible_ontology_label",
    "notes",
]


def parse_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def create_annotation_sheet(manifest: pd.DataFrame, annotator_id: str) -> pd.DataFrame:
    if not annotator_id.strip():
        raise ValueError("annotator_id cannot be blank")
    sheet = manifest[["sample_id", "relative_path", "food_class", "split"]].copy()
    sheet["annotator_id"] = annotator_id.strip()
    sheet["visible_ingredients"] = ""
    sheet["uncertain_ingredients"] = ""
    sheet["unreadable"] = False
    sheet["non_food"] = False
    sheet["no_visible_ontology_label"] = False
    sheet["notes"] = ""
    return sheet


def validate_annotation_sheet(
    sheet: pd.DataFrame,
    manifest: pd.DataFrame,
    ontology: IngredientOntology,
    require_complete: bool = True,
) -> pd.DataFrame:
    missing = set(ANNOTATION_COLUMNS).difference(sheet.columns)
    if missing:
        raise ValueError(f"Annotation sheet is missing columns: {sorted(missing)}")
    if sheet["sample_id"].duplicated().any():
        raise ValueError("Annotation sheet contains duplicate sample_id rows")
    expected = set(manifest["sample_id"])
    observed = set(sheet["sample_id"])
    if observed != expected:
        raise ValueError(
            f"Annotation sample mismatch: missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    annotators = set(sheet["annotator_id"].astype(str).str.strip())
    if "" in annotators or len(annotators) != 1:
        raise ValueError("Each sheet must contain exactly one non-empty annotator_id")

    normalized = sheet.copy()
    serialized_visible: list[str] = []
    serialized_uncertain: list[str] = []
    for row in normalized.itertuples(index=False):
        visible = ontology.parse_annotation_cell(row.visible_ingredients, strict=True)
        uncertain = ontology.parse_annotation_cell(row.uncertain_ingredients, strict=True)
        overlap = set(visible).intersection(uncertain)
        if overlap:
            raise ValueError(f"{row.sample_id}: labels cannot be visible and uncertain: {sorted(overlap)}")
        unreadable = parse_boolean(row.unreadable)
        non_food = parse_boolean(row.non_food)
        no_visible_label = parse_boolean(row.no_visible_ontology_label)
        if sum([unreadable, non_food, no_visible_label]) > 1:
            raise ValueError(
                f"{row.sample_id}: unreadable, non_food, and no_visible_ontology_label are mutually exclusive"
            )
        if (unreadable or non_food or no_visible_label) and (visible or uncertain):
            raise ValueError(f"{row.sample_id}: flagged images cannot contain ingredient labels")
        if (
            require_complete
            and not visible
            and not uncertain
            and not unreadable
            and not non_food
            and not no_visible_label
        ):
            raise ValueError(f"{row.sample_id}: blank annotation; label it or set an explicit completion flag")
        serialized_visible.append(ontology.serialize(visible))
        serialized_uncertain.append(ontology.serialize(uncertain))
    normalized["visible_ingredients"] = serialized_visible
    normalized["uncertain_ingredients"] = serialized_uncertain
    normalized["unreadable"] = normalized["unreadable"].map(parse_boolean)
    normalized["non_food"] = normalized["non_food"].map(parse_boolean)
    normalized["no_visible_ontology_label"] = normalized["no_visible_ontology_label"].map(parse_boolean)
    return normalized


def _label_matrix(sheet: pd.DataFrame, ontology: IngredientOntology) -> np.ndarray:
    matrix = np.zeros((len(sheet), len(ontology.ids)), dtype=np.int8)
    for row_index, value in enumerate(sheet["visible_ingredients"]):
        for label in ontology.parse_annotation_cell(value):
            matrix[row_index, ontology.ids.index(label)] = 1
    return matrix


def annotation_agreement(
    first: pd.DataFrame,
    second: pd.DataFrame,
    ontology: IngredientOntology,
) -> dict[str, object]:
    # Agreement metrics are only needed in Notebook 02. Keeping this import local
    # lets the annotation interface start without loading the full SciPy/sklearn stack.
    from sklearn.metrics import cohen_kappa_score

    first = first.sort_values("sample_id").reset_index(drop=True)
    second = second.sort_values("sample_id").reset_index(drop=True)
    if not first["sample_id"].equals(second["sample_id"]):
        raise ValueError("Annotation sheets must contain identical sample IDs")
    left, right = _label_matrix(first, ontology), _label_matrix(second, ontology)
    intersections = np.logical_and(left, right).sum(axis=1)
    unions = np.logical_or(left, right).sum(axis=1)
    jaccard = np.ones(len(unions), dtype=float)
    np.divide(intersections, unions, out=jaccard, where=unions != 0)
    label_kappa: dict[str, float | None] = {}
    for index, label in enumerate(ontology.ids):
        if len(np.unique(np.concatenate([left[:, index], right[:, index]]))) < 2:
            label_kappa[label] = None
        else:
            label_kappa[label] = float(cohen_kappa_score(left[:, index], right[:, index]))
    return {
        "sample_count": len(first),
        "mean_sample_jaccard": float(jaccard.mean()),
        "exact_set_agreement": float(np.all(left == right, axis=1).mean()),
        "micro_label_agreement": float((left == right).mean()),
        "unreadable_flag_agreement": float(
            np.mean(
                first["unreadable"].map(parse_boolean).to_numpy() == second["unreadable"].map(parse_boolean).to_numpy()
            )
        ),
        "non_food_flag_agreement": float(
            np.mean(first["non_food"].map(parse_boolean).to_numpy() == second["non_food"].map(parse_boolean).to_numpy())
        ),
        "no_visible_ontology_label_flag_agreement": float(
            np.mean(
                first["no_visible_ontology_label"].map(parse_boolean).to_numpy()
                == second["no_visible_ontology_label"].map(parse_boolean).to_numpy()
            )
        ),
        "per_label_kappa": label_kappa,
    }


def build_adjudication_queue(
    first: pd.DataFrame,
    second: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    comparison_columns = [
        "visible_ingredients",
        "uncertain_ingredients",
        "unreadable",
        "non_food",
        "no_visible_ontology_label",
        "notes",
    ]
    left = first[["sample_id", *comparison_columns]].rename(
        columns={column: f"{column}_a" for column in comparison_columns}
    )
    right = second[["sample_id", *comparison_columns]].rename(
        columns={column: f"{column}_b" for column in comparison_columns}
    )
    queue = (
        manifest[["sample_id", "relative_path", "food_class", "split"]]
        .merge(left, on="sample_id")
        .merge(right, on="sample_id")
    )
    agrees = (
        queue["visible_ingredients_a"].eq(queue["visible_ingredients_b"])
        & queue["uncertain_ingredients_a"].eq(queue["uncertain_ingredients_b"])
        & queue["unreadable_a"].eq(queue["unreadable_b"])
        & queue["non_food_a"].eq(queue["non_food_b"])
        & queue["no_visible_ontology_label_a"].eq(queue["no_visible_ontology_label_b"])
    )
    queue["status"] = np.where(agrees, "agreement", "needs_adjudication")
    queue["resolved_visible_ingredients"] = np.where(agrees, queue["visible_ingredients_a"], "")
    queue["resolved_uncertain_ingredients"] = np.where(agrees, queue["uncertain_ingredients_a"], "")
    queue["resolved_unreadable"] = np.where(agrees, queue["unreadable_a"], False)
    queue["resolved_non_food"] = np.where(agrees, queue["non_food_a"], False)
    queue["resolved_no_visible_ontology_label"] = np.where(agrees, queue["no_visible_ontology_label_a"], False)
    queue["resolution_notes"] = ""
    return queue


def finalize_adjudication(queue: pd.DataFrame, ontology: IngredientOntology) -> pd.DataFrame:
    unresolved = queue[
        queue["status"].eq("needs_adjudication") & queue["resolution_notes"].fillna("").astype(str).str.strip().eq("")
    ]
    if not unresolved.empty:
        raise ValueError(f"{len(unresolved)} disagreements still require human adjudication")
    final = queue[["sample_id", "relative_path", "food_class", "split"]].copy()
    visible_values: list[str] = []
    uncertain_values: list[str] = []
    unreadable_values: list[bool] = []
    non_food_values: list[bool] = []
    no_visible_label_values: list[bool] = []
    for row in queue.itertuples(index=False):
        visible = ontology.parse_annotation_cell(row.resolved_visible_ingredients)
        uncertain = ontology.parse_annotation_cell(row.resolved_uncertain_ingredients)
        if set(visible).intersection(uncertain):
            raise ValueError(f"{row.sample_id}: adjudicated visible/uncertain labels overlap")
        unreadable = parse_boolean(row.resolved_unreadable)
        non_food = parse_boolean(row.resolved_non_food)
        no_visible_label = parse_boolean(row.resolved_no_visible_ontology_label)
        if sum([unreadable, non_food, no_visible_label]) > 1:
            raise ValueError(f"{row.sample_id}: adjudicated completion flags conflict")
        if (unreadable or non_food or no_visible_label) and (visible or uncertain):
            raise ValueError(f"{row.sample_id}: adjudicated flag conflicts with ingredient labels")
        if not visible and not uncertain and not unreadable and not non_food and not no_visible_label:
            raise ValueError(f"{row.sample_id}: adjudicated row is blank")
        visible_values.append(ontology.serialize(visible))
        uncertain_values.append(ontology.serialize(uncertain))
        unreadable_values.append(unreadable)
        non_food_values.append(non_food)
        no_visible_label_values.append(no_visible_label)
    final["visible_ingredients"] = visible_values
    final["uncertain_ingredients"] = uncertain_values
    final["unreadable"] = unreadable_values
    final["non_food"] = non_food_values
    final["no_visible_ontology_label"] = no_visible_label_values
    final["resolution_notes"] = queue["resolution_notes"].fillna("")
    return final


def combine_primary_and_adjudicated_evaluation(
    primary: pd.DataFrame,
    adjudicated_evaluation: pd.DataFrame,
    manifest: pd.DataFrame,
    secondary_splits: list[str],
) -> pd.DataFrame:
    """Combine primary-only training labels with adjudicated evaluation labels."""
    evaluation_ids = set(manifest.loc[manifest["split"].isin(secondary_splits), "sample_id"])
    training_ids = set(manifest.loc[~manifest["split"].isin(secondary_splits), "sample_id"])
    primary_training = primary[primary["sample_id"].isin(training_ids)].copy()
    if set(primary_training["sample_id"]) != training_ids:
        raise ValueError("Primary annotations do not cover every training sample")
    if set(adjudicated_evaluation["sample_id"]) != evaluation_ids:
        raise ValueError("Adjudicated annotations do not cover every evaluation sample")
    if training_ids.intersection(evaluation_ids):
        raise ValueError("Training and evaluation sample IDs overlap")

    base_columns = [
        "sample_id",
        "relative_path",
        "food_class",
        "split",
        "visible_ingredients",
        "uncertain_ingredients",
        "unreadable",
        "non_food",
        "no_visible_ontology_label",
    ]
    primary_required = {*base_columns, "notes"}
    evaluation_required = {*base_columns, "resolution_notes"}
    if missing := primary_required.difference(primary_training.columns):
        raise ValueError(f"Primary training annotations missing columns: {sorted(missing)}")
    if missing := evaluation_required.difference(adjudicated_evaluation.columns):
        raise ValueError(f"Adjudicated evaluation annotations missing columns: {sorted(missing)}")

    primary_final = primary_training[[*base_columns, "notes"]].rename(columns={"notes": "resolution_notes"})
    primary_final["annotation_source"] = "primary_annotator_only"
    evaluation_final = adjudicated_evaluation[[*base_columns, "resolution_notes"]].copy()
    evaluation_final["annotation_source"] = "double_annotated_adjudicated"
    final = pd.concat([primary_final, evaluation_final], ignore_index=True)
    expected_ids = set(manifest["sample_id"])
    if set(final["sample_id"]) != expected_ids or len(final) != len(manifest):
        raise ValueError("Combined annotations do not match the sealed manifest")
    return final.sort_values("sample_id").reset_index(drop=True)


def annotation_progress(sheet: pd.DataFrame) -> dict[str, int]:
    completed = (
        sheet["visible_ingredients"].fillna("").astype(str).str.strip().ne("")
        | sheet["uncertain_ingredients"].fillna("").astype(str).str.strip().ne("")
        | sheet["unreadable"].fillna(False).map(parse_boolean)
        | sheet["non_food"].fillna(False).map(parse_boolean)
        | sheet["no_visible_ontology_label"].fillna(False).map(parse_boolean)
    )
    return {"completed": int(completed.sum()), "remaining": int((~completed).sum()), "total": len(sheet)}
