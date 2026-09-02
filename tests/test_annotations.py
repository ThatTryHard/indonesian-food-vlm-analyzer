from pathlib import Path

import pandas as pd

from src.annotations import (
    annotation_agreement,
    build_adjudication_queue,
    combine_primary_and_adjudicated_evaluation,
    create_annotation_sheet,
    finalize_adjudication,
    validate_annotation_sheet,
)
from src.ontology import IngredientOntology

ONTOLOGY = IngredientOntology.from_json(Path(__file__).parents[1] / "data" / "ontology" / "visible_ingredients.json")


def _manifest():
    return pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "relative_path": ["A/a.jpg", "B/b.jpg"],
            "food_class": ["A", "B"],
            "split": ["train", "test"],
        }
    )


def test_annotation_disagreement_requires_resolution():
    manifest = _manifest()
    first = create_annotation_sheet(manifest, "alice")
    second = create_annotation_sheet(manifest, "bob")
    first.loc[:, "visible_ingredients"] = ["rice", "egg"]
    second.loc[:, "visible_ingredients"] = ["rice", "chicken"]
    first = validate_annotation_sheet(first, manifest, ONTOLOGY)
    second = validate_annotation_sheet(second, manifest, ONTOLOGY)
    agreement = annotation_agreement(first, second, ONTOLOGY)
    assert agreement["exact_set_agreement"] == 0.5
    queue = build_adjudication_queue(first, second, manifest)
    assert (queue["status"] == "needs_adjudication").sum() == 1
    try:
        finalize_adjudication(queue, ONTOLOGY)
    except ValueError as error:
        assert "1 disagreements" in str(error)
    else:
        raise AssertionError("Unresolved disagreement should block finalization")


def test_visible_and_uncertain_overlap_fails():
    manifest = _manifest()
    sheet = create_annotation_sheet(manifest, "alice")
    sheet.loc[:, "visible_ingredients"] = ["rice", "egg"]
    sheet.loc[:, "uncertain_ingredients"] = ["rice", ""]
    try:
        validate_annotation_sheet(sheet, manifest, ONTOLOGY)
    except ValueError as error:
        assert "visible and uncertain" in str(error)
    else:
        raise AssertionError("Overlap should fail")


def test_explicit_all_negative_annotation_is_valid():
    manifest = _manifest()
    sheet = create_annotation_sheet(manifest, "alice")
    sheet.loc[0, "no_visible_ontology_label"] = True
    sheet.loc[1, "visible_ingredients"] = "egg"
    validated = validate_annotation_sheet(sheet, manifest, ONTOLOGY)
    assert bool(validated.loc[0, "no_visible_ontology_label"])
    assert validated.loc[0, "visible_ingredients"] == ""


def test_adjudicated_flags_are_validated():
    manifest = _manifest()
    first = create_annotation_sheet(manifest, "alice")
    second = create_annotation_sheet(manifest, "bob")
    first.loc[:, "visible_ingredients"] = ["rice", "egg"]
    second.loc[:, "visible_ingredients"] = ["rice", "chicken"]
    first = validate_annotation_sheet(first, manifest, ONTOLOGY)
    second = validate_annotation_sheet(second, manifest, ONTOLOGY)
    queue = build_adjudication_queue(first, second, manifest)
    disagreement = queue["status"].eq("needs_adjudication")
    queue.loc[disagreement, "resolved_visible_ingredients"] = ""
    queue.loc[disagreement, "resolved_no_visible_ontology_label"] = True
    queue.loc[disagreement, "resolution_notes"] = "No frozen label is visually supported."
    final = finalize_adjudication(queue, ONTOLOGY)
    assert bool(final.loc[final["sample_id"].eq("b"), "no_visible_ontology_label"].iloc[0])


def test_secondary_sheet_can_cover_evaluation_subset_only():
    manifest = _manifest()
    primary = create_annotation_sheet(manifest, "alice")
    primary.loc[:, "visible_ingredients"] = ["rice", "egg"]
    evaluation_manifest = manifest[manifest["split"].eq("test")].copy()
    secondary = create_annotation_sheet(evaluation_manifest, "bob")
    secondary.loc[:, "visible_ingredients"] = ["egg"]

    validated_primary = validate_annotation_sheet(primary, manifest, ONTOLOGY)
    validated_secondary = validate_annotation_sheet(secondary, evaluation_manifest, ONTOLOGY)

    assert len(validated_primary) == 2
    assert len(validated_secondary) == 1
    assert validated_secondary.iloc[0]["split"] == "test"


def test_primary_train_combines_with_adjudicated_evaluation():
    manifest = _manifest()
    primary = create_annotation_sheet(manifest, "alice")
    primary.loc[:, "visible_ingredients"] = ["rice", "egg"]
    primary = validate_annotation_sheet(primary, manifest, ONTOLOGY)
    adjudicated = pd.DataFrame(
        {
            "sample_id": ["b"],
            "relative_path": ["B/b.jpg"],
            "food_class": ["B"],
            "split": ["test"],
            "visible_ingredients": ["chicken"],
            "uncertain_ingredients": [""],
            "unreadable": [False],
            "non_food": [False],
            "no_visible_ontology_label": [False],
            "resolution_notes": ["Adjudicated disagreement."],
        }
    )

    final = combine_primary_and_adjudicated_evaluation(
        primary,
        adjudicated,
        manifest,
        secondary_splits=["test"],
    )

    indexed = final.set_index("sample_id")
    assert len(final) == 2
    assert indexed.loc["a", "annotation_source"] == "primary_annotator_only"
    assert indexed.loc["b", "annotation_source"] == "double_annotated_adjudicated"
    assert indexed.loc["b", "visible_ingredients"] == "chicken"
