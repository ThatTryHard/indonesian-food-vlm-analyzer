from pathlib import Path

from src.ontology import IngredientOntology

ONTOLOGY_PATH = Path(__file__).parents[1] / "data" / "ontology" / "visible_ingredients.json"


def test_phrase_extraction_preserves_multiword_labels():
    ontology = IngredientOntology.from_json(ONTOLOGY_PATH)
    assert len(ontology.ids) == 43
    assert ontology.extract_from_text("Nasi with coconut milk sauce and fried shallots") == [
        "rice",
        "fried_shallot",
        "coconut_sauce",
    ]


def test_recipe_modifiers_are_not_labels():
    ontology = IngredientOntology.from_json(ONTOLOGY_PATH)
    labels = ontology.extract_from_text("freshly coarsely chopped black boneless pieces")
    assert labels == []


def test_unknown_annotation_label_fails():
    ontology = IngredientOntology.from_json(ONTOLOGY_PATH)
    try:
        ontology.parse_annotation_cell("rice|garlic", strict=True)
    except ValueError as error:
        assert "garlic" in str(error)
    else:
        raise AssertionError("Unknown label should fail")
