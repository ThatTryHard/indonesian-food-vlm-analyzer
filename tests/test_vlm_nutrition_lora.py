from pathlib import Path

from src.lora import assistant_only_labels
from src.nutrition import validate_per_100g_nutrition
from src.ontology import IngredientOntology
from src.vlm import food_name_is_correct, parse_vlm_output

ONTOLOGY = IngredientOntology.from_json(Path(__file__).parents[1] / "data" / "ontology" / "visible_ingredients.json")


def test_vlm_parser_accepts_exact_valid_schema():
    parsed = parse_vlm_output(
        '{"food_name":"nasi goreng","visible_ingredients":["rice","egg"],'
        '"uncertain_ingredients":["dark_sauce"],"abstain":false,'
        '"reason":"Rice and egg are visibly distinct; the glaze is ambiguous."}',
        ONTOLOGY,
    )
    assert parsed.parse_ok
    assert parsed.visible_ingredients == ("rice", "egg")
    assert parsed.uncertain_ingredients == ("dark_sauce",)


def test_vlm_parser_rejects_unknown_labels_without_aliasing():
    parsed = parse_vlm_output(
        '{"food_name":"nasi goreng","visible_ingredients":["rice","garlic"],'
        '"uncertain_ingredients":[],"abstain":false,"reason":"rice is visible"}',
        ONTOLOGY,
    )
    assert parsed.visible_ingredients == ()
    assert parsed.unknown_labels == ("garlic",)
    assert not parsed.parse_ok
    assert parsed.parse_error == "unknown_ingredient_ids"
    assert food_name_is_correct("Nasi Goreng", "fried rice")


def test_impossible_pizza_nutrition_is_rejected():
    result = validate_per_100g_nutrition(1500, 50, 20, 30)
    assert not result.valid
    assert "energy_exceeds_physical_bound" in result.errors
    assert "atwater_energy_mismatch" in result.errors


def test_lora_masks_prompt_and_padding():
    labels = assistant_only_labels(
        [10, 11, 20, 21, 0, 0],
        assistant_start_index=2,
        attention_mask=[1, 1, 1, 1, 1, 0],
    )
    assert labels == [-100, -100, 20, 21, 0, -100]


def test_non_finite_nutrition_is_rejected():
    result = validate_per_100g_nutrition(float("nan"), 20, 10, 5)
    assert not result.valid
    assert "non_finite_value" in result.errors


def test_vlm_parser_requires_exact_schema_and_types():
    parsed = parse_vlm_output(
        '{"food_name":"sate","visible_ingredients":"chicken",'
        '"uncertain_ingredients":[],"abstain":false,"reason":"visible skewer"}',
        ONTOLOGY,
    )
    assert not parsed.parse_ok
    assert parsed.parse_error == "ingredient_fields_not_string_lists"


def test_vlm_parser_does_not_silently_strip_markdown_or_duplicate_keys():
    fenced = parse_vlm_output(
        '```json\n{"food_name":"sate","visible_ingredients":[],"uncertain_ingredients":[],'
        '"abstain":false,"reason":"No frozen component is clear."}\n```',
        ONTOLOGY,
    )
    duplicate = parse_vlm_output(
        '{"food_name":"sate","food_name":"satay","visible_ingredients":[],'
        '"uncertain_ingredients":[],"abstain":false,"reason":"Nothing is clear."}',
        ONTOLOGY,
    )
    assert fenced.parse_error == "non_json_envelope"
    assert duplicate.parse_error == "duplicate_key:food_name"
