"""Frozen zero-shot VLM prompt, schema parser, and ontology mapping.

The prompt asks only for visible components from the frozen canonical ontology.
Unknown outputs are retained as errors rather than retroactively aliased on test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .ontology import IngredientOntology, normalize_text

PROMPT_VERSION = "visible-zero-shot-v1"


def build_visible_prompt(ontology: IngredientOntology) -> str:
    allowed = ", ".join(ontology.ids)
    return f"""You are evaluating visible food components in one image.
Return raw JSON only, with exactly this schema:
{{
  "food_name": "short name or unknown",
  "visible_ingredients": ["canonical_id"],
  "uncertain_ingredients": ["canonical_id"],
  "abstain": false,
  "reason": "one short evidence-based sentence"
}}

Allowed canonical IDs: {allowed}

Rules:
- Mark visible_ingredients only when the specific image provides visual evidence.
- Do not infer hidden spices, oil, salt, garlic, or recipe ingredients from dish knowledge.
- Put plausible but visually ambiguous allowed labels in uncertain_ingredients.
- Never invent an ID outside the allowed list.
- If the image is not assessable food, set abstain=true and both lists empty.
- Do not estimate nutrition.
Prompt version: {PROMPT_VERSION}
"""


@dataclass(frozen=True)
class ParsedVLMOutput:
    food_name: str
    visible_ingredients: tuple[str, ...]
    uncertain_ingredients: tuple[str, ...]
    unknown_labels: tuple[str, ...]
    abstain: bool
    reason: str
    parse_ok: bool
    parse_error: str | None


def parse_vlm_output(text: object, ontology: IngredientOntology) -> ParsedVLMOutput:
    def failure(error: str, unknown: tuple[str, ...] = ()) -> ParsedVLMOutput:
        return ParsedVLMOutput("unknown", (), (), unknown, True, "", False, error)

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_key:{key}")
            result[key] = value
        return result

    raw = str(text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return failure("no_json_object")
    if raw[:start].strip() or raw[end + 1 :].strip():
        return failure("non_json_envelope")
    try:
        data = json.loads(raw[start : end + 1], object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        return failure(f"json_error:{exc.msg}")
    except ValueError as exc:
        return failure(str(exc))

    if not isinstance(data, dict):
        return failure("root_not_object")
    expected_keys = {
        "food_name",
        "visible_ingredients",
        "uncertain_ingredients",
        "abstain",
        "reason",
    }
    if set(data) != expected_keys:
        missing = sorted(expected_keys.difference(data))
        extra = sorted(set(data).difference(expected_keys))
        return failure(f"schema_keys:missing={missing},extra={extra}")
    if not isinstance(data["food_name"], str) or not data["food_name"].strip():
        return failure("food_name_not_nonempty_string")
    if not isinstance(data["reason"], str) or not data["reason"].strip():
        return failure("reason_not_nonempty_string")
    if type(data["abstain"]) is not bool:
        return failure("abstain_not_boolean")

    def list_field(name: str) -> list[str] | None:
        value = data[name]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return None
        return [normalize_text(item).replace(" ", "_") for item in value]

    visible_raw, uncertain_raw = list_field("visible_ingredients"), list_field("uncertain_ingredients")
    if visible_raw is None or uncertain_raw is None:
        return failure("ingredient_fields_not_string_lists")
    if len(visible_raw) != len(set(visible_raw)) or len(uncertain_raw) != len(set(uncertain_raw)):
        return failure("duplicate_ingredient_ids")
    unknown = sorted(set(visible_raw + uncertain_raw).difference(ontology.ids))
    if unknown:
        return failure("unknown_ingredient_ids", tuple(unknown))
    overlap = sorted(set(visible_raw).intersection(uncertain_raw))
    if overlap:
        return failure("visible_uncertain_overlap")
    visible = tuple(label for label in ontology.ids if label in visible_raw)
    uncertain = tuple(label for label in ontology.ids if label in uncertain_raw)
    abstain = data["abstain"]
    if abstain and (visible or uncertain):
        return failure("abstain_with_labels")
    return ParsedVLMOutput(
        food_name=data["food_name"].strip(),
        visible_ingredients=visible,
        uncertain_ingredients=uncertain,
        unknown_labels=(),
        abstain=abstain,
        reason=data["reason"].strip(),
        parse_ok=True,
        parse_error=None,
    )


FOOD_NAME_ALIASES = {
    "ayam goreng": {"ayam goreng", "fried chicken"},
    "burger": {"burger", "hamburger"},
    "french fries": {"french fries", "fries"},
    "gado gado": {"gado gado", "gado-gado"},
    "ikan goreng": {"ikan goreng", "fried fish"},
    "mie goreng": {"mie goreng", "mi goreng", "fried noodles"},
    "nasi goreng": {"nasi goreng", "fried rice"},
    "nasi padang": {"nasi padang"},
    "pizza": {"pizza"},
    "rawon": {"rawon"},
    "rendang": {"rendang"},
    "sate": {"sate", "satay"},
    "soto": {"soto", "soto ayam"},
}


def food_name_is_correct(true_class: str, predicted_name: str) -> bool:
    true_normalized = normalize_text(true_class)
    predicted_normalized = normalize_text(predicted_name)
    aliases = FOOD_NAME_ALIASES.get(true_normalized, {true_normalized})
    return predicted_normalized in {normalize_text(alias) for alias in aliases}
