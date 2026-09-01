from pathlib import Path

import pandas as pd
from PIL import Image

from src.ontology import IngredientOntology
from src.recipe_labels import (
    build_recipe_image_index,
    grouped_iterative_split,
    merge_group_constraints,
    resolve_recipe_image,
)

ONTOLOGY = IngredientOntology.from_json(Path(__file__).parents[1] / "data/ontology/visible_ingredients.json")


def test_group_constraints_form_connected_components():
    frame = pd.DataFrame(
        {
            "group_id": ["same-title", "same-title", "other-title", "isolated"],
            "duplicate_group": ["dup-a", "dup-b", "dup-b", "dup-c"],
        }
    )
    grouped = merge_group_constraints(frame)
    assert grouped.loc[0, "group_id"] == grouped.loc[1, "group_id"] == grouped.loc[2, "group_id"]
    assert grouped.loc[3, "group_id"] != grouped.loc[0, "group_id"]


def test_grouped_iterative_split_is_deterministic_and_group_safe():
    frame = pd.DataFrame(
        {
            "group_id": [f"recipe-{index // 2:02d}" for index in range(40)],
            "labels": [["rice", "egg"] if (index // 2) % 2 == 0 else ["noodles", "chicken"] for index in range(40)],
        }
    )
    first = grouped_iterative_split(frame, ONTOLOGY, seed=42)
    second = grouped_iterative_split(frame, ONTOLOGY, seed=42)
    assert first["split"].tolist() == second["split"].tolist()
    assert first.groupby("group_id")["split"].nunique().eq(1).all()
    assert set(first["split"]) == {"train", "validation", "test"}


def test_recursive_recipe_image_resolution_rejects_ambiguous_basename(tmp_path):
    for folder_name in ["a", "b"]:
        folder = tmp_path / folder_name
        folder.mkdir()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(folder / "same.jpg")
    unique_folder = tmp_path / "c"
    unique_folder.mkdir()
    Image.new("RGB", (4, 4), (30, 20, 10)).save(unique_folder / "unique.png")
    index = build_recipe_image_index(tmp_path)
    assert resolve_recipe_image("same.jpg", tmp_path, index) is None
    assert resolve_recipe_image("unique", tmp_path, index) == unique_folder / "unique.png"
