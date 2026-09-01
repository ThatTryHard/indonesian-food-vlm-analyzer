"""Canonical weak labels and grouped iterative splits for recipe pretraining.

Recipe strings map onto the fixed phrase ontology. Related recipes are grouped
before iterative multi-label splitting to reduce contamination.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ontology import IngredientOntology, normalize_text


def parse_ingredient_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [part.strip() for part in re.split(r"[;|]", text) if part.strip()]


def canonical_recipe_labels(value: object, ontology: IngredientOntology) -> list[str]:
    raw_items = parse_ingredient_list(value)
    return ontology.extract_from_text(" ; ".join(raw_items))


def build_recipe_image_index(image_root: str | Path) -> dict[str, Path | None]:
    """Index recursive image names; ambiguous basenames deliberately resolve to None."""
    root = Path(image_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    index: dict[str, Path | None] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        for key in {path.name.lower(), path.stem.lower()}:
            previous = index.get(key)
            if previous is None and key in index:
                continue
            if previous is not None and previous != path:
                index[key] = None
            else:
                index[key] = path
    return index


def resolve_recipe_image(
    image_name: object,
    image_root: str | Path,
    image_index: dict[str, Path | None] | None = None,
) -> Path | None:
    root = Path(image_root)
    name = str(image_name).strip()
    clean_name = name.replace("\\", "/")
    direct = root / Path(clean_name)
    if direct.is_file():
        return direct
    image_index = image_index if image_index is not None else build_recipe_image_index(root)
    keys = [Path(clean_name).name.lower(), Path(clean_name).stem.lower()]
    return next((image_index[key] for key in keys if key in image_index and image_index[key] is not None), None)


def prepare_recipe_dataframe(
    csv_path: str | Path,
    image_root: str | Path,
    ontology: IngredientOntology,
    ingredient_column: str = "Ingredients",
    image_column: str = "Image_Name",
    title_column: str = "Title",
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {ingredient_column, image_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Recipe CSV missing columns: {sorted(missing)}")
    frame = frame.copy()
    image_index = build_recipe_image_index(image_root)
    frame["labels"] = frame[ingredient_column].map(lambda value: canonical_recipe_labels(value, ontology))
    frame["image_path"] = frame[image_column].map(
        lambda value: resolve_recipe_image(value, image_root, image_index=image_index)
    )
    title_values = frame[title_column] if title_column in frame else frame[image_column]
    group_ids = []
    for title_value, image_value in zip(title_values, frame[image_column], strict=True):
        value = image_value if pd.isna(title_value) or not str(title_value).strip() else title_value
        normalized = re.sub(r"[^a-z0-9]+", "-", normalize_text(value)).strip("-")
        group_ids.append(normalized or normalize_text(image_value))
    frame["group_id"] = group_ids
    frame = frame[frame["image_path"].notna() & frame["labels"].map(bool)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No recipe rows have both a uniquely resolved image and a canonical label")
    return frame


def labels_to_matrix(labels: pd.Series, ontology: IngredientOntology) -> np.ndarray:
    matrix = np.zeros((len(labels), len(ontology.ids)), dtype=np.int8)
    for row_index, row_labels in enumerate(labels):
        for label in row_labels:
            matrix[row_index, ontology.ids.index(label)] = 1
    return matrix


def merge_group_constraints(
    frame: pd.DataFrame,
    constraint_columns: tuple[str, ...] = ("group_id", "duplicate_group"),
    output_column: str = "group_id",
) -> pd.DataFrame:
    """Create connected groups satisfying every title/duplicate constraint."""
    missing = set(constraint_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Grouping constraints missing columns: {sorted(missing)}")
    result = frame.copy().reset_index(drop=True)
    parent = list(range(len(result)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for column in constraint_columns:
        first_by_value: dict[str, int] = {}
        for index, raw_value in enumerate(result[column]):
            value = str(raw_value).strip()
            if not value:
                continue
            if value in first_by_value:
                union(first_by_value[value], index)
            else:
                first_by_value[value] = index

    members: dict[int, list[int]] = {}
    for index in range(len(result)):
        members.setdefault(find(index), []).append(index)
    assigned: dict[int, str] = {}
    for indices in members.values():
        tokens = sorted(
            {
                f"{column}={str(result.at[index, column]).strip()}"
                for index in indices
                for column in constraint_columns
                if str(result.at[index, column]).strip()
            }
        )
        if not tokens:
            tokens = [f"singleton_row={indices[0]}"]
        digest = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()[:16]
        for index in indices:
            assigned[index] = f"connected-{digest}"
    result[output_column] = [assigned[index] for index in result.index]
    return result


def grouped_iterative_split(
    frame: pd.DataFrame,
    ontology: IngredientOntology,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.DataFrame:
    """Assign groups with iterative multi-label stratification.

    The function fails explicitly if iterative-stratification is unavailable; silently
    falling back would make a supposedly reproducible split environment-dependent.
    """
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to one")
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError("Install iterative-stratification==0.1.9 before creating splits") from exc

    group_rows = []
    for group_id, group in frame.groupby("group_id", sort=True):
        labels = sorted(set().union(*group["labels"]), key=ontology.ids.index)
        group_rows.append({"group_id": group_id, "labels": labels})
    groups = pd.DataFrame(group_rows)
    matrix = labels_to_matrix(groups["labels"], ontology)

    outer = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=validation_fraction + test_fraction, random_state=seed
    )
    placeholder_features = np.zeros((len(groups), 1), dtype=np.int8)
    train_index, holdout_index = next(outer.split(placeholder_features, matrix))
    holdout = groups.iloc[holdout_index].reset_index(drop=True)
    holdout_matrix = matrix[holdout_index]
    relative_test = test_fraction / (validation_fraction + test_fraction)
    inner = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed + 1)
    validation_index, test_index = next(inner.split(np.zeros((len(holdout), 1), dtype=np.int8), holdout_matrix))

    split_by_group = {group_id: "train" for group_id in groups.iloc[train_index]["group_id"]}
    split_by_group.update({group_id: "validation" for group_id in holdout.iloc[validation_index]["group_id"]})
    split_by_group.update({group_id: "test" for group_id in holdout.iloc[test_index]["group_id"]})
    result = frame.copy()
    result["split"] = result["group_id"].map(split_by_group)
    if result["split"].isna().any():
        raise RuntimeError("Some recipe groups were not assigned")
    return result
