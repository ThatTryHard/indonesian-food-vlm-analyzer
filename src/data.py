"""Dataset inventory, corruption checks, duplicate grouping, and sampling.

Candidates come from validated, deduplicated images and receive human semantic
screening. Train, validation, and test membership is then sealed before ingredient
annotation so later modeling choices cannot reshape the test set.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, size: int = 8) -> str:
    """Compute a 64-bit difference hash, robust to small resize/compression changes."""
    grayscale = image.convert("L").resize((size + 1, size))
    values = np.asarray(grayscale, dtype=np.float32)
    bits = values[:, 1:] >= values[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


# Backward-compatible name for callers; the inventory now stores a difference hash.
average_hash = difference_hash


def hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def cross_dataset_overlap_report(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    perceptual_max_distance: int = 2,
) -> pd.DataFrame:
    """Find exact or near image copies that must be quarantined from pretraining."""
    required_candidate = {"image_path", "sha256", "dhash", "status"}
    required_reference = {"sample_id", "sha256", "dhash"}
    if missing := required_candidate.difference(candidates.columns):
        raise ValueError(f"Candidate inventory missing columns: {sorted(missing)}")
    if missing := required_reference.difference(reference.columns):
        raise ValueError(f"Reference manifest missing columns: {sorted(missing)}")
    if not 0 <= perceptual_max_distance <= 2:
        raise ValueError("perceptual_max_distance must be between 0 and 2")

    references = [(str(row.sample_id), str(row.sha256), str(row.dhash)) for row in reference.itertuples(index=False)]
    exact_reference = {sha256: sample_id for sample_id, sha256, _ in references}
    reference_hashes = [(sample_id, dhash) for sample_id, _, dhash in references]
    bit_masks = [0]
    if perceptual_max_distance >= 1:
        bit_masks.extend(1 << bit for bit in range(64))
    if perceptual_max_distance >= 2:
        bit_masks.extend((1 << left) | (1 << right) for left in range(64) for right in range(left + 1, 64))
    near_reference_values = {int(dhash, 16) ^ mask for _, dhash in reference_hashes for mask in bit_masks}

    rows: list[dict[str, object]] = []
    for row in candidates[candidates["status"].eq("ok")].itertuples(index=False):
        sha256 = str(row.sha256)
        dhash = str(row.dhash)
        if sha256 in exact_reference:
            rows.append(
                {
                    "image_path": str(row.image_path),
                    "candidate_sha256": sha256,
                    "reference_sample_id": exact_reference[sha256],
                    "match_type": "exact_sha256",
                    "hash_distance": 0,
                }
            )
            continue
        if int(dhash, 16) not in near_reference_values:
            continue
        matched_sample, distance = min(
            ((sample_id, hash_distance(dhash, reference_hash)) for sample_id, reference_hash in reference_hashes),
            key=lambda pair: pair[1],
        )
        if distance <= perceptual_max_distance:
            rows.append(
                {
                    "image_path": str(row.image_path),
                    "candidate_sha256": sha256,
                    "reference_sample_id": matched_sample,
                    "match_type": "near_difference_hash",
                    "hash_distance": distance,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "image_path",
            "candidate_sha256",
            "reference_sample_id",
            "match_type",
            "hash_distance",
        ],
    )


def discover_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def build_image_inventory(root: str | Path) -> pd.DataFrame:
    root = Path(root).resolve()
    rows: list[dict[str, object]] = []
    for path in discover_images(root):
        row: dict[str, object] = {
            "image_path": str(path.resolve()),
            "relative_path": path.resolve().relative_to(root).as_posix(),
            "food_class": path.parent.name,
            "bytes": path.stat().st_size,
            "status": "ok",
            "error": "",
        }
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                row.update({"width": rgb.width, "height": rgb.height, "dhash": difference_hash(rgb)})
            row["sha256"] = sha256_file(path)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            row.update({"width": np.nan, "height": np.nan, "dhash": "", "sha256": ""})
            row["status"] = "corrupt"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    columns = [
        "image_path",
        "relative_path",
        "food_class",
        "bytes",
        "status",
        "error",
        "width",
        "height",
        "dhash",
        "sha256",
    ]
    return pd.DataFrame(rows, columns=columns)


class _UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def assign_duplicate_groups(inventory: pd.DataFrame, perceptual_max_distance: int = 2) -> pd.DataFrame:
    """Group exact and near-duplicate images globally across all food classes."""
    result = inventory.copy().reset_index(drop=True)
    valid_indices = result.index[result["status"].eq("ok")].tolist()
    union_find = _UnionFind(valid_indices)

    for _, indices in result.loc[valid_indices].groupby("sha256").groups.items():
        indices = list(indices)
        for index in indices[1:]:
            union_find.union(indices[0], index)

    if perceptual_max_distance >= 0:
        if perceptual_max_distance > 2:
            raise ValueError("perceptual_max_distance above 2 is intentionally unsupported")
        indices_by_hash: dict[int, list[int]] = defaultdict(list)
        for index in valid_indices:
            indices_by_hash[int(result.at[index, "dhash"], 16)].append(index)
        bit_masks = [0]
        if perceptual_max_distance >= 1:
            bit_masks.extend(1 << bit for bit in range(64))
        if perceptual_max_distance >= 2:
            bit_masks.extend((1 << left) | (1 << right) for left in range(64) for right in range(left + 1, 64))
        for hash_value, left_indices in indices_by_hash.items():
            for mask in bit_masks:
                neighbor_value = hash_value ^ mask
                if neighbor_value not in indices_by_hash or neighbor_value < hash_value:
                    continue
                right_indices = indices_by_hash[neighbor_value]
                if neighbor_value == hash_value:
                    for offset, left in enumerate(left_indices):
                        for right in left_indices[offset + 1 :]:
                            union_find.union(left, right)
                else:
                    for left in left_indices:
                        for right in right_indices:
                            union_find.union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in valid_indices:
        members[union_find.find(index)].append(index)
    group_id_by_index: dict[int, str] = {}
    for member_indices in members.values():
        representative = min(result.at[index, "sha256"] for index in member_indices)
        group_id = f"dup-{representative[:16]}"
        for index in member_indices:
            group_id_by_index[index] = group_id

    result["duplicate_group"] = [group_id_by_index.get(index, "") for index in result.index]
    valid_group_sizes = result.loc[result["status"].eq("ok"), "duplicate_group"].value_counts()
    result["duplicate_group_size"] = result["duplicate_group"].map(valid_group_sizes).fillna(0).astype(int)
    return result


def _sealed_split_labels(n: int, train_n: int, validation_n: int, test_n: int) -> list[str]:
    if train_n + validation_n + test_n != n:
        raise ValueError("Per-class split counts must sum to samples_per_class")
    return ["train"] * train_n + ["validation"] * validation_n + ["test"] * test_n


def sample_quality_candidate_pool(
    inventory: pd.DataFrame,
    candidates_per_class: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic reserve pool for human semantic-quality screening.

    File integrity and duplicate checks cannot identify collages, heavy graphics,
    class mismatches, or unrelated multi-dish scenes. The candidate pool is built
    before any split exists, and a human accepts images until every class has enough
    clean representatives. Rejections are therefore replaced within the same class
    without altering an already sealed benchmark.
    """
    if candidates_per_class < 1:
        raise ValueError("candidates_per_class must be positive")
    required = {
        "relative_path",
        "food_class",
        "status",
        "sha256",
        "dhash",
        "duplicate_group",
        "width",
        "height",
        "bytes",
    }
    if missing := required.difference(inventory.columns):
        raise ValueError(f"Inventory missing columns: {sorted(missing)}")

    valid = inventory[inventory["status"].eq("ok")].copy()
    representatives = valid.sort_values(["food_class", "duplicate_group", "sha256"]).drop_duplicates(
        "duplicate_group", keep="first"
    )
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for food_class, group in representatives.groupby("food_class", sort=True):
        if len(group) < candidates_per_class:
            raise ValueError(
                f"Class {food_class!r} has only {len(group)} unique valid images; "
                f"{candidates_per_class} quality candidates are required"
            )
        order = rng.permutation(len(group))[:candidates_per_class]
        candidates = group.iloc[order].copy().reset_index(drop=True)
        candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
        rows.append(candidates)

    pool = pd.concat(rows, ignore_index=True)
    pool["candidate_id"] = pool["sha256"].str[:16]
    if pool["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs are not unique")
    columns = [
        "candidate_id",
        "candidate_rank",
        "relative_path",
        "food_class",
        "sha256",
        "dhash",
        "duplicate_group",
        "width",
        "height",
        "bytes",
    ]
    return pool[columns].sort_values(["food_class", "candidate_rank"]).reset_index(drop=True)


def manifest_from_screened_candidates(
    accepted_candidates: pd.DataFrame,
    samples_per_class: int = 20,
    train_per_class: int = 12,
    validation_per_class: int = 4,
    test_per_class: int = 4,
    seed: int = 42,
    dataset_slug: str = "",
    manifest_version: str = "visible-v2",
) -> pd.DataFrame:
    """Assign a sealed split only after semantic quality screening is complete."""
    required = {
        "candidate_id",
        "relative_path",
        "food_class",
        "sha256",
        "dhash",
        "duplicate_group",
        "width",
        "height",
        "bytes",
    }
    if missing := required.difference(accepted_candidates.columns):
        raise ValueError(f"Accepted candidates missing columns: {sorted(missing)}")
    if accepted_candidates["candidate_id"].duplicated().any():
        raise ValueError("Accepted candidate IDs must be unique")
    if accepted_candidates["duplicate_group"].duplicated().any():
        raise ValueError("Accepted candidates cannot share a duplicate group")

    split_labels = _sealed_split_labels(samples_per_class, train_per_class, validation_per_class, test_per_class)
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    for food_class, group in accepted_candidates.groupby("food_class", sort=True):
        if len(group) != samples_per_class:
            raise ValueError(
                f"Class {food_class!r} has {len(group)} accepted images; exactly {samples_per_class} are required"
            )
        chosen = group.sort_values("candidate_id").iloc[rng.permutation(samples_per_class)].copy()
        chosen["split"] = split_labels
        rows.append(chosen)

    manifest = pd.concat(rows, ignore_index=True)
    manifest["sample_id"] = manifest["candidate_id"]
    manifest["dataset_slug"] = dataset_slug
    manifest["manifest_version"] = manifest_version
    columns = [
        "sample_id",
        "relative_path",
        "food_class",
        "split",
        "dataset_slug",
        "manifest_version",
        "sha256",
        "dhash",
        "duplicate_group",
        "width",
        "height",
        "bytes",
    ]
    manifest = manifest[columns].sort_values(["food_class", "split", "sample_id"]).reset_index(drop=True)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Sample IDs are not unique")
    return manifest


def sample_annotation_manifest(
    inventory: pd.DataFrame,
    samples_per_class: int = 20,
    train_per_class: int = 12,
    validation_per_class: int = 4,
    test_per_class: int = 4,
    seed: int = 42,
    dataset_slug: str = "",
    manifest_version: str = "visible-v1",
) -> pd.DataFrame:
    """Select one image per duplicate group and seal balanced splits.

    Trade-off: we prioritize a pre-annotation, class-balanced sealed test over
    post-annotation iterative multi-label stratification. With 20 images per class,
    using labels to reassign test samples would make the benchmark easier to tune.
    """
    valid = inventory[inventory["status"].eq("ok")].copy()
    representatives = valid.sort_values(["food_class", "duplicate_group", "sha256"]).drop_duplicates(
        "duplicate_group", keep="first"
    )
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    split_labels = _sealed_split_labels(samples_per_class, train_per_class, validation_per_class, test_per_class)
    for food_class, group in representatives.groupby("food_class", sort=True):
        if len(group) < samples_per_class:
            raise ValueError(
                f"Class {food_class!r} has only {len(group)} unique valid images; {samples_per_class} are required"
            )
        chosen = group.iloc[rng.choice(len(group), size=samples_per_class, replace=False)].copy()
        order = rng.permutation(samples_per_class)
        chosen = chosen.iloc[order].reset_index(drop=True)
        chosen["split"] = split_labels
        rows.append(chosen)

    manifest = pd.concat(rows, ignore_index=True)
    manifest["sample_id"] = manifest["sha256"].str[:16]
    manifest["dataset_slug"] = dataset_slug
    manifest["manifest_version"] = manifest_version
    columns = [
        "sample_id",
        "relative_path",
        "food_class",
        "split",
        "dataset_slug",
        "manifest_version",
        "sha256",
        "dhash",
        "duplicate_group",
        "width",
        "height",
        "bytes",
    ]
    manifest = manifest[columns].sort_values(["food_class", "split", "sample_id"]).reset_index(drop=True)
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Sample IDs are not unique")
    return manifest


def manifest_digest(manifest: pd.DataFrame) -> str:
    stable = manifest.sort_values("sample_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()
