from pathlib import Path

import pandas as pd
from PIL import Image

from src.data import (
    assign_duplicate_groups,
    build_image_inventory,
    cross_dataset_overlap_report,
    manifest_digest,
    sample_annotation_manifest,
)


def _create_dataset(root: Path, classes: int = 2, images_per_class: int = 5):
    for class_index in range(classes):
        folder = root / f"class_{class_index}"
        folder.mkdir(parents=True)
        for image_index in range(images_per_class):
            image = Image.new("RGB", (16, 16), (class_index * 50, image_index * 20, 30))
            image.putpixel((image_index % 16, class_index % 16), (255, 255, image_index))
            image.save(folder / f"image_{image_index}.png")


def test_balanced_manifest_is_deterministic(tmp_path):
    _create_dataset(tmp_path, images_per_class=5)
    inventory = assign_duplicate_groups(build_image_inventory(tmp_path), perceptual_max_distance=-1)
    first = sample_annotation_manifest(
        inventory, samples_per_class=4, train_per_class=2, validation_per_class=1, test_per_class=1, seed=42
    )
    second = sample_annotation_manifest(
        inventory, samples_per_class=4, train_per_class=2, validation_per_class=1, test_per_class=1, seed=42
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 8
    assert first.groupby(["food_class", "split"]).size().to_dict() == {
        ("class_0", "test"): 1,
        ("class_0", "train"): 2,
        ("class_0", "validation"): 1,
        ("class_1", "test"): 1,
        ("class_1", "train"): 2,
        ("class_1", "validation"): 1,
    }
    assert manifest_digest(first) == manifest_digest(second)


def test_exact_duplicates_share_group(tmp_path):
    folder = tmp_path / "food"
    folder.mkdir()
    image = Image.new("RGB", (8, 8), (1, 2, 3))
    image.save(folder / "one.png")
    image.save(folder / "two.png")
    inventory = assign_duplicate_groups(build_image_inventory(tmp_path), perceptual_max_distance=0)
    assert inventory["duplicate_group"].nunique() == 1
    assert set(inventory["duplicate_group_size"]) == {2}
    assert "dhash" in inventory.columns


def test_cross_class_duplicates_share_global_group(tmp_path):
    for class_name in ["class_a", "class_b"]:
        folder = tmp_path / class_name
        folder.mkdir()
        Image.new("RGB", (8, 8), (12, 34, 56)).save(folder / "copy.png")
    inventory = assign_duplicate_groups(build_image_inventory(tmp_path), perceptual_max_distance=0)
    assert inventory["food_class"].nunique() == 2
    assert inventory["duplicate_group"].nunique() == 1


def test_cross_dataset_overlap_quarantines_exact_copy(tmp_path):
    reference_root = tmp_path / "reference" / "food"
    candidate_root = tmp_path / "candidate" / "recipes"
    reference_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    image = Image.new("RGB", (10, 10), (90, 20, 10))
    image.save(reference_root / "target.png")
    image.save(candidate_root / "leak.png")
    reference_inventory = build_image_inventory(tmp_path / "reference")
    reference = reference_inventory.rename_axis("sample_id").reset_index()
    reference["sample_id"] = "target-sample"
    candidates = build_image_inventory(tmp_path / "candidate")
    report = cross_dataset_overlap_report(candidates, reference, perceptual_max_distance=0)
    assert len(report) == 1
    assert report.iloc[0]["match_type"] == "exact_sha256"
