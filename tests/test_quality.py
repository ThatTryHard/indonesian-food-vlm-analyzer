from pathlib import Path

import pandas as pd
from PIL import Image

from src.data import (
    assign_duplicate_groups,
    build_image_inventory,
    manifest_from_screened_candidates,
    sample_quality_candidate_pool,
)
from src.quality import (
    accepted_candidates_from_screen,
    create_quality_screen,
    quality_screen_progress,
    validate_quality_screen,
)
from src.quality_ui import QualityScreenApp


def _quality_dataset(root: Path, images_per_class: int = 6):
    for class_index, class_name in enumerate(["class_a", "class_b"]):
        folder = root / class_name
        folder.mkdir(parents=True)
        for image_index in range(images_per_class):
            image = Image.new(
                "RGB",
                (24, 24),
                (class_index * 80, image_index * 25, (image_index + 1) * 20),
            )
            image.putpixel((image_index, class_index), (255, image_index, class_index))
            image.save(folder / f"image_{image_index}.png")


def _candidate_pool(root: Path) -> pd.DataFrame:
    _quality_dataset(root)
    inventory = assign_duplicate_groups(build_image_inventory(root), perceptual_max_distance=-1)
    return sample_quality_candidate_pool(inventory, candidates_per_class=6, seed=42)


def test_rejected_candidate_is_replaced_before_split_sealing(tmp_path):
    pool = _candidate_pool(tmp_path)
    screen = create_quality_screen(pool)

    first_a = screen.index[screen["food_class"].eq("class_a")][0]
    screen.loc[first_a, ["decision", "rejection_reason"]] = [
        "reject",
        "composite_or_multi_panel",
    ]
    for class_name in ["class_a", "class_b"]:
        accepted_indices = screen.index[screen["food_class"].eq(class_name) & screen["decision"].eq("")][:4]
        screen.loc[accepted_indices, "decision"] = "accept"

    normalized = validate_quality_screen(screen, pool, samples_per_class=4, require_complete=True)
    accepted = accepted_candidates_from_screen(normalized, pool, samples_per_class=4)
    manifest = manifest_from_screened_candidates(
        accepted,
        samples_per_class=4,
        train_per_class=2,
        validation_per_class=1,
        test_per_class=1,
        seed=42,
    )

    assert first_a not in normalized.index[normalized["decision"].eq("accept")]
    assert quality_screen_progress(normalized, pool, 4)["rejected"] == 1
    assert manifest.groupby(["food_class", "split"]).size().to_dict() == {
        ("class_a", "test"): 1,
        ("class_a", "train"): 2,
        ("class_a", "validation"): 1,
        ("class_b", "test"): 1,
        ("class_b", "train"): 2,
        ("class_b", "validation"): 1,
    }


def test_quality_ui_moves_to_same_class_reserve_after_rejection(tmp_path):
    pool = _candidate_pool(tmp_path)
    screen = create_quality_screen(pool)
    output_path = tmp_path / "quality_screen.csv"
    app = QualityScreenApp(screen, pool, tmp_path, output_path, samples_per_class=4)

    current_class = app.screen.at[app.position, "food_class"]
    current_rank = int(app.screen.at[app.position, "candidate_rank"])
    app.decision.value = "composite_or_multi_panel"
    assert app._save_current()

    next_position = app._next_required_position(current_class, current_rank)
    assert next_position is not None
    assert app.screen.at[next_position, "food_class"] == current_class
    saved = pd.read_csv(output_path, keep_default_na=False)
    assert saved.loc[app.position, "decision"] == "reject"
    assert saved.loc[app.position, "rejection_reason"] == "composite_or_multi_panel"


def test_quality_ui_starts_next_class_at_first_candidate(tmp_path):
    pool = _candidate_pool(tmp_path)
    screen = create_quality_screen(pool)
    first_class = sorted(screen["food_class"].unique())[0]
    first_class_indices = screen.index[screen["food_class"].eq(first_class)][:4]
    screen.loc[first_class_indices, "decision"] = "accept"
    app = QualityScreenApp(screen, pool, tmp_path, tmp_path / "screen.csv", samples_per_class=4)

    assert app.screen.at[app.position, "food_class"] != first_class
    assert int(app.screen.at[app.position, "candidate_rank"]) == 1
