#!/usr/bin/env python
"""Prepare a semantic quality screen, then seal the 260-image benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import create_annotation_sheet  # noqa: E402
from src.artifacts import project_protocol_digest  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import (  # noqa: E402
    assign_duplicate_groups,
    build_image_inventory,
    manifest_digest,
    manifest_from_screened_candidates,
    sample_quality_candidate_pool,
    sha256_file,
)
from src.ontology import IngredientOntology  # noqa: E402
from src.quality import (  # noqa: E402
    accepted_candidates_from_screen,
    create_quality_screen,
    quality_screen_progress,
    validate_quality_screen,
)
from src.vlm import build_visible_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "seal"])
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "artifacts" / "benchmark", type=Path)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "project.json", type=Path)
    parser.add_argument("--force", action="store_true", help="Intentionally replace artifacts for this stage")
    return parser.parse_args()


def _verify_class_inventory(inventory: pd.DataFrame, expected_classes: list[str]) -> None:
    observed = sorted(inventory.loc[inventory["status"].eq("ok"), "food_class"].unique())
    expected = sorted(expected_classes)
    if observed != expected:
        raise ValueError(
            "Target class inventory differs from config: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"unexpected={sorted(set(observed) - set(expected))}"
        )


def prepare_quality_screen(args: argparse.Namespace, config: dict) -> None:
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required for the prepare stage")
    benchmark = config["benchmark"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "benchmark_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} is already sealed; create a fresh benchmark directory instead of replacing it"
        )
    output_paths = [
        args.output_dir / "image_inventory.csv",
        args.output_dir / "corrupt_images.csv",
        args.output_dir / "quality_candidate_pool.csv",
        args.output_dir / "quality_screen.csv",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Quality-screen artifacts already exist: {[str(path) for path in existing]}")

    inventory = assign_duplicate_groups(
        build_image_inventory(args.dataset_root),
        perceptual_max_distance=benchmark["perceptual_hash_max_distance"],
    )
    _verify_class_inventory(inventory, benchmark["expected_classes"])
    inventory.to_csv(args.output_dir / "image_inventory.csv", index=False)
    inventory[inventory["status"].ne("ok")].to_csv(args.output_dir / "corrupt_images.csv", index=False)

    candidate_pool = sample_quality_candidate_pool(
        inventory,
        candidates_per_class=benchmark["quality_candidates_per_class"],
        seed=config["project"]["seed"],
    )
    candidate_pool.to_csv(args.output_dir / "quality_candidate_pool.csv", index=False)
    quality_screen = create_quality_screen(candidate_pool)
    quality_screen.to_csv(args.output_dir / "quality_screen.csv", index=False)
    summary = {
        "stage": "quality_screen_prepared",
        "candidate_rows": len(candidate_pool),
        "candidates_per_class": candidate_pool["food_class"].value_counts().sort_index().to_dict(),
        "required_acceptances_per_class": benchmark["samples_per_class"],
        "quality_screen_version": benchmark["quality_screen_version"],
        "manifest_sealed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def seal_benchmark(args: argparse.Namespace, config: dict) -> None:
    benchmark = config["benchmark"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "benchmark_manifest.csv"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} already exists; refusing to replace the sealed benchmark")

    pool_path = args.output_dir / "quality_candidate_pool.csv"
    screen_path = args.output_dir / "quality_screen.csv"
    if not pool_path.is_file() or not screen_path.is_file():
        raise FileNotFoundError("Prepare and complete quality_candidate_pool.csv and quality_screen.csv first")
    candidate_pool = pd.read_csv(pool_path, keep_default_na=False)
    quality_screen = pd.read_csv(screen_path, keep_default_na=False)
    _verify_class_inventory(candidate_pool.assign(status="ok"), benchmark["expected_classes"])
    normalized_screen = validate_quality_screen(
        quality_screen,
        candidate_pool,
        samples_per_class=benchmark["samples_per_class"],
        require_complete=True,
    )
    normalized_screen.to_csv(screen_path, index=False)
    accepted = accepted_candidates_from_screen(
        normalized_screen,
        candidate_pool,
        samples_per_class=benchmark["samples_per_class"],
    )
    manifest = manifest_from_screened_candidates(
        accepted,
        samples_per_class=benchmark["samples_per_class"],
        train_per_class=benchmark["train_per_class"],
        validation_per_class=benchmark["validation_per_class"],
        test_per_class=benchmark["test_per_class"],
        seed=config["project"]["seed"],
        dataset_slug=config["datasets"]["indonesian_target"]["slug"],
        manifest_version=benchmark["manifest_version"],
    )
    manifest.to_csv(manifest_path, index=False)

    primary = create_annotation_sheet(manifest, "annotator_a").sample(frac=1, random_state=101).reset_index(drop=True)
    secondary_manifest = manifest[manifest["split"].isin(benchmark["secondary_annotation_splits"])].copy()
    secondary = (
        create_annotation_sheet(secondary_manifest, "annotator_b")
        .sample(frac=1, random_state=202)
        .reset_index(drop=True)
    )
    primary.to_csv(args.output_dir / "annotations_annotator_a.csv", index=False)
    secondary.to_csv(args.output_dir / "annotations_annotator_b.csv", index=False)

    ontology_path = PROJECT_ROOT / "data/ontology/visible_ingredients.json"
    ontology = IngredientOntology.from_json(ontology_path)
    vlm_prompt = build_visible_prompt(ontology)
    progress = quality_screen_progress(normalized_screen, candidate_pool, benchmark["samples_per_class"])
    lock = {
        "manifest_sha256": manifest_digest(manifest),
        "candidate_pool_sha256": sha256_file(pool_path),
        "quality_screen_sha256": sha256_file(screen_path),
        "quality_screen_version": benchmark["quality_screen_version"],
        "quality_screen_reviewed_rows": progress["reviewed"],
        "quality_screen_rejected_rows": progress["rejected"],
        "ontology_sha256": sha256_file(ontology_path),
        "config_sha256": sha256_file(args.config),
        "vlm_prompt_sha256": hashlib.sha256(vlm_prompt.encode("utf-8")).hexdigest(),
        "vlm_prompt_version": config["vlm"]["prompt_version"],
        "project_protocol_sha256": project_protocol_digest(PROJECT_ROOT),
        "rows": len(manifest),
        "annotation_rows": {"annotator_a": len(primary), "annotator_b": len(secondary)},
        "secondary_annotation_splits": benchmark["secondary_annotation_splits"],
        "class_counts": manifest["food_class"].value_counts().sort_index().to_dict(),
        "split_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "config_version": config["project"]["version"],
        "screened_before_split": True,
        "sealed_before_ingredient_annotation": True,
        "sealed_before_annotation": True,
    }
    with (args.output_dir / "manifest_lock.json").open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage == "prepare":
        prepare_quality_screen(args, config)
    else:
        seal_benchmark(args, config)


if __name__ == "__main__":
    main()
