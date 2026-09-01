#!/usr/bin/env python
"""Create the immutable 260-image annotation benchmark packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import create_annotation_sheet  # noqa: E402
from src.artifacts import project_protocol_digest  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import (  # noqa: E402
    assign_duplicate_groups,
    build_image_inventory,
    manifest_digest,
    sample_annotation_manifest,
    sha256_file,
)
from src.ontology import IngredientOntology  # noqa: E402
from src.vlm import build_visible_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=PROJECT_ROOT / "artifacts" / "benchmark", type=Path)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "project.json", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest intentionally")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    benchmark = config["benchmark"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "benchmark_manifest.csv"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} already exists; refusing to replace the sealed benchmark")

    inventory = assign_duplicate_groups(
        build_image_inventory(args.dataset_root),
        perceptual_max_distance=benchmark["perceptual_hash_max_distance"],
    )
    observed_classes = sorted(inventory.loc[inventory["status"].eq("ok"), "food_class"].unique())
    expected_classes = sorted(benchmark["expected_classes"])
    if observed_classes != expected_classes:
        raise ValueError(
            "Target class inventory does not match the frozen benchmark: "
            f"missing={sorted(set(expected_classes) - set(observed_classes))}, "
            f"unexpected={sorted(set(observed_classes) - set(expected_classes))}"
        )
    corrupt = inventory[inventory["status"].ne("ok")]
    inventory.to_csv(args.output_dir / "image_inventory.csv", index=False)
    corrupt.to_csv(args.output_dir / "corrupt_images.csv", index=False)

    manifest = sample_annotation_manifest(
        inventory,
        samples_per_class=benchmark["samples_per_class"],
        train_per_class=benchmark["train_per_class"],
        validation_per_class=benchmark["validation_per_class"],
        test_per_class=benchmark["test_per_class"],
        seed=config["project"]["seed"],
        dataset_slug=config["datasets"]["indonesian_target"]["slug"],
        manifest_version=benchmark["manifest_version"],
    )
    manifest.to_csv(manifest_path, index=False)
    # Different seeded orders reduce anchoring while preserving identical sample membership.
    create_annotation_sheet(manifest, "annotator_a").sample(frac=1, random_state=101).reset_index(drop=True).to_csv(
        args.output_dir / "annotations_annotator_a.csv", index=False
    )
    create_annotation_sheet(manifest, "annotator_b").sample(frac=1, random_state=202).reset_index(drop=True).to_csv(
        args.output_dir / "annotations_annotator_b.csv", index=False
    )
    ontology = IngredientOntology.from_json(PROJECT_ROOT / "data/ontology/visible_ingredients.json")
    vlm_prompt = build_visible_prompt(ontology)
    lock = {
        "manifest_sha256": manifest_digest(manifest),
        "ontology_sha256": sha256_file(PROJECT_ROOT / "data/ontology/visible_ingredients.json"),
        "config_sha256": sha256_file(args.config),
        "vlm_prompt_sha256": hashlib.sha256(vlm_prompt.encode("utf-8")).hexdigest(),
        "vlm_prompt_version": config["vlm"]["prompt_version"],
        "project_protocol_sha256": project_protocol_digest(PROJECT_ROOT),
        "rows": len(manifest),
        "class_counts": manifest["food_class"].value_counts().sort_index().to_dict(),
        "split_counts": manifest["split"].value_counts().sort_index().to_dict(),
        "config_version": config["project"]["version"],
        "sealed_before_annotation": True,
    }
    with (args.output_dir / "manifest_lock.json").open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
