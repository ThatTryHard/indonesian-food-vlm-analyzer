#!/usr/bin/env python
"""Validate primary/full and secondary/evaluation annotation coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.annotations import annotation_agreement, build_adjudication_queue, validate_annotation_sheet  # noqa: E402
from src.artifacts import project_protocol_digest  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import manifest_digest, sha256_file  # noqa: E402
from src.ontology import IngredientOntology  # noqa: E402
from src.quality import validate_quality_screen  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--annotator-a", required=True, type=Path)
    parser.add_argument("--annotator-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    ontology_path = PROJECT_ROOT / "data" / "ontology" / "visible_ingredients.json"
    config_path = PROJECT_ROOT / "configs" / "project.json"
    config = load_config(config_path)
    ontology = IngredientOntology.from_json(ontology_path)
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    with args.lock.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    if manifest_digest(manifest) != lock.get("manifest_sha256"):
        raise ValueError("Manifest does not match its pre-annotation lock")
    if sha256_file(ontology_path) != lock.get("ontology_sha256"):
        raise ValueError("Ontology does not match the pre-annotation lock")
    if sha256_file(config_path) != lock.get("config_sha256"):
        raise ValueError("Config does not match the pre-annotation lock")
    if project_protocol_digest(PROJECT_ROOT) != lock.get("project_protocol_sha256"):
        raise ValueError("Protocol code does not match the pre-annotation lock")
    candidate_pool_path = args.lock.parent / "quality_candidate_pool.csv"
    quality_screen_path = args.lock.parent / "quality_screen.csv"
    if sha256_file(candidate_pool_path) != lock.get("candidate_pool_sha256"):
        raise ValueError("Quality candidate pool does not match the pre-annotation lock")
    if sha256_file(quality_screen_path) != lock.get("quality_screen_sha256"):
        raise ValueError("Quality screen does not match the pre-annotation lock")
    candidate_pool = pd.read_csv(candidate_pool_path, keep_default_na=False)
    validate_quality_screen(
        pd.read_csv(quality_screen_path, keep_default_na=False),
        candidate_pool,
        samples_per_class=config["benchmark"]["samples_per_class"],
        require_complete=True,
    )

    primary = validate_annotation_sheet(pd.read_csv(args.annotator_a, keep_default_na=False), manifest, ontology)
    evaluation_manifest = manifest[manifest["split"].isin(config["benchmark"]["secondary_annotation_splits"])].copy()
    secondary = validate_annotation_sheet(
        pd.read_csv(args.annotator_b, keep_default_na=False), evaluation_manifest, ontology
    )
    primary_evaluation = validate_annotation_sheet(
        primary[primary["sample_id"].isin(evaluation_manifest["sample_id"])].copy(),
        evaluation_manifest,
        ontology,
    )
    if primary["annotator_id"].iloc[0] == secondary["annotator_id"].iloc[0]:
        raise ValueError("Two independent annotator IDs are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.output_dir / "annotations_a_validated.csv", index=False)
    secondary.to_csv(args.output_dir / "annotations_b_validated.csv", index=False)
    queue = build_adjudication_queue(primary_evaluation, secondary, evaluation_manifest)
    queue.to_csv(args.output_dir / "adjudication_queue.csv", index=False)
    agreement = annotation_agreement(primary_evaluation, secondary, ontology)
    with (args.output_dir / "annotation_agreement.json").open("w", encoding="utf-8") as handle:
        json.dump(agreement, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "agreement": agreement,
                "needs_adjudication": int(queue["status"].eq("needs_adjudication").sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
