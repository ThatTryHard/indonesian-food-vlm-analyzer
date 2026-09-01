#!/usr/bin/env python
"""Validate two annotation passes and create an adjudication queue."""

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
from src.data import manifest_digest, sha256_file  # noqa: E402
from src.ontology import IngredientOntology  # noqa: E402


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
    first = validate_annotation_sheet(pd.read_csv(args.annotator_a, keep_default_na=False), manifest, ontology)
    second = validate_annotation_sheet(pd.read_csv(args.annotator_b, keep_default_na=False), manifest, ontology)
    if first["annotator_id"].iloc[0] == second["annotator_id"].iloc[0]:
        raise ValueError("Two independent annotator IDs are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first.to_csv(args.output_dir / "annotations_a_validated.csv", index=False)
    second.to_csv(args.output_dir / "annotations_b_validated.csv", index=False)
    queue = build_adjudication_queue(first, second, manifest)
    queue.to_csv(args.output_dir / "adjudication_queue.csv", index=False)
    agreement = annotation_agreement(first, second, ontology)
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
