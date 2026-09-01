#!/usr/bin/env python
"""Generate the two sequential, reviewable Kaggle notebooks with stdlib JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOK_01 = notebook(
    [
        markdown("""# 01: Build and Annotate the 260-Image Benchmark

This notebook creates a corruption-checked, deduplicated, class-balanced benchmark. It seals train, validation, and test membership before preparing two independent visible-ingredient annotation passes.

Primary task: **per-image visible-component recognition**. Recipe knowledge, hidden ingredients, web associations, food naming, and nutrition are not ground truth for this task.

Run this notebook first. Notebook 02 is intentionally blocked until both annotation passes and human adjudication are complete."""),
        markdown("""## Define the target and frozen ontology

The ontology uses canonical multiword component labels and gives every label a visual-evidence rule. Recipe-writing fragments and preparation terms such as `all`, `at`, `freshly`, and `coarsely` are not model targets.

This keeps the task visually defensible. Hidden recipe ingredients cannot be scored as observable image evidence. Label validity therefore takes priority over vocabulary breadth, and hidden spices such as garlic and turmeric are outside the primary ground truth unless they are visibly identifiable."""),
        code("""# Kaggle setup: clone the project when the notebook was imported without repository files.
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/ThatTryHard/indonesian-food-vlm-analyzer.git"
REPOSITORY_REVISION = os.environ.get("FOOD_VLM_REVISION", "main")
PROJECT_ROOT = Path(os.environ.get("FOOD_VLM_PROJECT_ROOT", "/kaggle/working/indonesian-food-vlm-analyzer"))

if not (PROJECT_ROOT / "src").exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", REPOSITORY_REVISION,
        REPOSITORY_URL, str(PROJECT_ROOT),
    ], check=True)

# Notebook 01 only needs packages already supplied by Kaggle's pinned base image.
# Replacing NumPy or Pandas inside a running kernel can mix incompatible binary modules.
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Environment: Kaggle pinned base packages (no in-kernel replacement)")
print("Git revision:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())"""),
        code("""import hashlib
import json
import shutil

import kagglehub
import pandas as pd
from IPython.display import display

from src.artifacts import project_protocol_digest
from src.config import artifact_dir, load_config
from src.data import manifest_digest, sha256_file
from src.ontology import IngredientOntology
from src.vlm import build_visible_prompt

config_path = PROJECT_ROOT / "configs/project.json"
ontology_path = PROJECT_ROOT / "data/ontology/visible_ingredients.json"
config = load_config(config_path)
ontology = IngredientOntology.from_json(ontology_path)
ARTIFACT_ROOT = artifact_dir(config)
BENCHMARK_DIR = ARTIFACT_ROOT / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Resume safely from a packet exported by an earlier Kaggle session.
# Copy it into writable storage because /kaggle/input is read-only.
packet_override = os.environ.get("FOOD_VLM_BENCHMARK_PACKET")
packet_source = Path(packet_override) if packet_override else None
if packet_source is None and Path("/kaggle/input").exists():
    attached_manifests = list(Path("/kaggle/input").rglob("benchmark_manifest.csv"))
    attached_archives = list(Path("/kaggle/input").rglob("benchmark_packet.zip"))
    candidates = attached_manifests + attached_archives
    if len(candidates) == 1:
        packet_source = candidates[0]
    elif len(candidates) > 1:
        print("Multiple prior packets found; set FOOD_VLM_BENCHMARK_PACKET to resume one explicitly.")
if packet_source is not None and not (BENCHMARK_DIR / "benchmark_manifest.csv").exists():
    if not packet_source.exists():
        raise FileNotFoundError(packet_source)
    if packet_source.is_file() and packet_source.suffix.lower() == ".zip":
        shutil.unpack_archive(str(packet_source), str(BENCHMARK_DIR))
    else:
        source_dir = packet_source.parent if packet_source.is_file() else packet_source
        packet_filenames = {
            "benchmark_manifest.csv", "manifest_lock.json", "image_inventory.csv", "corrupt_images.csv",
            "annotations_annotator_a.csv", "annotations_annotator_b.csv",
        }
        for filename in packet_filenames:
            source_file = source_dir / filename
            if source_file.is_file():
                shutil.copy2(source_file, BENCHMARK_DIR / filename)
    print("Restored writable benchmark packet from:", packet_source)

ontology_table = pd.DataFrame([
    {"id": label.id, "category": label.category, "visual_rule": label.hint}
    for label in ontology.labels
])
print("Primary task:", config["project"]["primary_task"])
print("Ontology version:", ontology.version, "| labels:", len(ontology.ids))
display(ontology_table)"""),
        markdown("""## Validate images and seal the split

Every image is verified, hashed with SHA-256, assigned to a perceptual duplicate group, and sampled once. The manifest contains 20 unique images per class: 12 train, 4 validation, and 4 test.

Split membership is frozen before annotation to prevent duplicate contamination and keep finished labels from influencing test composition. The benchmark uses pre-annotation class balance instead of rearranging samples after ingredient labels are known."""),
        code("""# Resolve the exact Kaggle dataset by its frozen slug; an explicit environment path can override it.
dataset_config = config["datasets"]["indonesian_target"]
explicit_target = os.environ.get(dataset_config["path_env"])
TARGET_DATASET_ROOT = Path(explicit_target) if explicit_target else Path(kagglehub.dataset_download(dataset_config["slug"]))

print("Dataset slug:", dataset_config["slug"])
print("Resolved root:", TARGET_DATASET_ROOT)
if not TARGET_DATASET_ROOT.exists():
    raise FileNotFoundError(TARGET_DATASET_ROOT)"""),
        code("""# Build once. Existing manifests are never silently overwritten.
manifest_path = BENCHMARK_DIR / "benchmark_manifest.csv"
if not manifest_path.exists():
    subprocess.run([
        sys.executable,
        str(PROJECT_ROOT / "scripts/build_benchmark.py"),
        "--dataset-root", str(TARGET_DATASET_ROOT),
        "--output-dir", str(BENCHMARK_DIR),
    ], check=True)
else:
    print("Using existing sealed manifest:", manifest_path)

manifest = pd.read_csv(manifest_path, keep_default_na=False)
lock = json.loads((BENCHMARK_DIR / "manifest_lock.json").read_text(encoding="utf-8"))

assert len(manifest) == 260, f"Expected 260 rows, found {len(manifest)}"
assert manifest["sample_id"].is_unique
assert manifest.groupby("food_class").size().eq(20).all()
assert sorted(manifest["food_class"].unique()) == sorted(config["benchmark"]["expected_classes"])
assert manifest_digest(manifest) == lock["manifest_sha256"], "Existing manifest no longer matches its lock"
assert sha256_file(ontology_path) == lock["ontology_sha256"], "Ontology changed after benchmark sealing"
assert sha256_file(config_path) == lock["config_sha256"], "Config changed after benchmark sealing"
assert hashlib.sha256(build_visible_prompt(ontology).encode("utf-8")).hexdigest() == lock["vlm_prompt_sha256"]
assert project_protocol_digest(PROJECT_ROOT) == lock["project_protocol_sha256"], "Protocol code changed after sealing"
per_class_splits = manifest.groupby(["food_class", "split"]).size().unstack(fill_value=0)
assert per_class_splits["train"].eq(12).all()
assert per_class_splits["validation"].eq(4).all()
assert per_class_splits["test"].eq(4).all()

print(json.dumps(lock, indent=2, sort_keys=True))
display(per_class_splits)
display(pd.read_csv(BENCHMARK_DIR / "corrupt_images.csv").head())"""),
        markdown("""## Create two independent per-image annotation passes

Each image receives `visible_ingredients`, `uncertain_ingredients`, explicit exclusion or all-negative flags, and notes from two independent passes. The two sheets use different random orders. Per-image judgments are required because a class-level recipe list cannot serve as ground truth for every photograph.

Read `docs/ANNOTATION_GUIDE.md` completely before starting. Annotators must not inspect each other's sheets. If you personally perform both passes, use a washout period and disclose that it is a weaker design than two people."""),
        code("""print((PROJECT_ROOT / "docs/ANNOTATION_GUIDE.md").read_text(encoding="utf-8"))"""),
        code("""# Choose exactly one sheet for this independent pass, then run this cell.
# Use a fresh session and choose annotator_b for the second pass.
ANNOTATOR_ID = "annotator_a"  # change to annotator_b only in the separate second pass

if ANNOTATOR_ID not in {"annotator_a", "annotator_b"}:
    raise ValueError("ANNOTATOR_ID must be annotator_a or annotator_b")

annotation_path = BENCHMARK_DIR / f"annotations_{ANNOTATOR_ID}.csv"
sheet = pd.read_csv(annotation_path, keep_default_na=False)
from src.annotations import validate_annotation_sheet
sheet = validate_annotation_sheet(sheet, manifest, ontology, require_complete=False)
print("Editing:", annotation_path)
print("Do not open the other annotator's CSV during this pass.")"""),
        code("""from src.annotation_ui import AnnotationApp

app = AnnotationApp(
    sheet=sheet,
    ontology=ontology,
    image_root=TARGET_DATASET_ROOT,
    output_csv=annotation_path,
)
app.display()"""),
        markdown("""## Annotation completion check and export

The validator below does not accept blank rows, unknown labels, or labels marked both visible and uncertain. If an assessable image supports none of the 43 labels, select `No supported ontology label`; never force a positive. Export the packet after each session; `/kaggle/working` is temporary."""),
        code("""from src.annotations import annotation_progress, validate_annotation_sheet

saved_sheet = pd.read_csv(annotation_path, keep_default_na=False)
progress = annotation_progress(saved_sheet)
print(progress)

if progress["remaining"] == 0:
    validate_annotation_sheet(saved_sheet, manifest, ontology, require_complete=True)
    print("This annotation pass is structurally complete.")
else:
    print("Resume the interface before moving to notebook 02.")

archive_base = ARTIFACT_ROOT / "benchmark_packet"
archive_path = shutil.make_archive(str(archive_base), "zip", BENCHMARK_DIR)
print("Download and preserve:", archive_path)"""),
        markdown("""## Handoff to Notebook 02

Proceed only when `annotations_annotator_a.csv` and `annotations_annotator_b.csv` are independently complete. Upload the benchmark packet as a private Kaggle dataset or attach it to Notebook 02. Notebook 02 validates the lock digest, measures agreement, forces explicit adjudication, and only then trains models."""),
    ]
)


NOTEBOOK_02 = notebook(
    [
        markdown("""# 02: Train and Evaluate the CNN and VLM Benchmark

Run this notebook only after Notebook 01 produced two complete annotation sheets. It validates and adjudicates the benchmark, trains honest baselines, compares a frozen VLM on the same ontology, and opens the sealed test exactly once after all choices are frozen.

No headline score is generated while annotations or adjudications are missing."""),
        code("""# Kaggle setup: clone project files if this notebook was imported alone.
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/ThatTryHard/indonesian-food-vlm-analyzer.git"
REPOSITORY_REVISION = os.environ.get("FOOD_VLM_REVISION", "main")
PROJECT_ROOT = Path(os.environ.get("FOOD_VLM_PROJECT_ROOT", "/kaggle/working/indonesian-food-vlm-analyzer"))
if not (PROJECT_ROOT / "src").exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", REPOSITORY_REVISION,
        REPOSITORY_URL, str(PROJECT_ROOT),
    ], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_ROOT / "requirements-kaggle.txt")], check=True)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Git revision:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())"""),
        code("""import gc
import hashlib
import json
import shutil

import kagglehub
import numpy as np
import pandas as pd
import torch
from IPython.display import display

from src.artifacts import project_protocol_digest, write_json
from src.config import artifact_dir, load_config
from src.data import (
    assign_duplicate_groups,
    build_image_inventory,
    cross_dataset_overlap_report,
    manifest_digest,
    sha256_file,
)
from src.ontology import IngredientOntology
from src.training import seed_everything, seed_worker

config_path = PROJECT_ROOT / "configs/project.json"
ontology_path = PROJECT_ROOT / "data/ontology/visible_ingredients.json"
config = load_config(config_path)
seed_everything(config["project"]["seed"])
ontology = IngredientOntology.from_json(ontology_path)
ARTIFACT_ROOT = artifact_dir(config)
RUN_DIR = ARTIFACT_ROOT / "run"
RUN_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)"""),
        markdown("""## Verify benchmark integrity and annotation completeness

The manifest hash must match the pre-annotation lock, and two complete sheets with distinct annotator IDs are required. Training and evaluation remain blocked if the split has changed or either annotation pass is incomplete."""),
        code("""# Point this to the packet from Notebook 01 (directory, manifest, or ZIP).
# If exactly one manifest/packet ZIP exists under /kaggle/input, it is selected automatically.
explicit_packet = os.environ.get("FOOD_VLM_BENCHMARK_PACKET")
packet_source = Path(explicit_packet) if explicit_packet else None
if packet_source is None:
    manifests = list(Path("/kaggle/input").rglob("benchmark_manifest.csv"))
    archives = list(Path("/kaggle/input").rglob("benchmark_packet.zip"))
    candidates = manifests + archives
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one benchmark manifest/ZIP under /kaggle/input, found {len(candidates)}. "
            "Set FOOD_VLM_BENCHMARK_PACKET explicitly."
        )
    packet_source = candidates[0]

if not packet_source.exists():
    raise FileNotFoundError(packet_source)
if packet_source.is_file() and packet_source.suffix.lower() == ".zip":
    BENCHMARK_PACKET_DIR = Path("/kaggle/working/restored-benchmark-packet")
    if BENCHMARK_PACKET_DIR.exists():
        shutil.rmtree(BENCHMARK_PACKET_DIR)
    shutil.unpack_archive(str(packet_source), str(BENCHMARK_PACKET_DIR))
elif packet_source.is_file():
    BENCHMARK_PACKET_DIR = packet_source.parent
else:
    BENCHMARK_PACKET_DIR = packet_source

manifest = pd.read_csv(BENCHMARK_PACKET_DIR / "benchmark_manifest.csv", keep_default_na=False)
lock = json.loads((BENCHMARK_PACKET_DIR / "manifest_lock.json").read_text(encoding="utf-8"))
actual_digest = manifest_digest(manifest)
if actual_digest != lock["manifest_sha256"]:
    raise RuntimeError("Manifest lock mismatch; do not train or evaluate")
if sha256_file(ontology_path) != lock.get("ontology_sha256"):
    raise RuntimeError("Ontology content changed after the benchmark was sealed")
if sha256_file(config_path) != lock.get("config_sha256"):
    raise RuntimeError("Project config changed after the benchmark was sealed")
from src.vlm import build_visible_prompt
if hashlib.sha256(build_visible_prompt(ontology).encode("utf-8")).hexdigest() != lock.get("vlm_prompt_sha256"):
    raise RuntimeError("VLM prompt changed after the benchmark was sealed")
if project_protocol_digest(PROJECT_ROOT) != lock.get("project_protocol_sha256"):
    raise RuntimeError("Protocol code changed after the benchmark was sealed")
if len(manifest) != 260 or not manifest["sample_id"].is_unique:
    raise RuntimeError("Benchmark must contain exactly 260 unique sample IDs")
if sorted(manifest["food_class"].unique()) != sorted(config["benchmark"]["expected_classes"]):
    raise RuntimeError("Benchmark class set differs from the frozen config")
expected_split_counts = {"train": 12, "validation": 4, "test": 4}
actual_split_counts = manifest.groupby(["food_class", "split"]).size().unstack(fill_value=0)
if any(not actual_split_counts[split_name].eq(count).all() for split_name, count in expected_split_counts.items()):
    raise RuntimeError("Per-class 12/4/4 split invariant failed")

target_cfg = config["datasets"]["indonesian_target"]
explicit_target = os.environ.get(target_cfg["path_env"])
TARGET_DATASET_ROOT = Path(explicit_target) if explicit_target else Path(kagglehub.dataset_download(target_cfg["slug"]))
missing_images = []
changed_images = []
for row in manifest.itertuples(index=False):
    image_path = TARGET_DATASET_ROOT / row.relative_path
    if not image_path.exists():
        missing_images.append(row.relative_path)
    elif sha256_file(image_path) != row.sha256:
        changed_images.append(row.relative_path)
if missing_images or changed_images:
    raise RuntimeError(
        f"Target image integrity failed: missing={len(missing_images)}, changed={len(changed_images)}"
    )
print("Manifest lock verified:", actual_digest)
print("All 260 sampled image hashes verified.")
print("Target images:", TARGET_DATASET_ROOT)"""),
        code("""from src.annotations import (
    annotation_agreement,
    build_adjudication_queue,
    finalize_adjudication,
    validate_annotation_sheet,
)

first = validate_annotation_sheet(
    pd.read_csv(BENCHMARK_PACKET_DIR / "annotations_annotator_a.csv", keep_default_na=False),
    manifest, ontology, require_complete=True,
)
second = validate_annotation_sheet(
    pd.read_csv(BENCHMARK_PACKET_DIR / "annotations_annotator_b.csv", keep_default_na=False),
    manifest, ontology, require_complete=True,
)
if first["annotator_id"].iloc[0] == second["annotator_id"].iloc[0]:
    raise RuntimeError("Two distinct annotator IDs are required")

agreement = annotation_agreement(first, second, ontology)
write_json(RUN_DIR / "annotation_agreement.json", agreement)
print(json.dumps(agreement, indent=2, sort_keys=True))

adjudication_path = RUN_DIR / "adjudication_queue.csv"
if adjudication_path.exists():
    adjudication_queue = pd.read_csv(adjudication_path, keep_default_na=False)
else:
    adjudication_queue = build_adjudication_queue(first, second, manifest)
    adjudication_queue.to_csv(adjudication_path, index=False)

print("Rows requiring human adjudication:", int(adjudication_queue["status"].eq("needs_adjudication").sum()))
print("Use the blinded interface below; do not inspect food_class or split while adjudicating.")"""),
        markdown("""## Adjudicate annotation disagreements

Exact agreements are accepted, while every disagreement requires a human resolution and rationale. An automatic union would inflate recall targets, and an automatic intersection would remove difficult labels. Neither rule is neutral."""),
        code("""# Run only when disagreements exist. The interface saves after each resolution.
from src.annotation_ui import AdjudicationApp

pending_adjudication = (
    adjudication_queue["status"].eq("needs_adjudication")
    & adjudication_queue["resolution_notes"].fillna("").astype(str).str.strip().eq("")
)
if pending_adjudication.any():
    adjudication_app = AdjudicationApp(
        queue=adjudication_queue,
        ontology=ontology,
        image_root=TARGET_DATASET_ROOT,
        output_csv=adjudication_path,
    )
    adjudication_app.display()
else:
    print("No unresolved disagreements require adjudication.")"""),
        code("""# Rerun after the interface is complete. This is a hard gate.
adjudication_queue = pd.read_csv(adjudication_path, keep_default_na=False)
final_annotations = finalize_adjudication(adjudication_queue, ontology)
if set(final_annotations["sample_id"]) != set(manifest["sample_id"]):
    raise RuntimeError("Adjudicated annotations do not match the sealed manifest")
final_annotation_path = RUN_DIR / "final_annotations.csv"
final_annotations.to_csv(final_annotation_path, index=False)
print("Final adjudicated annotations:", len(final_annotations))
print("Valid all-negative rows:", int(final_annotations["no_visible_ontology_label"].sum()))"""),
        markdown("""## Prepare optional recipe-presence pretraining

Recipe data is used only for optional weak pretraining. Phrases map into the frozen ontology, related recipes are grouped, and validation or test words cannot create new labels. All target systems predict the same 43 labels.

Recipe labels indicate recipe presence rather than visibility, so they can initialize features but cannot serve as primary benchmark truth."""),
        code("""from torch.utils.data import DataLoader

from src.datasets import MultiLabelImageDataset, build_transforms
from src.recipe_labels import (
    grouped_iterative_split,
    labels_to_matrix,
    merge_group_constraints,
    prepare_recipe_dataframe,
)

image_size = config["training"]["image_size"]
train_transform, evaluation_transform = build_transforms(image_size)

excluded_annotations = final_annotations[final_annotations["unreadable"] | final_annotations["non_food"]].copy()
target = final_annotations[~final_annotations["unreadable"] & ~final_annotations["non_food"]].copy()
target["labels"] = target["visible_ingredients"]
target["image_path"] = target["relative_path"].map(lambda path: str(TARGET_DATASET_ROOT / path))
target_train = target[target["split"].eq("train")].reset_index(drop=True)
target_validation = target[target["split"].eq("validation")].reset_index(drop=True)
target_test = target[target["split"].eq("test")].reset_index(drop=True)

print({name: len(frame) for name, frame in {
    "train": target_train, "validation": target_validation, "test_sealed": target_test
}.items()})
print("Human-excluded unreadable/non-food rows:", len(excluded_annotations))

def make_loader(frame, transform, shuffle=False):
    generator = torch.Generator().manual_seed(config["project"]["seed"])
    return DataLoader(
        MultiLabelImageDataset(frame, ontology, transform=transform),
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=2,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
        worker_init_fn=seed_worker,
    )

target_train_loader = make_loader(target_train, train_transform, shuffle=True)
target_validation_loader = make_loader(target_validation, evaluation_transform)
target_test_loader = make_loader(target_test, evaluation_transform)  # not iterated until sealed gate opens"""),
        code("""# Optional but recommended: canonical weak recipe-presence pretraining.
RUN_WEAK_RECIPE_PRETRAINING = True
recipe_pretraining_frame = None
recipe_split_digest = None
recipe_overlap_digest = None

if RUN_WEAK_RECIPE_PRETRAINING:
    recipe_cfg = config["datasets"]["recipe_pretraining"]
    explicit_recipe = os.environ.get(recipe_cfg["path_env"])
    RECIPE_ROOT = Path(explicit_recipe) if explicit_recipe else Path(kagglehub.dataset_download(recipe_cfg["slug"]))
    csv_candidates = sorted(RECIPE_ROOT.rglob("*.csv"))
    if not csv_candidates:
        raise RuntimeError("Could not resolve a recipe CSV")
    recipe_csv = max(csv_candidates, key=lambda path: path.stat().st_size)
    recipe_image_root = RECIPE_ROOT
    recipe_pretraining_frame = prepare_recipe_dataframe(recipe_csv, recipe_image_root, ontology)

    # Cross-dataset overlap control: quarantine recipe images that copy a sealed target image.
    # Recipe grouping then honors both normalized title and exact/near duplicate components.
    recipe_inventory = assign_duplicate_groups(
        build_image_inventory(recipe_image_root),
        perceptual_max_distance=config["benchmark"]["perceptual_hash_max_distance"],
    )
    recipe_inventory["image_path"] = recipe_inventory["image_path"].map(lambda value: str(Path(value).resolve()))
    recipe_pretraining_frame["image_path"] = recipe_pretraining_frame["image_path"].map(
        lambda value: str(Path(value).resolve())
    )
    recipe_pretraining_frame = recipe_pretraining_frame.merge(
        recipe_inventory[["image_path", "status", "sha256", "dhash", "duplicate_group"]],
        on="image_path", how="left", validate="many_to_one",
    )
    if recipe_pretraining_frame["status"].isna().any():
        raise RuntimeError("Recipe image integrity inventory did not cover every resolved image")
    corrupt_recipe = recipe_pretraining_frame[recipe_pretraining_frame["status"].ne("ok")].copy()
    corrupt_recipe.to_csv(RUN_DIR / "recipe_corrupt_images.csv", index=False)
    recipe_pretraining_frame = recipe_pretraining_frame[recipe_pretraining_frame["status"].eq("ok")].copy()
    overlap_report = cross_dataset_overlap_report(
        recipe_pretraining_frame,
        manifest,
        perceptual_max_distance=config["benchmark"]["perceptual_hash_max_distance"],
    )
    overlap_path = RUN_DIR / "recipe_target_overlap_quarantine.csv"
    overlap_report.to_csv(overlap_path, index=False)
    recipe_overlap_digest = sha256_file(overlap_path)
    leaking_paths = set(overlap_report["image_path"])
    recipe_pretraining_frame = recipe_pretraining_frame[
        ~recipe_pretraining_frame["image_path"].isin(leaking_paths)
    ].reset_index(drop=True)
    recipe_pretraining_frame = merge_group_constraints(
        recipe_pretraining_frame, ("group_id", "duplicate_group")
    )
    recipe_fractions = config["training"]["recipe_split_fractions"]
    recipe_pretraining_frame = grouped_iterative_split(
        recipe_pretraining_frame,
        ontology,
        seed=config["project"]["seed"],
        train_fraction=recipe_fractions["train"],
        validation_fraction=recipe_fractions["validation"],
        test_fraction=recipe_fractions["test"],
    )
    recipe_split_export = recipe_pretraining_frame[
        ["Image_Name", "sha256", "dhash", "duplicate_group", "group_id", "labels", "split"]
    ].copy()
    recipe_split_export["labels"] = recipe_split_export["labels"].map(ontology.serialize)
    recipe_split_path = RUN_DIR / "recipe_pretraining_split.csv"
    recipe_split_export.sort_values(["group_id", "Image_Name"]).to_csv(recipe_split_path, index=False)
    recipe_split_digest = sha256_file(recipe_split_path)
    print("Quarantined cross-dataset image overlaps:", len(overlap_report))
    print("Rejected corrupt recipe images:", len(corrupt_recipe))
    print(recipe_pretraining_frame["split"].value_counts())
    print("Canonical label coverage:", recipe_pretraining_frame["labels"].explode().nunique(), "/", len(ontology.ids))"""),
        markdown("""## Train prevalence and CNN baselines

The comparison includes a prevalence top-*k* baseline, an ImageNet frozen-backbone linear probe, and a last-block model that can optionally start from weak recipe pretraining. Checkpoints and global thresholds are chosen using validation only.

The prevalence result gives the learned models a meaningful floor and shows how much performance can be explained by common labels alone."""),
        code("""import torch.nn as nn

from src.models import build_resnet18
from src.recipe_labels import labels_to_matrix
from src.training import compute_pos_weight, fit_model

target_train_y = labels_to_matrix(
    target_train["visible_ingredients"].map(ontology.parse_annotation_cell), ontology
)
criterion_target = nn.BCEWithLogitsLoss(
    pos_weight=compute_pos_weight(
        target_train_y, maximum=config["training"]["maximum_positive_class_weight"]
    ).to(DEVICE)
)
threshold_grid = config["training"]["global_threshold_grid"]

pretrained_state = None
if recipe_pretraining_frame is not None:
    recipe_train = recipe_pretraining_frame[recipe_pretraining_frame["split"].eq("train")].reset_index(drop=True)
    recipe_validation = recipe_pretraining_frame[recipe_pretraining_frame["split"].eq("validation")].reset_index(drop=True)
    recipe_train_loader = make_loader(recipe_train, train_transform, shuffle=True)
    recipe_validation_loader = make_loader(recipe_validation, evaluation_transform)
    recipe_train_y = labels_to_matrix(recipe_train["labels"], ontology)
    weak_model = build_resnet18(len(ontology.ids), pretrained=True, trainable_scope="layer4_and_head").to(DEVICE)
    weak_criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(
            recipe_train_y, maximum=config["training"]["maximum_positive_class_weight"]
        ).to(DEVICE)
    )
    weak_optimizer = torch.optim.AdamW(
        [parameter for parameter in weak_model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"],
    )
    weak_fit = fit_model(
        weak_model, recipe_train_loader, recipe_validation_loader, weak_optimizer, weak_criterion,
        DEVICE, config["training"]["pretraining_epochs"], threshold_grid,
        RUN_DIR / "weak_recipe_pretraining.pt",
        {
            "role": "weak_recipe_presence_initialization",
            "ontology_version": ontology.version,
            "ontology_sha256": sha256_file(ontology_path),
            "recipe_split_sha256": recipe_split_digest,
        },
    )
    pretrained_state = torch.load(RUN_DIR / "weak_recipe_pretraining.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    del weak_model, weak_optimizer, weak_criterion
    gc.collect()
    torch.cuda.empty_cache()

linear_model = build_resnet18(len(ontology.ids), pretrained=True, trainable_scope="head").to(DEVICE)
linear_optimizer = torch.optim.AdamW(
    [parameter for parameter in linear_model.parameters() if parameter.requires_grad],
    lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"],
)
linear_fit = fit_model(
    linear_model, make_loader(target_train, train_transform, shuffle=True), target_validation_loader,
    linear_optimizer, criterion_target,
    DEVICE, config["training"]["finetuning_epochs"], threshold_grid,
    RUN_DIR / "linear_probe.pt",
    {
        "system": "resnet18_linear_probe",
        "ontology_version": ontology.version,
        "ontology_sha256": sha256_file(ontology_path),
        "target_manifest_sha256": actual_digest,
        "final_annotations_sha256": sha256_file(final_annotation_path),
    },
)

adapted_model = build_resnet18(len(ontology.ids), pretrained=True, trainable_scope="layer4_and_head").to(DEVICE)
if pretrained_state is not None:
    adapted_model.load_state_dict(pretrained_state)
adapted_system_name = (
    "Weak-pretrained ResNet18" if pretrained_state is not None
    else "ResNet18 last-block fine-tune"
)
adapted_optimizer = torch.optim.AdamW(
    [parameter for parameter in adapted_model.parameters() if parameter.requires_grad],
    lr=(
        config["training"]["learning_rate"]
        * config["training"]["adapted_learning_rate_multiplier"]
    ),
    weight_decay=config["training"]["weight_decay"],
)
adapted_fit = fit_model(
    adapted_model, make_loader(target_train, train_transform, shuffle=True), target_validation_loader,
    adapted_optimizer, criterion_target,
    DEVICE, config["training"]["finetuning_epochs"], threshold_grid,
    RUN_DIR / "adapted_resnet18.pt",
    {
        "system": adapted_system_name,
        "ontology_version": ontology.version,
        "ontology_sha256": sha256_file(ontology_path),
        "target_manifest_sha256": actual_digest,
        "final_annotations_sha256": sha256_file(final_annotation_path),
        "recipe_split_sha256": recipe_split_digest,
    },
)

print("Linear validation-selected threshold:", linear_fit["best_threshold"])
print("Adapted validation-selected threshold:", adapted_fit["best_threshold"])

# Release training graphs and optimizer state before loading the VLM on a Kaggle GPU.
del linear_model, adapted_model, linear_optimizer, adapted_optimizer, criterion_target
if pretrained_state is not None:
    del pretrained_state
gc.collect()
torch.cuda.empty_cache()"""),
        markdown("""## Run the frozen zero-shot VLM

A frozen zero-shot prompt lists the common ontology and asks only for visible evidence. It contains no examples from evaluated dishes. Unknown labels and malformed JSON are counted instead of being retroactively aliased after evaluation."""),
        code("""from src.vlm import PROMPT_VERSION
from src.vlm_inference import QwenVisibleIngredientAnalyzer

vlm_cfg = config["vlm"]
if vlm_cfg["prompt_version"] != PROMPT_VERSION:
    raise RuntimeError("Config and implemented VLM prompt versions disagree")
vlm_analyzer = QwenVisibleIngredientAnalyzer(
    ontology=ontology,
    model_id=vlm_cfg["model_id"],
    revision=vlm_cfg["revision"],
    max_image_side=vlm_cfg["max_image_side"],
)
prompt_path = RUN_DIR / "vlm_prompt.txt"
prompt_path.write_text(vlm_analyzer.prompt, encoding="utf-8")
vlm_metadata = {
    "model_id": vlm_cfg["model_id"],
    "model_revision": vlm_cfg["revision"],
    "prompt_version": vlm_cfg["prompt_version"],
    "prompt_sha256": hashlib.sha256(vlm_analyzer.prompt.encode("utf-8")).hexdigest(),
    "ontology_version": ontology.version,
    "ontology_sha256": sha256_file(ontology_path),
}
write_json(RUN_DIR / "vlm_metadata.json", vlm_metadata)

def run_vlm(frame, output_path):
    current_prompt_sha256 = hashlib.sha256(vlm_analyzer.prompt.encode("utf-8")).hexdigest()
    if current_prompt_sha256 != vlm_metadata["prompt_sha256"]:
        raise RuntimeError("In-memory VLM prompt changed after metadata was frozen")
    required_columns = {
        "sample_id", "food_class", "raw_output", "food_name", "visible_ingredients",
        "uncertain_ingredients", "unknown_labels", "abstain", "reason", "parse_ok", "parse_error",
        *vlm_metadata,
    }
    frame = frame.reset_index(drop=True)
    expected_ids = frame["sample_id"].tolist()
    rows = []
    if output_path.exists():
        output = pd.read_csv(output_path, keep_default_na=False)
        missing_columns = required_columns.difference(output.columns)
        if missing_columns:
            raise RuntimeError(f"Stale VLM cache lacks columns: {sorted(missing_columns)}")
        if output["sample_id"].duplicated().any() or not set(output["sample_id"]).issubset(expected_ids):
            raise RuntimeError("VLM cache contains duplicate or unexpected sample IDs")
        expected_classes = frame.set_index("sample_id")["food_class"].to_dict()
        if any(expected_classes[row.sample_id] != row.food_class for row in output.itertuples(index=False)):
            raise RuntimeError("VLM cache food classes do not match requested samples")
        for field, expected in vlm_metadata.items():
            if not output[field].astype(str).eq(str(expected)).all():
                raise RuntimeError(f"VLM cache was produced with different {field}")
        rows = output.to_dict("records")
        if output["sample_id"].tolist() == expected_ids:
            return output
        print(f"Resuming VLM cache: {len(rows)}/{len(frame)} rows complete")
    completed_ids = {row["sample_id"] for row in rows}
    for row in frame.itertuples(index=False):
        if row.sample_id in completed_ids:
            continue
        result = vlm_analyzer.infer(row.image_path, max_new_tokens=vlm_cfg["max_new_tokens"])
        rows.append({
            "sample_id": row.sample_id,
            "food_class": row.food_class,
            **result,
            **vlm_metadata,
        })
        print(row.sample_id, result["parse_ok"], result["visible_ingredients"])
        order = {sample_id: index for index, sample_id in enumerate(expected_ids)}
        output = pd.DataFrame(rows).sort_values("sample_id", key=lambda values: values.map(order))
        temporary = output_path.with_suffix(".tmp.csv")
        output.to_csv(temporary, index=False)
        temporary.replace(output_path)
    output = pd.read_csv(output_path, keep_default_na=False)
    if output["sample_id"].tolist() != expected_ids:
        raise RuntimeError("Completed VLM cache order/membership differs from requested split")
    return output

vlm_validation = run_vlm(target_validation, RUN_DIR / "vlm_validation_outputs.csv")
print("Validation parse success:", vlm_validation["parse_ok"].mean())
print("Validation abstention:", vlm_validation["abstain"].mean())
display(vlm_validation.head())"""),
        markdown("""## Freeze validation decisions before test

Both CNN checkpoints and the VLM are scored on validation using the common ontology. The final CNN is selected by validation micro-average-precision, with ties favoring the simpler linear probe. The selected checkpoint, threshold, prompt hash, and validation sample digest are written to `validation_decisions.json`.

Persisting the selection rule prevents the test results from influencing which trained system is presented as final."""),
        code("""from src.metrics import multilabel_metrics, prevalence_baseline, threshold_probabilities
from src.models import build_resnet18
from src.training import predict_probabilities

validation_truth = labels_to_matrix(
    target_validation["visible_ingredients"].map(ontology.parse_annotation_cell), ontology
)
validation_prevalence_pred, validation_prevalence_prob = prevalence_baseline(
    target_train_y, len(target_validation)
)

linear_checkpoint = torch.load(RUN_DIR / "linear_probe.pt", map_location=DEVICE, weights_only=False)
expected_checkpoint_metadata = {
    "ontology_sha256": sha256_file(ontology_path),
    "target_manifest_sha256": actual_digest,
    "final_annotations_sha256": sha256_file(final_annotation_path),
}
for field, expected in expected_checkpoint_metadata.items():
    if linear_checkpoint["metadata"].get(field) != expected:
        raise RuntimeError(f"Linear checkpoint metadata mismatch: {field}")
if linear_checkpoint["best_threshold"] not in threshold_grid:
    raise RuntimeError("Linear checkpoint threshold is outside the frozen grid")
linear_model = build_resnet18(len(ontology.ids), pretrained=False, trainable_scope="head").to(DEVICE)
linear_model.load_state_dict(linear_checkpoint["model_state_dict"])
linear_validation_prob, linear_validation_truth, linear_validation_ids = predict_probabilities(
    linear_model, target_validation_loader, DEVICE
)
linear_validation_pred = threshold_probabilities(
    linear_validation_prob, linear_checkpoint["best_threshold"]
)

adapted_checkpoint = torch.load(RUN_DIR / "adapted_resnet18.pt", map_location=DEVICE, weights_only=False)
for field, expected in expected_checkpoint_metadata.items():
    if adapted_checkpoint["metadata"].get(field) != expected:
        raise RuntimeError(f"Adapted checkpoint metadata mismatch: {field}")
if adapted_checkpoint["best_threshold"] not in threshold_grid:
    raise RuntimeError("Adapted checkpoint threshold is outside the frozen grid")
adapted_model = build_resnet18(len(ontology.ids), pretrained=False, trainable_scope="layer4_and_head").to(DEVICE)
adapted_model.load_state_dict(adapted_checkpoint["model_state_dict"])
adapted_validation_prob, adapted_validation_truth, adapted_validation_ids = predict_probabilities(
    adapted_model, target_validation_loader, DEVICE
)
adapted_validation_pred = threshold_probabilities(
    adapted_validation_prob, adapted_checkpoint["best_threshold"]
)

if not np.array_equal(validation_truth, linear_validation_truth):
    raise RuntimeError("Linear validation target order mismatch")
if not np.array_equal(validation_truth, adapted_validation_truth):
    raise RuntimeError("Adapted validation target order mismatch")
if target_validation["sample_id"].tolist() != linear_validation_ids or linear_validation_ids != adapted_validation_ids:
    raise RuntimeError("CNN validation sample order mismatch")

vlm_validation = target_validation[["sample_id", "food_class"]].merge(
    vlm_validation, on=["sample_id", "food_class"], validate="one_to_one"
)
vlm_validation_pred = np.zeros_like(validation_truth)
for row_index, labels in enumerate(vlm_validation["visible_ingredients"]):
    for label in ontology.parse_annotation_cell(labels):
        vlm_validation_pred[row_index, ontology.ids.index(label)] = 1

validation_systems_raw = [
    ("Prevalence top-k", validation_prevalence_pred, validation_prevalence_prob),
    ("ResNet18 linear probe", linear_validation_pred, linear_validation_prob),
    (adapted_system_name, adapted_validation_pred, adapted_validation_prob),
    ("Qwen2-VL zero-shot", vlm_validation_pred, None),
]
validation_systems = [
    {
        "name": name,
        **multilabel_metrics(
            validation_truth,
            prediction,
            probability,
            precision_k=config["evaluation"]["precision_at_k"],
        ),
    }
    for name, prediction, probability in validation_systems_raw
]
cnn_validation_candidates = [
    {
        "name": "ResNet18 linear probe",
        "checkpoint": "linear_probe.pt",
        "checkpoint_sha256": sha256_file(RUN_DIR / "linear_probe.pt"),
        "threshold": float(linear_checkpoint["best_threshold"]),
        "micro_average_precision": float(validation_systems[1]["micro_average_precision"]),
    },
    {
        "name": adapted_system_name,
        "checkpoint": "adapted_resnet18.pt",
        "checkpoint_sha256": sha256_file(RUN_DIR / "adapted_resnet18.pt"),
        "threshold": float(adapted_checkpoint["best_threshold"]),
        "micro_average_precision": float(validation_systems[2]["micro_average_precision"]),
    },
]
selected_cnn = max(
    cnn_validation_candidates,
    key=lambda row: (
        row["micro_average_precision"],
        row["name"] == "ResNet18 linear probe",
    ),
)
validation_sample_digest = hashlib.sha256(
    "\\n".join(target_validation["sample_id"]).encode("utf-8")
).hexdigest()
validation_decisions = {
    "selection_rule": "highest validation micro_average_precision; ties favor linear probe",
    "selected_cnn": selected_cnn,
    "cnn_candidates": cnn_validation_candidates,
    "validation_sample_digest": validation_sample_digest,
    "vlm_validation_outputs_sha256": sha256_file(RUN_DIR / "vlm_validation_outputs.csv"),
    "ontology_version": ontology.version,
    "weak_recipe_pretraining_enabled": RUN_WEAK_RECIPE_PRETRAINING,
    **vlm_metadata,
}
write_json(RUN_DIR / "validation_decisions.json", validation_decisions)
np.savez_compressed(
    RUN_DIR / "validation_probabilities.npz",
    sample_ids=target_validation["sample_id"].to_numpy(),
    truth=validation_truth,
    linear=linear_validation_prob,
    adapted=adapted_validation_prob,
)
display(pd.DataFrame(validation_systems))
print("Frozen CNN choice:", selected_cnn)"""),
        markdown("""## Open the sealed test and publish results

Test evaluation requires explicit confirmation and validates every upstream artifact. Results are written once to `artifacts/metrics.json`, and `reports/RESULTS.md` is generated from that file. This keeps every published result tied to one machine-readable source.

Before changing the flag, confirm that you will not modify ontology, aliases, prompt, model selection, or thresholds after seeing test results."""),
        code("""OPEN_SEALED_TEST = False  # change to True once, only after all validation decisions are frozen

required_files = [
    final_annotation_path,
    RUN_DIR / "linear_probe.pt",
    RUN_DIR / "adapted_resnet18.pt",
    RUN_DIR / "vlm_validation_outputs.csv",
    RUN_DIR / "validation_decisions.json",
    RUN_DIR / "annotation_agreement.json",
]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise RuntimeError(f"Test gate blocked; missing artifacts: {missing_files}")
validation_decisions = json.loads((RUN_DIR / "validation_decisions.json").read_text(encoding="utf-8"))
if validation_decisions["prompt_sha256"] != vlm_metadata["prompt_sha256"]:
    raise RuntimeError("Prompt changed after validation decisions were frozen")
if hashlib.sha256(vlm_analyzer.prompt.encode("utf-8")).hexdigest() != vlm_metadata["prompt_sha256"]:
    raise RuntimeError("In-memory prompt changed after validation decisions were frozen")
if sha256_file(prompt_path) != vlm_metadata["prompt_sha256"]:
    raise RuntimeError("Saved prompt changed after validation decisions were frozen")
if validation_decisions["model_revision"] != vlm_cfg["revision"]:
    raise RuntimeError("VLM revision changed after validation decisions were frozen")
if validation_decisions["ontology_version"] != ontology.version:
    raise RuntimeError("Ontology changed after validation decisions were frozen")
if validation_decisions["ontology_sha256"] != sha256_file(ontology_path):
    raise RuntimeError("Ontology content changed after validation decisions were frozen")
expected_validation_digest = hashlib.sha256(
    "\\n".join(target_validation["sample_id"]).encode("utf-8")
).hexdigest()
if validation_decisions["validation_sample_digest"] != expected_validation_digest:
    raise RuntimeError("Validation membership changed after model selection")
if sha256_file(RUN_DIR / "vlm_validation_outputs.csv") != validation_decisions["vlm_validation_outputs_sha256"]:
    raise RuntimeError("VLM validation outputs changed after model selection")
for candidate in validation_decisions["cnn_candidates"]:
    checkpoint_path = RUN_DIR / candidate["checkpoint"]
    if sha256_file(checkpoint_path) != candidate["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint changed after validation: {checkpoint_path.name}")
if not OPEN_SEALED_TEST:
    raise RuntimeError(
        "SEALED TEST REMAINS CLOSED. Review validation outputs, freeze all choices, "
        "then set OPEN_SEALED_TEST=True exactly once."
    )"""),
        code("""from sklearn.metrics import f1_score

from src.artifacts import build_run_manifest, json_digest, render_results_markdown, write_json
from src.metrics import (
    bootstrap_grouped_difference,
    bootstrap_grouped_metric,
    multilabel_metrics,
    paired_permutation_test,
    per_label_metrics,
    prevalence_baseline,
    threshold_probabilities,
)
from src.training import predict_probabilities
from src.vlm import food_name_is_correct

test_truth = labels_to_matrix(target_test["visible_ingredients"].map(ontology.parse_annotation_cell), ontology)
test_groups = target_test["food_class"].to_numpy()

prevalence_pred, prevalence_prob = prevalence_baseline(target_train_y, len(target_test))

linear_checkpoint = torch.load(RUN_DIR / "linear_probe.pt", map_location=DEVICE, weights_only=False)
linear_model.load_state_dict(linear_checkpoint["model_state_dict"])
linear_prob, linear_truth_check, linear_ids = predict_probabilities(linear_model, target_test_loader, DEVICE)
linear_pred = threshold_probabilities(linear_prob, linear_checkpoint["best_threshold"])

adapted_checkpoint = torch.load(RUN_DIR / "adapted_resnet18.pt", map_location=DEVICE, weights_only=False)
adapted_model.load_state_dict(adapted_checkpoint["model_state_dict"])
adapted_prob, adapted_truth_check, adapted_ids = predict_probabilities(adapted_model, target_test_loader, DEVICE)
adapted_pred = threshold_probabilities(adapted_prob, adapted_checkpoint["best_threshold"])

assert np.array_equal(test_truth, linear_truth_check)
assert np.array_equal(test_truth, adapted_truth_check)
assert list(target_test["sample_id"]) == linear_ids == adapted_ids

vlm_test = run_vlm(target_test, RUN_DIR / "vlm_test_outputs.csv")
vlm_test = target_test[["sample_id", "food_class"]].merge(vlm_test, on=["sample_id", "food_class"], validate="one_to_one")
vlm_pred = np.zeros_like(test_truth)
for row_index, labels in enumerate(vlm_test["visible_ingredients"]):
    for label in ontology.parse_annotation_cell(labels):
        vlm_pred[row_index, ontology.ids.index(label)] = 1
systems_raw = [
    ("Prevalence top-k", prevalence_pred, prevalence_prob),
    ("ResNet18 linear probe", linear_pred, linear_prob),
    (adapted_system_name, adapted_pred, adapted_prob),
    ("Qwen2-VL zero-shot", vlm_pred, None),
]
systems = []
for name, prediction, probability in systems_raw:
    result = {
        "name": name,
        **multilabel_metrics(
            test_truth,
            prediction,
            probability,
            precision_k=config["evaluation"]["precision_at_k"],
        ),
    }
    interval = bootstrap_grouped_metric(
        test_truth, prediction, test_groups,
        iterations=config["evaluation"]["bootstrap_iterations"],
        confidence=config["evaluation"]["confidence_level"],
        seed=config["project"]["seed"],
    )
    result.update({"ci_lower": interval["ci_lower"], "ci_upper": interval["ci_upper"]})
    systems.append(result)

prediction_by_name = {name: prediction for name, prediction, _ in systems_raw}
selected_cnn_name = validation_decisions["selected_cnn"]["name"]
selected_cnn_pred = prediction_by_name[selected_cnn_name]
paired_test = paired_permutation_test(
    test_truth, selected_cnn_pred, vlm_pred, groups=test_groups,
    iterations=config["evaluation"]["permutation_iterations"],
    seed=config["project"]["seed"],
)
paired_interval = bootstrap_grouped_difference(
    test_truth, selected_cnn_pred, vlm_pred, test_groups,
    metric=lambda truth, prediction: float(
        f1_score(truth, prediction, average="samples", zero_division=1)
    ),
    iterations=config["evaluation"]["bootstrap_iterations"],
    confidence=config["evaluation"]["confidence_level"],
    seed=config["project"]["seed"],
)
food_name_accuracy = float(np.mean([
    food_name_is_correct(row.food_class, row.food_name)
    for row in vlm_test.itertuples(index=False)
]))

per_label_rows = []
for name, prediction, _ in systems_raw:
    per_label_rows.extend({"system": name, **row} for row in per_label_metrics(test_truth, prediction, ontology.ids))
pd.DataFrame(per_label_rows).to_csv(RUN_DIR / "per_label_metrics.csv", index=False)

prediction_rows = target_test[["sample_id", "food_class", "visible_ingredients"]].copy()
for name, prediction, _ in systems_raw:
    column = name.lower().replace(" ", "_").replace("-", "_")
    prediction_rows[column] = [ontology.serialize(np.array(ontology.ids)[row.astype(bool)]) for row in prediction]
prediction_rows.to_csv(RUN_DIR / "test_predictions.csv", index=False)
np.savez_compressed(
    RUN_DIR / "test_probabilities.npz",
    sample_ids=target_test["sample_id"].to_numpy(),
    truth=test_truth,
    prevalence=prevalence_prob,
    linear=linear_prob,
    adapted=adapted_prob,
)

dataset_digests = {
    "target_manifest": actual_digest,
    "final_annotations": sha256_file(final_annotation_path),
    "ontology": sha256_file(ontology_path),
}
if recipe_split_digest is not None:
    dataset_digests["recipe_pretraining_split"] = recipe_split_digest
if recipe_overlap_digest is not None:
    dataset_digests["recipe_target_overlap_quarantine"] = recipe_overlap_digest

run_manifest = build_run_manifest(
    config=config,
    project_root=PROJECT_ROOT,
    dataset_digests=dataset_digests,
    split_manifest_digest=actual_digest,
)
run_manifest["artifact_digests"] = {
    "annotation_agreement": sha256_file(RUN_DIR / "annotation_agreement.json"),
    "linear_checkpoint": sha256_file(RUN_DIR / "linear_probe.pt"),
    "adapted_checkpoint": sha256_file(RUN_DIR / "adapted_resnet18.pt"),
    "validation_decisions": sha256_file(RUN_DIR / "validation_decisions.json"),
    "vlm_metadata": sha256_file(RUN_DIR / "vlm_metadata.json"),
    "vlm_prompt": sha256_file(prompt_path),
    "vlm_validation_outputs": sha256_file(RUN_DIR / "vlm_validation_outputs.csv"),
    "vlm_test_outputs": sha256_file(RUN_DIR / "vlm_test_outputs.csv"),
    "validation_probabilities": sha256_file(RUN_DIR / "validation_probabilities.npz"),
    "test_probabilities": sha256_file(RUN_DIR / "test_probabilities.npz"),
    "test_predictions": sha256_file(RUN_DIR / "test_predictions.csv"),
    "per_label_metrics": sha256_file(RUN_DIR / "per_label_metrics.csv"),
}
run_manifest["run_options"] = {
    "weak_recipe_pretraining_enabled": RUN_WEAK_RECIPE_PRETRAINING,
    "device_type": DEVICE.type,
}
write_json(RUN_DIR / "run_manifest.json", run_manifest)

metrics_artifact = {
    "status": "complete",
    "benchmark_rows_annotated": len(final_annotations),
    "benchmark_rows_evaluable": len(target),
    "benchmark_rows_excluded_unreadable_or_non_food": len(excluded_annotations),
    "benchmark_rows_valid_all_negative": int(final_annotations["no_visible_ontology_label"].sum()),
    "test_rows": len(target_test),
    "ontology_version": ontology.version,
    "prompt_version": vlm_cfg["prompt_version"],
    "weak_recipe_pretraining_enabled": RUN_WEAK_RECIPE_PRETRAINING,
    "selected_cnn_from_validation": selected_cnn_name,
    "annotation_agreement": agreement,
    "validation_systems": validation_systems,
    "systems": systems,
    "vlm_minus_selected_cnn_paired_permutation": paired_test,
    "vlm_minus_selected_cnn_grouped_bootstrap": paired_interval,
    "vlm_food_name_accuracy_separate_task": food_name_accuracy,
    "vlm_parse_success": float(vlm_test["parse_ok"].mean()),
    "vlm_abstention_rate": float(vlm_test["abstain"].mean()),
    "vlm_unknown_label_violation_rate": float(vlm_test["unknown_labels"].astype(str).str.strip().ne("").mean()),
    "run_manifest_digest": json_digest(run_manifest),
}
write_json(RUN_DIR / "metrics.json", metrics_artifact)
(RUN_DIR / "RESULTS.md").write_text(render_results_markdown(metrics_artifact) + "\\n", encoding="utf-8")

# Create a lightweight overlay containing only GitHub-safe
# reports. Raw images, checkpoints, VLM text, and per-image predictions stay out.
publication_root = RUN_DIR / "github-publication"
(publication_root / "artifacts").mkdir(parents=True, exist_ok=True)
(publication_root / "reports").mkdir(parents=True, exist_ok=True)
shutil.copy2(RUN_DIR / "metrics.json", publication_root / "artifacts/metrics.json")
shutil.copy2(RUN_DIR / "RESULTS.md", publication_root / "reports/RESULTS.md")
shutil.copy2(RUN_DIR / "per_label_metrics.csv", publication_root / "reports/per_label_metrics.csv")
shutil.copy2(RUN_DIR / "run_manifest.json", publication_root / "reports/RUN_MANIFEST.json")
shutil.copy2(RUN_DIR / "validation_decisions.json", publication_root / "reports/VALIDATION_DECISIONS.json")
publication_archive = shutil.make_archive(
    str(RUN_DIR / "github-publication-overlay"), "zip", publication_root
)

display(pd.DataFrame(systems))
print("VLM food-name accuracy (separate task):", food_name_accuracy)
print("Selected CNN from validation:", selected_cnn_name)
print("VLM minus selected-CNN paired test:", paired_test)
print("VLM minus selected-CNN grouped interval:", paired_interval)
print("Published artifacts:", RUN_DIR)
print("GitHub-safe overlay:", publication_archive)"""),
        markdown("""## Nutrition boundary

The primary pipeline stops at visible-component recognition. A future nutrition task must use standardized recipes, ingredient masses, and cooked yield. `src/nutrition.py` rejects per-100g records whose macros exceed 100 g, energy exceeds physical bounds, or stated kcal contradict the Atwater calculation.

## Reproducibility and handoff

The run directory now contains raw VLM outputs, parser states, model checkpoints, validation-selected thresholds, test predictions, annotation agreement, a run manifest, one metrics source, and a rendered results report. Preserve the entire directory as a Kaggle output and commit only lightweight reports permitted by dataset/model licenses."""),
    ]
)


def write_notebook(name: str, value: dict) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    write_notebook("01_build_benchmark.ipynb", NOTEBOOK_01)
    write_notebook("02_train_evaluate.ipynb", NOTEBOOK_02)
