"""Single-source metrics and reproducibility manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_protocol_digest(project_root: str | Path) -> str:
    """Hash code/config files that define sampling, labels, training, and evaluation."""
    root = Path(project_root)
    paths = [
        *sorted((root / "src").glob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        root / "configs/project.json",
        root / "data/ontology/visible_ingredients.json",
        root / "requirements-kaggle.txt",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_versions(packages: list[str]) -> dict[str, str | None]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit(project_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty(project_root: str | Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_manifest(
    config: dict[str, Any],
    project_root: str | Path,
    dataset_digests: dict[str, str],
    split_manifest_digest: str,
) -> dict[str, Any]:
    torch_runtime: dict[str, Any] | None = None
    try:
        import torch

        torch_runtime = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        }
    except ImportError:
        pass
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(project_root),
        "git_dirty": git_is_dirty(project_root),
        "config_digest": json_digest(config),
        "project_protocol_digest": project_protocol_digest(project_root),
        "dataset_digests": dataset_digests,
        "split_manifest_digest": split_manifest_digest,
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "torch",
                "torchvision",
                "transformers",
                "qwen-vl-utils",
                "kagglehub",
                "iterative-stratification",
                "ipywidgets",
                "ddgs",
                "accelerate",
                "safetensors",
            ]
        ),
        "torch_runtime": torch_runtime,
    }


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def render_results_markdown(metrics: dict[str, Any]) -> str:
    status = metrics.get("status", "unknown")
    lines = ["# Reproducible Results", "", f"**Status:** `{status}`", ""]
    if status != "complete":
        lines.extend(
            [
                "Final scores are intentionally unavailable until the 260-image annotations are complete,",
                "all disagreements are adjudicated, and the sealed test gate is opened once.",
            ]
        )
        return "\n".join(lines)
    if not metrics.get("systems"):
        raise ValueError("Complete metrics must contain at least one evaluated system")
    lines.extend(
        [
            "| System | Micro-F1 | Supported macro-F1 | Micro-AP | Supported macro-AP | Micro-F1 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for system in metrics["systems"]:
        average_precision = system.get("micro_average_precision")
        average_precision_text = f"{average_precision:.4f}" if average_precision is not None else "N/A"
        macro_average_precision = system.get("macro_average_precision_supported")
        macro_average_precision_text = (
            f"{macro_average_precision:.4f}" if macro_average_precision is not None else "N/A"
        )
        lines.append(
            f"| {system['name']} | {system['micro_f1']:.4f} | "
            f"{system['macro_f1_supported']:.4f} | {average_precision_text} | "
            f"{macro_average_precision_text} | "
            f"[{system['ci_lower']:.4f}, {system['ci_upper']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "| System | Sample-F1 | Micro-precision | Micro-recall | Exact match | Precision@5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for system in metrics["systems"]:
        precision_at_five = system.get("precision_at_5")
        precision_at_five_text = f"{precision_at_five:.4f}" if precision_at_five is not None else "N/A"
        lines.append(
            f"| {system['name']} | {system['sample_f1']:.4f} | "
            f"{system['micro_precision']:.4f} | {system['micro_recall']:.4f} | "
            f"{system['exact_match']:.4f} | {precision_at_five_text} |"
        )
    selected_cnn = metrics.get("selected_cnn_from_validation")
    if selected_cnn:
        lines.extend(["", f"Validation-selected CNN: **{selected_cnn}**."])
    paired = metrics.get("vlm_minus_selected_cnn_paired_permutation")
    interval = metrics.get("vlm_minus_selected_cnn_grouped_bootstrap")
    if paired and interval:
        lines.extend(
            [
                "",
                "The paired effect is VLM minus the validation-selected CNN on sample-F1: "
                f"{paired['mean_sample_f1_difference']:.4f} "
                f"(grouped 95% CI [{interval['ci_lower']:.4f}, {interval['ci_upper']:.4f}], "
                f"two-sided p={paired['p_value_two_sided']:.4g}; {paired['method']}).",
            ]
        )
    agreement = metrics.get("annotation_agreement")
    if agreement:
        lines.extend(
            [
                "",
                "Annotation agreement before adjudication: "
                f"mean sample Jaccard {agreement['mean_sample_jaccard']:.4f}; "
                f"exact visible-set agreement {agreement['exact_set_agreement']:.4f}.",
            ]
        )
    if "vlm_parse_success" in metrics:
        lines.extend(
            [
                "",
                "VLM output quality: "
                f"parse success {metrics['vlm_parse_success']:.4f}; "
                f"abstention {metrics['vlm_abstention_rate']:.4f}; "
                f"unknown-label violation {metrics['vlm_unknown_label_violation_rate']:.4f}.",
            ]
        )
    if "vlm_food_name_accuracy_separate_task" in metrics:
        lines.extend(
            [
                "",
                "VLM food-name accuracy (separate dish-recognition outcome): "
                f"{metrics['vlm_food_name_accuracy_separate_task']:.4f}.",
            ]
        )
    lines.extend(
        [
            "",
            f"Evaluable benchmark rows: {metrics.get('benchmark_rows_evaluable', 'unknown')}; "
            f"sealed-test rows: {metrics.get('test_rows', 'unknown')}; "
            f"labels with test support: {metrics['systems'][0].get('labels_with_test_support', 'unknown')}.",
            "Excluded unreadable/non-food rows: "
            f"{metrics.get('benchmark_rows_excluded_unreadable_or_non_food', 'unknown')}; "
            "valid all-negative rows: "
            f"{metrics.get('benchmark_rows_valid_all_negative', 'unknown')}.",
            "",
            f"Run manifest: `{metrics.get('run_manifest_digest', 'missing')}`",
            "",
        ]
    )
    return "\n".join(lines)
