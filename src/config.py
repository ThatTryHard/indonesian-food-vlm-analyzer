"""Central project configuration and portable path resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "project.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    benchmark = config["benchmark"]
    split_total = benchmark["train_per_class"] + benchmark["validation_per_class"] + benchmark["test_per_class"]
    if split_total != benchmark["samples_per_class"]:
        raise ValueError("Target split counts must sum to samples_per_class")
    expected_classes = benchmark["expected_classes"]
    if len(expected_classes) != len(set(expected_classes)):
        raise ValueError("expected_classes contains duplicates")
    if benchmark["samples_per_class"] * len(expected_classes) != 260:
        raise ValueError("The frozen portfolio benchmark must contain exactly 260 images")
    if benchmark["required_annotators"] != 2:
        raise ValueError("This protocol requires exactly two independent annotation passes")

    training = config["training"]
    thresholds = training["global_threshold_grid"]
    if not thresholds or thresholds != sorted(set(thresholds)):
        raise ValueError("global_threshold_grid must be non-empty, unique, and sorted")
    if any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("global thresholds must lie strictly between zero and one")
    fractions = training["recipe_split_fractions"]
    if set(fractions) != {"train", "validation", "test"}:
        raise ValueError("recipe_split_fractions must define train/validation/test")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("recipe_split_fractions must sum to one")
    for field in ["pretraining_epochs", "finetuning_epochs", "batch_size", "image_size"]:
        if int(training[field]) <= 0:
            raise ValueError(f"training.{field} must be positive")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")

    vlm = config["vlm"]
    for field in ["model_id", "revision", "prompt_version"]:
        if not str(vlm[field]).strip():
            raise ValueError(f"vlm.{field} cannot be blank")
    evaluation = config["evaluation"]
    if not 0 < evaluation["confidence_level"] < 1:
        raise ValueError("evaluation.confidence_level must lie between zero and one")
    if evaluation["bootstrap_iterations"] < 1 or evaluation["permutation_iterations"] < 1:
        raise ValueError("Evaluation iteration counts must be positive")
    if evaluation["precision_at_k"] < 1:
        raise ValueError("evaluation.precision_at_k must be positive")


def artifact_dir(config: dict[str, Any] | None = None) -> Path:
    """Return a writable artifact directory for Kaggle or local execution."""
    config = config or load_config()
    override = os.environ.get("FOOD_VLM_ARTIFACT_DIR")
    if override:
        path = Path(override)
    elif Path("/kaggle/working").exists():
        path = Path("/kaggle/working") / config["project"]["artifact_subdir"]
    else:
        path = PROJECT_ROOT / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_root_from_env(dataset_key: str, config: dict[str, Any] | None = None) -> Path | None:
    """Resolve a dataset path only from an explicit environment variable.

    Automatic discovery lives in the notebook because ambiguous substring matching can
    select the wrong dataset. Explicit paths always win.
    """
    config = config or load_config()
    env_name = config["datasets"][dataset_key]["path_env"]
    value = os.environ.get(env_name)
    return Path(value) if value else None
