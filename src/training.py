"""Deterministic training, validation-only tuning, and checkpoint handling."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from .metrics import multilabel_metrics, threshold_probabilities


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id: int) -> None:
    """Seed Python/NumPy from PyTorch's deterministic per-worker seed."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compute_pos_weight(targets: np.ndarray, maximum: float = 20.0):
    import torch

    targets = np.asarray(targets, dtype=np.float32)
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    weights = np.clip(negatives / np.maximum(positives, 1.0), 1.0, maximum)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    import torch.nn as nn

    model.train()
    # A "frozen" backbone must also keep BatchNorm running statistics frozen.
    # Calling model.train() alone would silently adapt those statistics even when
    # every backbone parameter has requires_grad=False.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
    total_loss = 0.0
    total_rows = 0
    for images, targets, _, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(images)
        total_rows += len(images)
    return total_loss / max(total_rows, 1)


def predict_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch

    model.eval()
    probabilities, targets, sample_ids = [], [], []
    with torch.inference_mode():
        for images, labels, batch_ids, _ in loader:
            logits = model(images.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.numpy().astype(np.int8))
            sample_ids.extend(batch_ids)
    return np.vstack(probabilities), np.vstack(targets), sample_ids


def tune_global_threshold(y_true: np.ndarray, probabilities: np.ndarray, grid: list[float]) -> dict[str, object]:
    rows = []
    for threshold in grid:
        predictions = threshold_probabilities(probabilities, threshold)
        metrics = multilabel_metrics(y_true, predictions, probabilities)
        rows.append({"threshold": float(threshold), **metrics})
    best = max(rows, key=lambda row: (row["micro_f1"], row["micro_precision"], row["threshold"]))
    return {"best_threshold": best["threshold"], "validation_rows": rows}


def fit_model(
    model,
    train_loader,
    validation_loader,
    optimizer,
    criterion,
    device,
    epochs: int,
    threshold_grid: list[float],
    checkpoint_path: str | Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Select epochs and threshold on validation only; never receives test data."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation_score = float("-inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        probabilities, targets, _ = predict_probabilities(model, validation_loader, device)
        validation_predictions = threshold_probabilities(probabilities, 0.5)
        validation_metrics = multilabel_metrics(targets, validation_predictions, probabilities)
        row = {"epoch": epoch, "train_loss": train_loss, **validation_metrics}
        history.append(row)
        validation_score = float(validation_metrics["micro_average_precision"])
        if not np.isfinite(validation_score):
            raise ValueError("Validation micro-average-precision is non-finite; inspect target support")
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "metadata": metadata}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_probabilities, validation_targets, _ = predict_probabilities(model, validation_loader, device)
    tuning = tune_global_threshold(validation_targets, validation_probabilities, threshold_grid)
    checkpoint["best_threshold"] = tuning["best_threshold"]
    torch.save(checkpoint, checkpoint_path)
    return {
        "history": history,
        "checkpoint_epoch": checkpoint["epoch"],
        "best_threshold": tuning["best_threshold"],
        "threshold_tuning": tuning["validation_rows"],
    }
