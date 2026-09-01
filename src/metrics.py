"""Multi-label metrics, baselines, uncertainty, and paired tests.

Reported test metrics include label support, confidence intervals, and paired
comparisons instead of relying on one optimized micro-F1 value.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


def threshold_probabilities(probabilities: np.ndarray, thresholds: float | np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    return (probabilities >= np.asarray(thresholds)).astype(np.int8)


def precision_at_k(y_true: np.ndarray, probabilities: np.ndarray, k: int = 5) -> float:
    k = min(k, probabilities.shape[1])
    top_indices = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    hits = [y_true[row, top_indices[row]].sum() / k for row in range(len(y_true))]
    return float(np.mean(hits))


def multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    precision_k: int = 5,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be equal-shape 2D arrays")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_true and y_pred must be binary")
    y_true = y_true.astype(np.int8, copy=False)
    y_pred = y_pred.astype(np.int8, copy=False)
    support_mask = y_true.sum(axis=0) > 0
    result: dict[str, float | int] = {
        "samples": int(y_true.shape[0]),
        "labels": int(y_true.shape[1]),
        "labels_with_test_support": int(support_mask.sum()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1_supported": float(
            f1_score(y_true[:, support_mask], y_pred[:, support_mask], average="macro", zero_division=0)
            if support_mask.any()
            else 0.0
        ),
        # Correct all-negative images receive sample-F1=1; false positives on
        # those rows still receive zero. Exact-match is reported separately.
        "sample_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=1)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != y_true.shape:
            raise ValueError("probabilities must match y_true shape")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("probabilities must be finite values in [0, 1]")
        result["micro_average_precision"] = float(average_precision_score(y_true, probabilities, average="micro"))
        result["macro_average_precision_supported"] = float(
            average_precision_score(y_true[:, support_mask], probabilities[:, support_mask], average="macro")
            if support_mask.any()
            else 0.0
        )
        result[f"precision_at_{precision_k}"] = precision_at_k(y_true, probabilities, k=precision_k)
    return result


def per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    if y_true.shape != y_pred.shape or y_true.shape[1] != len(label_names):
        raise ValueError("Shapes and label_names do not align")
    rows = []
    for index, label in enumerate(label_names):
        rows.append(
            {
                "label": label,
                "support": int(y_true[:, index].sum()),
                "predicted_positive": int(y_pred[:, index].sum()),
                "precision": float(precision_score(y_true[:, index], y_pred[:, index], zero_division=0)),
                "recall": float(recall_score(y_true[:, index], y_pred[:, index], zero_division=0)),
                "f1": float(f1_score(y_true[:, index], y_pred[:, index], zero_division=0)),
            }
        )
    return rows


def prevalence_baseline(
    train_y: np.ndarray, evaluation_rows: int, top_k: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    train_y = np.asarray(train_y, dtype=np.int8)
    if train_y.ndim != 2 or len(train_y) == 0:
        raise ValueError("train_y must be a non-empty 2D array")
    if evaluation_rows < 0:
        raise ValueError("evaluation_rows cannot be negative")
    prevalence = train_y.mean(axis=0)
    if top_k is None:
        cardinality = np.median(train_y.sum(axis=1))
        top_k = max(0, int(round(float(cardinality))))
    if not 0 <= top_k <= train_y.shape[1]:
        raise ValueError("top_k must be between zero and the number of labels")
    selected = np.argsort(-prevalence)[:top_k]
    prediction = np.zeros((evaluation_rows, train_y.shape[1]), dtype=np.int8)
    prediction[:, selected] = 1
    probabilities = np.tile(prevalence, (evaluation_rows, 1))
    return prediction, probabilities


def bootstrap_grouped_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap by food class; positive values favor model B."""
    metric = metric or (lambda truth, pred: float(f1_score(truth, pred, average="micro", zero_division=0)))
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        differences.append(
            metric(y_true[sampled_indices], pred_b[sampled_indices])
            - metric(y_true[sampled_indices], pred_a[sampled_indices])
        )
    alpha = 1.0 - confidence
    lower, upper = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    return {
        "observed_difference": metric(y_true, pred_b) - metric(y_true, pred_a),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "iterations": iterations,
    }


def bootstrap_grouped_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Food-class-grouped bootstrap interval for one paired prediction set."""
    metric = metric or (lambda truth, pred: float(f1_score(truth, pred, average="micro", zero_division=0)))
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        estimates[iteration] = metric(y_true[sampled_indices], y_pred[sampled_indices])
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": metric(y_true, y_pred),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "iterations": iterations,
    }


def paired_permutation_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray | None = None,
    iterations: int = 10000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Paired sign-flip test on sample-F1, optionally clustered by food class.

    When groups are supplied, every image in one food class receives the same
    random sign. This preserves the benchmark's within-class dependence instead
    of pretending that all images are independent experimental units.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if y_true.shape != pred_a.shape or y_true.shape != pred_b.shape or y_true.ndim != 2:
        raise ValueError("y_true, pred_a, and pred_b must be equal-shape 2D arrays")
    if len(y_true) == 0:
        raise ValueError("paired permutation test requires at least one row")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    sample_a = np.array([f1_score(t, p, zero_division=1) for t, p in zip(y_true, pred_a, strict=True)])
    sample_b = np.array([f1_score(t, p, zero_division=1) for t, p in zip(y_true, pred_b, strict=True)])
    differences = sample_b - sample_a
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    if groups is None:
        group_inverse = np.arange(len(differences))
        group_count = len(differences)
        method = "sample_level_monte_carlo_sign_flip"
    else:
        groups = np.asarray(groups)
        if len(groups) != len(differences):
            raise ValueError("groups must contain one value per row")
        _, group_inverse = np.unique(groups, return_inverse=True)
        group_count = int(group_inverse.max()) + 1
        method = "group_clustered_monte_carlo_sign_flip"
    exact_permutations = 2**group_count
    if exact_permutations <= iterations:
        effective_iterations = exact_permutations
        permuted = np.empty(effective_iterations, dtype=float)
        for permutation in range(effective_iterations):
            group_signs = np.array([1.0 if permutation & (1 << bit) else -1.0 for bit in range(group_count)])
            permuted[permutation] = np.mean(differences * group_signs[group_inverse])
        p_value = float(np.mean(np.abs(permuted) >= abs(observed)))
        method = method.replace("monte_carlo", "exact")
    else:
        effective_iterations = iterations
        permuted = np.empty(effective_iterations, dtype=float)
        for index in range(effective_iterations):
            group_signs = rng.choice([-1.0, 1.0], size=group_count)
            permuted[index] = np.mean(differences * group_signs[group_inverse])
        p_value = float((np.sum(np.abs(permuted) >= abs(observed)) + 1) / (effective_iterations + 1))
    return {
        "mean_sample_f1_difference": observed,
        "p_value_two_sided": p_value,
        "iterations": effective_iterations,
        "requested_iterations": iterations,
        "permutation_units": group_count,
        "method": method,
    }
