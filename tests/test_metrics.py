import numpy as np

from src.metrics import (
    bootstrap_grouped_difference,
    bootstrap_grouped_metric,
    multilabel_metrics,
    paired_permutation_test,
    prevalence_baseline,
)


def test_metrics_are_exact_for_perfect_prediction():
    truth = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    metrics = multilabel_metrics(truth, truth, truth.astype(float), precision_k=2)
    assert metrics["micro_f1"] == 1.0
    assert metrics["sample_f1"] == 1.0
    assert metrics["exact_match"] == 1.0


def test_prevalence_baseline_shape():
    train = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=int)
    predictions, probabilities = prevalence_baseline(train, evaluation_rows=4, top_k=1)
    assert predictions.shape == probabilities.shape == (4, 3)
    assert np.all(predictions[:, 0] == 1)


def test_sample_f1_credits_correct_all_negative_row():
    truth = np.array([[0, 0], [1, 0]], dtype=int)
    prediction = np.array([[0, 0], [1, 0]], dtype=int)
    metrics = multilabel_metrics(truth, prediction)
    assert metrics["sample_f1"] == 1.0


def test_paired_statistics_favor_better_model():
    truth = np.tile(np.array([[1, 0], [0, 1]], dtype=int), (10, 1))
    bad = np.zeros_like(truth)
    good = truth.copy()
    groups = np.repeat(["a", "b", "c", "d"], 5)
    interval = bootstrap_grouped_difference(truth, bad, good, groups, iterations=100, seed=7)
    single_interval = bootstrap_grouped_metric(truth, good, groups, iterations=100, seed=7)
    test = paired_permutation_test(truth, bad, good, groups=groups, iterations=200, seed=7)
    assert interval["observed_difference"] > 0
    assert interval["ci_lower"] > 0
    assert single_interval["estimate"] == 1.0
    assert test["mean_sample_f1_difference"] > 0
    assert test["permutation_units"] == 4
    assert test["method"] == "group_clustered_exact_sign_flip"
    assert test["iterations"] == 16
