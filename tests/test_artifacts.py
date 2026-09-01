from src.artifacts import render_results_markdown


def test_incomplete_metrics_never_render_scores():
    text = render_results_markdown({"status": "blocked_pending_human_annotations", "systems": []})
    assert "Final scores are intentionally unavailable" in text
    assert "Micro-F1" not in text


def test_complete_metrics_render_selection_and_paired_inference():
    metrics = {
        "status": "complete",
        "selected_cnn_from_validation": "ResNet18 linear probe",
        "systems": [
            {
                "name": "ResNet18 linear probe",
                "micro_f1": 0.5,
                "macro_f1_supported": 0.4,
                "micro_average_precision": 0.6,
                "sample_f1": 0.45,
                "micro_precision": 0.55,
                "micro_recall": 0.46,
                "exact_match": 0.2,
                "precision_at_5": 0.3,
                "ci_lower": 0.3,
                "ci_upper": 0.7,
            }
        ],
        "vlm_minus_selected_cnn_paired_permutation": {
            "mean_sample_f1_difference": 0.1,
            "p_value_two_sided": 0.05,
            "method": "group_clustered_exact_sign_flip",
        },
        "vlm_minus_selected_cnn_grouped_bootstrap": {"ci_lower": -0.02, "ci_upper": 0.2},
        "vlm_parse_success": 0.9,
        "vlm_abstention_rate": 0.1,
        "vlm_unknown_label_violation_rate": 0.02,
        "benchmark_rows_evaluable": 258,
        "test_rows": 51,
        "run_manifest_digest": "abc",
    }
    text = render_results_markdown(metrics)
    assert "Validation-selected CNN: **ResNet18 linear probe**" in text
    assert "two-sided p=0.05" in text
    assert "parse success 0.9000" in text
