from src.config import load_config


def test_frozen_config_has_260_rows_and_valid_splits():
    config = load_config()
    benchmark = config["benchmark"]
    assert benchmark["samples_per_class"] * len(benchmark["expected_classes"]) == 260
    assert sum(config["training"]["recipe_split_fractions"].values()) == 1.0
