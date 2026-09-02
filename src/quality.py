"""Human semantic-quality screening before benchmark membership is sealed."""

from __future__ import annotations

import pandas as pd

QUALITY_SCREEN_COLUMNS = [
    "candidate_id",
    "candidate_rank",
    "relative_path",
    "food_class",
    "decision",
    "rejection_reason",
    "notes",
]

REJECTION_REASON_LABELS = {
    "composite_or_multi_panel": "Collage or multi-panel image",
    "heavy_graphic_overlay": "Heavy text, cartoon, or graphic overlay",
    "multiple_unrelated_dishes": "Multiple unrelated dishes with no primary subject",
    "wrong_food_class": "Wrong source class",
    "non_food": "No assessable food",
    "unreadable_or_low_quality": "Unreadable or extremely low quality",
    "privacy_or_sensitive_content": "Visible personal or sensitive information",
    "other": "Other benchmark-quality problem",
}


def create_quality_screen(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "candidate_rank", "relative_path", "food_class"}
    if missing := required.difference(candidate_pool.columns):
        raise ValueError(f"Candidate pool missing columns: {sorted(missing)}")
    screen = candidate_pool[["candidate_id", "candidate_rank", "relative_path", "food_class"]].copy()
    screen["decision"] = ""
    screen["rejection_reason"] = ""
    screen["notes"] = ""
    return screen[QUALITY_SCREEN_COLUMNS]


def validate_quality_screen(
    screen: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    samples_per_class: int,
    require_complete: bool = False,
) -> pd.DataFrame:
    if missing := set(QUALITY_SCREEN_COLUMNS).difference(screen.columns):
        raise ValueError(f"Quality screen missing columns: {sorted(missing)}")
    if screen["candidate_id"].duplicated().any():
        raise ValueError("Quality screen contains duplicate candidate IDs")
    expected_ids = set(candidate_pool["candidate_id"].astype(str))
    observed_ids = set(screen["candidate_id"].astype(str))
    if observed_ids != expected_ids:
        raise ValueError(
            f"Quality-screen candidate mismatch: missing={len(expected_ids - observed_ids)}, "
            f"extra={len(observed_ids - expected_ids)}"
        )

    metadata_columns = ["candidate_id", "candidate_rank", "relative_path", "food_class"]
    expected_metadata = candidate_pool[metadata_columns].copy().sort_values("candidate_id").reset_index(drop=True)
    observed_metadata = screen[metadata_columns].copy().sort_values("candidate_id").reset_index(drop=True)
    expected_metadata["candidate_rank"] = pd.to_numeric(expected_metadata["candidate_rank"], errors="raise").astype(int)
    observed_metadata["candidate_rank"] = pd.to_numeric(observed_metadata["candidate_rank"], errors="raise").astype(int)
    for column in ["candidate_id", "relative_path", "food_class"]:
        expected_metadata[column] = expected_metadata[column].astype(str)
        observed_metadata[column] = observed_metadata[column].astype(str)
    if not expected_metadata.equals(observed_metadata):
        raise ValueError("Quality-screen candidate metadata differs from the frozen candidate pool")

    normalized = screen[QUALITY_SCREEN_COLUMNS].copy()
    normalized["candidate_rank"] = pd.to_numeric(normalized["candidate_rank"], errors="raise").astype(int)
    for column in ["candidate_id", "relative_path", "food_class", "decision", "rejection_reason", "notes"]:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    normalized["decision"] = normalized["decision"].str.lower()
    normalized["rejection_reason"] = normalized["rejection_reason"].str.lower()

    invalid_decisions = set(normalized["decision"]).difference({"", "accept", "reject"})
    if invalid_decisions:
        raise ValueError(f"Unknown quality-screen decisions: {sorted(invalid_decisions)}")
    accepted_with_reason = normalized["decision"].eq("accept") & normalized["rejection_reason"].ne("")
    if accepted_with_reason.any():
        raise ValueError("Accepted quality-screen rows cannot have a rejection reason")
    blank_with_reason = normalized["decision"].eq("") & normalized["rejection_reason"].ne("")
    if blank_with_reason.any():
        raise ValueError("Unreviewed quality-screen rows cannot have a rejection reason")
    rejected = normalized["decision"].eq("reject")
    invalid_reasons = set(normalized.loc[rejected, "rejection_reason"]).difference(REJECTION_REASON_LABELS)
    if invalid_reasons:
        raise ValueError(f"Unknown quality-screen rejection reasons: {sorted(invalid_reasons)}")
    if (rejected & normalized["rejection_reason"].eq("")).any():
        raise ValueError("Rejected quality-screen rows require a rejection reason")
    other_without_notes = rejected & normalized["rejection_reason"].eq("other") & normalized["notes"].eq("")
    if other_without_notes.any():
        raise ValueError("The 'other' rejection reason requires a note")

    expected_classes = sorted(candidate_pool["food_class"].astype(str).unique())
    accepted_counts = normalized.loc[normalized["decision"].eq("accept"), "food_class"].value_counts()
    excessive = {
        name: int(accepted_counts.get(name, 0))
        for name in expected_classes
        if accepted_counts.get(name, 0) > samples_per_class
    }
    if excessive:
        raise ValueError(f"Too many accepted candidates in classes: {excessive}")
    if require_complete:
        incomplete = {
            name: int(accepted_counts.get(name, 0))
            for name in expected_classes
            if accepted_counts.get(name, 0) != samples_per_class
        }
        if incomplete:
            raise ValueError(
                f"Quality screen is incomplete; accepted counts must equal {samples_per_class}: {incomplete}"
            )
    return normalized


def quality_screen_progress(
    screen: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    samples_per_class: int,
) -> dict[str, object]:
    normalized = validate_quality_screen(screen, candidate_pool, samples_per_class, require_complete=False)
    expected_classes = sorted(candidate_pool["food_class"].astype(str).unique())
    reviewed = normalized["decision"].ne("")
    accepted = normalized["decision"].eq("accept")
    rejected = normalized["decision"].eq("reject")
    class_counts = {
        name: {
            "accepted": int((accepted & normalized["food_class"].eq(name)).sum()),
            "rejected": int((rejected & normalized["food_class"].eq(name)).sum()),
            "needed": max(
                samples_per_class - int((accepted & normalized["food_class"].eq(name)).sum()),
                0,
            ),
        }
        for name in expected_classes
    }
    return {
        "reviewed": int(reviewed.sum()),
        "accepted": int(accepted.sum()),
        "rejected": int(rejected.sum()),
        "remaining_acceptances": int(sum(item["needed"] for item in class_counts.values())),
        "complete": all(item["needed"] == 0 for item in class_counts.values()),
        "class_counts": class_counts,
    }


def accepted_candidates_from_screen(
    screen: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    samples_per_class: int,
) -> pd.DataFrame:
    normalized = validate_quality_screen(screen, candidate_pool, samples_per_class, require_complete=True)
    accepted_ids = set(normalized.loc[normalized["decision"].eq("accept"), "candidate_id"])
    accepted = candidate_pool[candidate_pool["candidate_id"].astype(str).isin(accepted_ids)].copy()
    counts = accepted["food_class"].value_counts()
    if not counts.eq(samples_per_class).all() or len(counts) != candidate_pool["food_class"].nunique():
        raise ValueError("Accepted candidate merge did not preserve per-class counts")
    return accepted.sort_values(["food_class", "candidate_rank"]).reset_index(drop=True)
