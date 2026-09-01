"""Reproducible, development-only web evidence collection.

Evidence is timestamped, hashed, URL-deduplicated, and aggregated using
independent source votes. It cannot be used as final test ground truth.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from urllib.parse import urlparse

import pandas as pd

from .ontology import IngredientOntology, normalize_text


def evidence_hash(title: str, snippet: str, url: str) -> str:
    payload = "\n".join([normalize_text(title), normalize_text(snippet), url.strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalized_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().strip(".")
    for prefix in ("www.", "m.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    return host


def collect_evidence(food_class: str, queries: list[str], max_results: int = 8, region: str = "id-id") -> pd.DataFrame:
    from ddgs import DDGS

    collected_at = datetime.now(UTC).isoformat()
    rows = []
    with DDGS() as search:
        for query in queries:
            for rank, result in enumerate(search.text(query, region=region, max_results=max_results), start=1):
                url = result.get("href", "")
                title = result.get("title", "")
                snippet = result.get("body", "")
                rows.append(
                    {
                        "food_class": food_class,
                        "query": query,
                        "rank": rank,
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "domain": normalized_domain(url),
                        "collected_at_utc": collected_at,
                        "evidence_hash": evidence_hash(title, snippet, url),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["query", "rank"]).drop_duplicates(["url", "evidence_hash"]).reset_index(drop=True)


def source_vote_labels(evidence: pd.DataFrame, ontology: IngredientOntology, minimum_sources: int = 2) -> pd.DataFrame:
    votes = []
    for row in evidence.itertuples(index=False):
        if not str(row.domain).strip():
            continue
        labels = ontology.extract_from_text(f"{row.title} {row.snippet}")
        for label in labels:
            votes.append({"food_class": row.food_class, "label": label, "domain": row.domain, "url": row.url})
    if not votes:
        return pd.DataFrame(columns=["food_class", "label", "independent_sources", "accepted"])
    vote_frame = pd.DataFrame(votes).drop_duplicates(["food_class", "label", "domain", "url"])
    summary = (
        vote_frame.groupby(["food_class", "label"])["domain"].nunique().rename("independent_sources").reset_index()
    )
    summary["accepted"] = summary["independent_sources"] >= minimum_sources
    return summary.sort_values(["food_class", "independent_sources", "label"], ascending=[True, False, True])
