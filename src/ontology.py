"""Canonical phrase-level ontology utilities.

The primary target is *visible components in one image*. Multiword ingredients
remain intact, aliases map to one canonical ID, and recipe modifiers are never
promoted to model classes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class IngredientLabel:
    id: str
    category: str
    aliases: tuple[str, ...]
    hint: str


class IngredientOntology:
    def __init__(self, labels: Iterable[IngredientLabel], version: str, scope: str):
        self.labels = tuple(labels)
        self.version = version
        self.scope = scope
        self.ids = tuple(label.id for label in self.labels)
        self._by_id = {label.id: label for label in self.labels}
        if len(self._by_id) != len(self.labels):
            raise ValueError("Ontology label IDs must be unique")

        alias_to_id: dict[str, str] = {}
        for label in self.labels:
            canonical_aliases = set(label.aliases) | {label.id, label.id.replace("_", " ")}
            for raw_alias in canonical_aliases:
                alias = normalize_text(raw_alias)
                previous = alias_to_id.get(alias)
                if previous is not None and previous != label.id:
                    raise ValueError(f"Alias collision: {alias!r} -> {previous!r}/{label.id!r}")
                alias_to_id[alias] = label.id
        self.alias_to_id = alias_to_id
        alternatives = sorted(alias_to_id, key=lambda item: (-len(item), item))
        self._pattern = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(x) for x in alternatives) + r")(?![a-z0-9])")

    @classmethod
    def from_json(cls, path: str | Path) -> IngredientOntology:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        labels = [
            IngredientLabel(
                id=item["id"],
                category=item["category"],
                aliases=tuple(item["aliases"]),
                hint=item["hint"],
            )
            for item in raw["labels"]
        ]
        return cls(labels, version=raw["version"], scope=raw["scope"])

    def extract_from_text(self, text: object) -> list[str]:
        """Extract canonical labels from free text without splitting phrases."""
        normalized = normalize_text(text)
        found = {self.alias_to_id[match.group(1)] for match in self._pattern.finditer(normalized)}
        return sorted(found, key=self.ids.index)

    def canonicalize_label(self, value: object) -> str | None:
        normalized = normalize_text(value)
        return self.alias_to_id.get(normalized)

    def parse_annotation_cell(self, value: object, strict: bool = True) -> list[str]:
        if value is None:
            return []
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        parts = [item.strip() for item in re.split(r"[|,;]", text) if item.strip()]
        canonical: list[str] = []
        unknown: list[str] = []
        for part in parts:
            label = self.canonicalize_label(part)
            if label is None:
                unknown.append(part)
            elif label not in canonical:
                canonical.append(label)
        if strict and unknown:
            raise ValueError(f"Unknown ontology labels: {unknown}")
        return sorted(canonical, key=self.ids.index)

    def serialize(self, labels: Iterable[str]) -> str:
        unique = set(labels)
        unknown = unique.difference(self._by_id)
        if unknown:
            raise ValueError(f"Cannot serialize unknown labels: {sorted(unknown)}")
        return "|".join(label for label in self.ids if label in unique)
