"""Small notebook UI for independent visible-ingredient annotation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from .annotations import annotation_progress, parse_boolean


class AnnotationApp:
    """One-image-at-a-time annotator that saves after every confirmed sample."""

    def __init__(self, sheet: pd.DataFrame, ontology, image_root: str | Path, output_csv: str | Path):
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError("ipywidgets is required for the annotation interface") from exc
        self.widgets = widgets
        self.sheet = sheet.copy().reset_index(drop=True)
        self.ontology = ontology
        self.image_root = Path(image_root)
        self.output_csv = Path(output_csv)
        completed = (
            self.sheet["visible_ingredients"].fillna("").astype(str).str.strip().ne("")
            | self.sheet["uncertain_ingredients"].fillna("").astype(str).str.strip().ne("")
            | self.sheet["unreadable"].fillna(False).map(parse_boolean)
            | self.sheet["non_food"].fillna(False).map(parse_boolean)
            | self.sheet["no_visible_ontology_label"].fillna(False).map(parse_boolean)
        )
        unfinished = self.sheet.index[~completed].tolist()
        self.position = unfinished[0] if unfinished else 0

        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="700px", max_height="520px"))
        options = [(f"{label.id}: {label.hint}", label.id) for label in ontology.labels]
        list_layout = widgets.Layout(width="49%", height="430px")
        self.visible = widgets.SelectMultiple(options=options, description="Visible", layout=list_layout)
        self.uncertain = widgets.SelectMultiple(options=options, description="Uncertain", layout=list_layout)
        self.unreadable = widgets.Checkbox(description="Unreadable image")
        self.non_food = widgets.Checkbox(description="Not assessable food")
        self.no_visible_label = widgets.Checkbox(description="No supported ontology label")
        self.notes = widgets.Textarea(description="Notes", layout=widgets.Layout(width="100%", height="70px"))
        self.status = widgets.HTML()
        self.message = widgets.HTML()
        self.previous_button = widgets.Button(description="Previous")
        self.save_button = widgets.Button(description="Save + Next", button_style="success")
        self.next_unfinished_button = widgets.Button(description="Next unfinished")
        self.previous_button.on_click(self._previous)
        self.save_button.on_click(self._save_and_next)
        self.next_unfinished_button.on_click(self._next_unfinished)
        self.container = widgets.VBox(
            [
                self.status,
                self.image,
                widgets.HBox([self.visible, self.uncertain]),
                widgets.HBox([self.unreadable, self.non_food, self.no_visible_label]),
                self.notes,
                widgets.HBox([self.previous_button, self.save_button, self.next_unfinished_button]),
                self.message,
            ]
        )
        self._load_position()

    def display(self):
        from IPython.display import display

        display(self.container)

    def _image_bytes(self, path: Path) -> bytes:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            image.thumbnail((900, 650))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()

    def _load_position(self) -> None:
        row = self.sheet.iloc[self.position]
        path = self.image_root / row["relative_path"]
        self.image.value = self._image_bytes(path)
        self.visible.value = tuple(self.ontology.parse_annotation_cell(row["visible_ingredients"]))
        self.uncertain.value = tuple(self.ontology.parse_annotation_cell(row["uncertain_ingredients"]))
        self.unreadable.value = parse_boolean(row["unreadable"])
        self.non_food.value = parse_boolean(row["non_food"])
        self.no_visible_label.value = parse_boolean(row["no_visible_ontology_label"])
        self.notes.value = str(row["notes"] or "")
        progress = annotation_progress(self.sheet)
        self.status.value = (
            f"<b>{self.position + 1}/{len(self.sheet)}</b> | "
            f"Sample {row['sample_id']}<br>"
            f"Completed {progress['completed']} · Remaining {progress['remaining']}"
        )
        self.message.value = ""

    def _save_current(self) -> bool:
        visible, uncertain = set(self.visible.value), set(self.uncertain.value)
        overlap = visible.intersection(uncertain)
        if overlap:
            self.message.value = f"<span style='color:red'>Remove overlap: {sorted(overlap)}</span>"
            return False
        completion_flags = [self.unreadable.value, self.non_food.value, self.no_visible_label.value]
        if sum(completion_flags) > 1:
            self.message.value = "<span style='color:red'>Choose at most one completion flag.</span>"
            return False
        if any(completion_flags) and (visible or uncertain):
            self.message.value = "<span style='color:red'>A flagged image cannot also carry labels.</span>"
            return False
        if not visible and not uncertain and not any(completion_flags):
            self.message.value = "<span style='color:red'>Add a label or choose a completion flag.</span>"
            return False
        index = self.position
        self.sheet.at[index, "visible_ingredients"] = self.ontology.serialize(visible)
        self.sheet.at[index, "uncertain_ingredients"] = self.ontology.serialize(uncertain)
        self.sheet.at[index, "unreadable"] = self.unreadable.value
        self.sheet.at[index, "non_food"] = self.non_food.value
        self.sheet.at[index, "no_visible_ontology_label"] = self.no_visible_label.value
        self.sheet.at[index, "notes"] = self.notes.value.strip()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_csv.with_suffix(".tmp.csv")
        self.sheet.to_csv(temporary, index=False)
        temporary.replace(self.output_csv)
        return True

    def _save_and_next(self, _button) -> None:
        if self._save_current():
            self.position = min(self.position + 1, len(self.sheet) - 1)
            self._load_position()

    def _previous(self, _button) -> None:
        self.position = max(0, self.position - 1)
        self._load_position()

    def _next_unfinished(self, _button) -> None:
        completed = (
            self.sheet["visible_ingredients"].fillna("").astype(str).str.strip().ne("")
            | self.sheet["uncertain_ingredients"].fillna("").astype(str).str.strip().ne("")
            | self.sheet["unreadable"].fillna(False).map(parse_boolean)
            | self.sheet["non_food"].fillna(False).map(parse_boolean)
            | self.sheet["no_visible_ontology_label"].fillna(False).map(parse_boolean)
        )
        unfinished = [index for index in range(self.position + 1, len(self.sheet)) if not completed.iloc[index]]
        if not unfinished:
            unfinished = [index for index in range(0, self.position) if not completed.iloc[index]]
        if unfinished:
            self.position = unfinished[0]
            self._load_position()
        else:
            self.message.value = "<b>All rows have an annotation. Download and preserve the CSV.</b>"


class AdjudicationApp:
    """Resolve only rows on which the two independent passes disagree."""

    def __init__(self, queue: pd.DataFrame, ontology, image_root: str | Path, output_csv: str | Path):
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError("ipywidgets is required for the adjudication interface") from exc
        self.widgets = widgets
        self.queue = queue.copy().reset_index(drop=True)
        self.indices = self.queue.index[self.queue["status"].eq("needs_adjudication")].tolist()
        if not self.indices:
            raise ValueError("No disagreements require adjudication")
        self.ontology = ontology
        self.image_root = Path(image_root)
        self.output_csv = Path(output_csv)
        pending_positions = [
            position
            for position, index in enumerate(self.indices)
            if not str(self.queue.at[index, "resolution_notes"]).strip()
        ]
        self.pointer = pending_positions[0] if pending_positions else 0
        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="700px", max_height="520px"))
        options = [(f"{label.id}: {label.hint}", label.id) for label in ontology.labels]
        layout = widgets.Layout(width="49%", height="430px")
        self.visible = widgets.SelectMultiple(options=options, description="Resolved visible", layout=layout)
        self.uncertain = widgets.SelectMultiple(options=options, description="Resolved uncertain", layout=layout)
        self.notes = widgets.Textarea(description="Rationale", layout=widgets.Layout(width="100%", height="70px"))
        self.unreadable = widgets.Checkbox(description="Resolved unreadable")
        self.non_food = widgets.Checkbox(description="Resolved non-food")
        self.no_visible_label = widgets.Checkbox(description="Resolved no supported label")
        self.status = widgets.HTML()
        self.comparison = widgets.HTML()
        self.message = widgets.HTML()
        self.previous_button = widgets.Button(description="Previous")
        self.save_button = widgets.Button(description="Resolve + Next", button_style="warning")
        self.previous_button.on_click(self._previous)
        self.save_button.on_click(self._save_and_next)
        self.container = widgets.VBox(
            [
                self.status,
                self.image,
                self.comparison,
                widgets.HBox([self.visible, self.uncertain]),
                widgets.HBox([self.unreadable, self.non_food, self.no_visible_label]),
                self.notes,
                widgets.HBox([self.previous_button, self.save_button]),
                self.message,
            ]
        )
        self._load()

    def display(self):
        from IPython.display import display

        display(self.container)

    def _image_bytes(self, path: Path) -> bytes:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            image.thumbnail((900, 650))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()

    def _load(self) -> None:
        index = self.indices[self.pointer]
        row = self.queue.loc[index]
        self.image.value = self._image_bytes(self.image_root / row["relative_path"])
        has_resolution = bool(str(row["resolution_notes"]).strip())
        initial_visible = row["resolved_visible_ingredients"] if has_resolution else row["visible_ingredients_a"]
        initial_uncertain = row["resolved_uncertain_ingredients"] if has_resolution else row["uncertain_ingredients_a"]
        self.visible.value = tuple(self.ontology.parse_annotation_cell(initial_visible))
        self.uncertain.value = tuple(self.ontology.parse_annotation_cell(initial_uncertain))
        self.notes.value = str(row["resolution_notes"] or "")
        self.unreadable.value = parse_boolean(row["resolved_unreadable"] if has_resolution else row["unreadable_a"])
        self.non_food.value = parse_boolean(row["resolved_non_food"] if has_resolution else row["non_food_a"])
        self.no_visible_label.value = parse_boolean(
            row["resolved_no_visible_ontology_label"] if has_resolution else row["no_visible_ontology_label_a"]
        )
        self.status.value = f"<b>Disagreement {self.pointer + 1}/{len(self.indices)}</b> | sample {row['sample_id']}"
        self.comparison.value = (
            f"<b>Annotator A</b> visible: {row['visible_ingredients_a'] or '(none)'}; "
            f"uncertain: {row['uncertain_ingredients_a'] or '(none)'}; "
            f"unreadable={row['unreadable_a']}; non-food={row['non_food_a']}; "
            f"no-supported-label={row['no_visible_ontology_label_a']}<br>"
            f"<b>Annotator B</b> visible: {row['visible_ingredients_b'] or '(none)'}; "
            f"uncertain: {row['uncertain_ingredients_b'] or '(none)'}; "
            f"unreadable={row['unreadable_b']}; non-food={row['non_food_b']}; "
            f"no-supported-label={row['no_visible_ontology_label_b']}"
        )
        self.message.value = ""

    def _save_and_next(self, _button) -> None:
        visible, uncertain = set(self.visible.value), set(self.uncertain.value)
        if visible.intersection(uncertain):
            self.message.value = "<span style='color:red'>A label cannot be both visible and uncertain.</span>"
            return
        completion_flags = [self.unreadable.value, self.non_food.value, self.no_visible_label.value]
        if sum(completion_flags) > 1:
            self.message.value = "<span style='color:red'>Choose at most one completion flag.</span>"
            return
        if any(completion_flags) and (visible or uncertain):
            self.message.value = "<span style='color:red'>A flagged image cannot also carry labels.</span>"
            return
        if not visible and not uncertain and not any(completion_flags):
            self.message.value = "<span style='color:red'>Add a label or choose a completion flag.</span>"
            return
        if not self.notes.value.strip():
            self.message.value = "<span style='color:red'>Add a brief adjudication rationale.</span>"
            return
        index = self.indices[self.pointer]
        self.queue.at[index, "resolved_visible_ingredients"] = self.ontology.serialize(visible)
        self.queue.at[index, "resolved_uncertain_ingredients"] = self.ontology.serialize(uncertain)
        self.queue.at[index, "resolved_unreadable"] = self.unreadable.value
        self.queue.at[index, "resolved_non_food"] = self.non_food.value
        self.queue.at[index, "resolved_no_visible_ontology_label"] = self.no_visible_label.value
        self.queue.at[index, "resolution_notes"] = self.notes.value.strip()
        temporary = self.output_csv.with_suffix(".tmp.csv")
        self.queue.to_csv(temporary, index=False)
        temporary.replace(self.output_csv)
        pending_positions = [
            position
            for position, pending_index in enumerate(self.indices)
            if not str(self.queue.at[pending_index, "resolution_notes"]).strip()
        ]
        if pending_positions:
            later = [position for position in pending_positions if position > self.pointer]
            self.pointer = later[0] if later else pending_positions[0]
            self._load()
        else:
            self.message.value = "<b>All disagreements have a saved resolution. Run the finalization cell.</b>"

    def _previous(self, _button) -> None:
        self.pointer = max(0, self.pointer - 1)
        self._load()
