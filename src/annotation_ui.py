"""Notebook interfaces for independent annotation and adjudication."""

from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from .annotations import annotation_progress, parse_boolean


class _IngredientStatePanel:
    """Responsive, single-state ingredient controls grouped by ontology category."""

    def __init__(self, widgets, ontology):
        self.widgets = widgets
        self.ontology = ontology
        self.controls = {}
        self._suspend_summary = False
        self.summary = widgets.HTML()

        categories: dict[str, list] = {}
        for label in ontology.labels:
            categories.setdefault(label.category, []).append(label)

        category_boxes = []
        for _category, labels in categories.items():
            rows = []
            for label in labels:
                selector = widgets.ToggleButtons(
                    options=[("None", ""), ("Visible", "visible"), ("Uncertain", "uncertain")],
                    value="",
                    tooltips=[
                        "This component is not supported by the image.",
                        "This component is visually identifiable.",
                        "A component is present, but its identity is visually ambiguous.",
                    ],
                    layout=widgets.Layout(width="340px"),
                    style={"button_width": "105px"},
                )
                selector.observe(self._update_summary, names="value")
                self.controls[label.id] = selector
                label_text = widgets.HTML(
                    value=(
                        f"<div style='line-height:1.25'>"
                        f"<b>{escape(label.id.replace('_', ' '))}</b><br>"
                        f"<span style='opacity:0.82'>{escape(label.hint)}</span>"
                        f"</div>"
                    ),
                    layout=widgets.Layout(min_width="330px", width="58%"),
                )
                rows.append(
                    widgets.HBox(
                        [label_text, selector],
                        layout=widgets.Layout(
                            width="100%",
                            display="flex",
                            flex_flow="row wrap",
                            align_items="center",
                            justify_content="space-between",
                            padding="5px 2px",
                        ),
                    )
                )
            category_boxes.append(widgets.VBox(rows, layout=widgets.Layout(width="100%")))

        self.accordion = widgets.Accordion(children=category_boxes, selected_index=0)
        for index, (category, labels) in enumerate(categories.items()):
            title = category.replace("_", " ").title()
            self.accordion.set_title(index, f"{title} ({len(labels)})")
        self.accordion.layout = widgets.Layout(width="100%")

        legend = widgets.HTML(
            "<b>Ingredient evidence</b> &nbsp; "
            "Choose <b>Visible</b> only when identifiable from the pixels. "
            "Choose <b>Uncertain</b> when a component is present but its identity is ambiguous. "
            "Leave unrelated labels at <b>None</b>."
        )
        self.widget = widgets.VBox(
            [legend, self.summary, self.accordion],
            layout=widgets.Layout(width="100%"),
        )
        self._update_summary()

    def _update_summary(self, _change=None) -> None:
        if self._suspend_summary:
            return
        visible, uncertain = self.values()
        visible_text = ", ".join(sorted(visible)) or "none"
        uncertain_text = ", ".join(sorted(uncertain)) or "none"
        self.summary.value = (
            f"<div style='padding:6px 0'>"
            f"<b>Visible:</b> {escape(visible_text)}<br>"
            f"<b>Uncertain:</b> {escape(uncertain_text)}"
            f"</div>"
        )

    def values(self) -> tuple[set[str], set[str]]:
        visible = {label_id for label_id, control in self.controls.items() if control.value == "visible"}
        uncertain = {label_id for label_id, control in self.controls.items() if control.value == "uncertain"}
        return visible, uncertain

    def set_values(self, visible: set[str], uncertain: set[str]) -> None:
        self._suspend_summary = True
        try:
            for label_id, control in self.controls.items():
                if label_id in visible:
                    control.value = "visible"
                elif label_id in uncertain:
                    control.value = "uncertain"
                else:
                    control.value = ""
        finally:
            self._suspend_summary = False
        self._update_summary()

    def clear(self) -> None:
        self.set_values(set(), set())

    def set_disabled(self, disabled: bool) -> None:
        for control in self.controls.values():
            control.disabled = disabled


def _special_status_value(unreadable: object, non_food: object, no_visible_label: object) -> str:
    if parse_boolean(unreadable):
        return "unreadable"
    if parse_boolean(non_food):
        return "non_food"
    if parse_boolean(no_visible_label):
        return "no_visible_label"
    return ""


def _special_status_control(widgets, description: str = "Image status"):
    return widgets.ToggleButtons(
        options=[
            ("Normal", ""),
            ("Unreadable", "unreadable"),
            ("Not food", "non_food"),
            ("No supported label", "no_visible_label"),
        ],
        value="",
        description=description,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "100px", "button_width": "145px"},
    )


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

        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="680px", max_height="500px"))
        self.label_panel = _IngredientStatePanel(widgets, ontology)
        self.special_status = _special_status_control(widgets)
        self.special_status.observe(self._special_status_changed, names="value")
        self.notes = widgets.Textarea(
            description="Notes",
            placeholder="Optional: record a genuine ambiguity or image-quality issue.",
            layout=widgets.Layout(width="100%", height="75px"),
            style={"description_width": "70px"},
        )
        self.status = widgets.HTML()
        self.message = widgets.HTML()
        button_layout = widgets.Layout(width="150px", height="40px", margin="0 8px 0 0")
        self.previous_button = widgets.Button(description="Previous", layout=button_layout)
        self.save_button = widgets.Button(description="Save + Next", button_style="success", layout=button_layout)
        self.next_unfinished_button = widgets.Button(
            description="Next unfinished", icon="forward", layout=button_layout
        )
        self.previous_button.on_click(self._previous)
        self.save_button.on_click(self._save_and_next)
        self.next_unfinished_button.on_click(self._next_unfinished)

        image_box = widgets.HBox(
            [self.image],
            layout=widgets.Layout(width="100%", justify_content="center", padding="8px 0"),
        )
        self.container = widgets.VBox(
            [
                self.status,
                image_box,
                self.label_panel.widget,
                widgets.HTML(
                    "<b>Use a special status only when ingredient labeling does not apply.</b> "
                    "Selecting one clears and disables ingredient choices."
                ),
                self.special_status,
                self.notes,
                widgets.HBox(
                    [self.previous_button, self.save_button, self.next_unfinished_button],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                self.message,
            ],
            layout=widgets.Layout(width="100%"),
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

    def _special_status_changed(self, change) -> None:
        special = bool(change["new"])
        if special:
            self.label_panel.clear()
        self.label_panel.set_disabled(special)
        if special:
            self.message.value = "Ingredient choices cleared because a special image status was selected."

    def _load_position(self) -> None:
        row = self.sheet.iloc[self.position]
        path = self.image_root / row["relative_path"]
        self.image.value = self._image_bytes(path)
        visible = set(self.ontology.parse_annotation_cell(row["visible_ingredients"]))
        uncertain = set(self.ontology.parse_annotation_cell(row["uncertain_ingredients"]))
        special = _special_status_value(row["unreadable"], row["non_food"], row["no_visible_ontology_label"])
        self.special_status.value = special
        self.label_panel.set_disabled(bool(special))
        self.label_panel.set_values(visible, uncertain)
        self.notes.value = str(row["notes"] or "")
        progress = annotation_progress(self.sheet)
        self.status.value = (
            f"<div style='font-size:1.05em;padding:6px 0'>"
            f"<b>{self.position + 1}/{len(self.sheet)}</b> &nbsp; Sample {escape(str(row['sample_id']))}<br>"
            f"Completed {progress['completed']} &nbsp; Remaining {progress['remaining']}"
            f"</div>"
        )
        self.message.value = ""

    def _save_current(self) -> bool:
        visible, uncertain = self.label_panel.values()
        special = self.special_status.value
        if special and (visible or uncertain):
            self.message.value = "<span style='color:red'>A special-status image cannot carry ingredient labels.</span>"
            return False
        if not visible and not uncertain and not special:
            self.message.value = "<span style='color:red'>Add a label or choose a special image status.</span>"
            return False
        index = self.position
        self.sheet.at[index, "visible_ingredients"] = self.ontology.serialize(visible)
        self.sheet.at[index, "uncertain_ingredients"] = self.ontology.serialize(uncertain)
        self.sheet.at[index, "unreadable"] = special == "unreadable"
        self.sheet.at[index, "non_food"] = special == "non_food"
        self.sheet.at[index, "no_visible_ontology_label"] = special == "no_visible_label"
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
            self.message.value = "<b>All rows have an annotation. Export and preserve the benchmark packet.</b>"


class AdjudicationApp:
    """Resolve evaluation rows on which the two independent annotations disagree."""

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

        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="680px", max_height="500px"))
        self.label_panel = _IngredientStatePanel(widgets, ontology)
        self.special_status = _special_status_control(widgets, description="Resolution")
        self.special_status.observe(self._special_status_changed, names="value")
        self.notes = widgets.Textarea(
            description="Rationale",
            placeholder="Required: briefly justify the final resolution.",
            layout=widgets.Layout(width="100%", height="80px"),
            style={"description_width": "80px"},
        )
        self.status = widgets.HTML()
        self.comparison = widgets.HTML()
        self.message = widgets.HTML()
        button_layout = widgets.Layout(width="160px", height="40px", margin="0 8px 0 0")
        self.previous_button = widgets.Button(description="Previous", layout=button_layout)
        self.save_button = widgets.Button(description="Resolve + Next", button_style="warning", layout=button_layout)
        self.previous_button.on_click(self._previous)
        self.save_button.on_click(self._save_and_next)
        self.container = widgets.VBox(
            [
                self.status,
                widgets.HBox([self.image], layout=widgets.Layout(width="100%", justify_content="center")),
                self.comparison,
                self.label_panel.widget,
                self.special_status,
                self.notes,
                widgets.HBox([self.previous_button, self.save_button]),
                self.message,
            ],
            layout=widgets.Layout(width="100%"),
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

    def _special_status_changed(self, change) -> None:
        special = bool(change["new"])
        if special:
            self.label_panel.clear()
        self.label_panel.set_disabled(special)

    def _load(self) -> None:
        index = self.indices[self.pointer]
        row = self.queue.loc[index]
        self.image.value = self._image_bytes(self.image_root / row["relative_path"])
        has_resolution = bool(str(row["resolution_notes"]).strip())
        visible_cell = row["resolved_visible_ingredients"] if has_resolution else row["visible_ingredients_a"]
        uncertain_cell = row["resolved_uncertain_ingredients"] if has_resolution else row["uncertain_ingredients_a"]
        special = _special_status_value(
            row["resolved_unreadable"] if has_resolution else row["unreadable_a"],
            row["resolved_non_food"] if has_resolution else row["non_food_a"],
            row["resolved_no_visible_ontology_label"] if has_resolution else row["no_visible_ontology_label_a"],
        )
        self.special_status.value = special
        self.label_panel.set_disabled(bool(special))
        self.label_panel.set_values(
            set(self.ontology.parse_annotation_cell(visible_cell)),
            set(self.ontology.parse_annotation_cell(uncertain_cell)),
        )
        self.notes.value = str(row["resolution_notes"] or "")
        self.status.value = (
            f"<b>Disagreement {self.pointer + 1}/{len(self.indices)}</b> &nbsp; Sample {escape(str(row['sample_id']))}"
        )
        self.comparison.value = (
            f"<div style='padding:8px 0'>"
            f"<b>Annotator A</b> visible: {escape(str(row['visible_ingredients_a'] or '(none)'))}; "
            f"uncertain: {escape(str(row['uncertain_ingredients_a'] or '(none)'))}<br>"
            f"<b>Annotator B</b> visible: {escape(str(row['visible_ingredients_b'] or '(none)'))}; "
            f"uncertain: {escape(str(row['uncertain_ingredients_b'] or '(none)'))}"
            f"</div>"
        )
        self.message.value = ""

    def _save_and_next(self, _button) -> None:
        visible, uncertain = self.label_panel.values()
        special = self.special_status.value
        if special and (visible or uncertain):
            self.message.value = "<span style='color:red'>A special-status image cannot carry labels.</span>"
            return
        if not visible and not uncertain and not special:
            self.message.value = "<span style='color:red'>Add a label or choose a special status.</span>"
            return
        if not self.notes.value.strip():
            self.message.value = "<span style='color:red'>Add a brief adjudication rationale.</span>"
            return
        index = self.indices[self.pointer]
        self.queue.at[index, "resolved_visible_ingredients"] = self.ontology.serialize(visible)
        self.queue.at[index, "resolved_uncertain_ingredients"] = self.ontology.serialize(uncertain)
        self.queue.at[index, "resolved_unreadable"] = special == "unreadable"
        self.queue.at[index, "resolved_non_food"] = special == "non_food"
        self.queue.at[index, "resolved_no_visible_ontology_label"] = special == "no_visible_label"
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
