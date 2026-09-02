"""Notebook interface for semantic image-quality screening before sealing."""

from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from .quality import REJECTION_REASON_LABELS, quality_screen_progress


class QualityScreenApp:
    """Accept clean candidates and replace rejected ones within the same class."""

    def __init__(
        self,
        screen: pd.DataFrame,
        candidate_pool: pd.DataFrame,
        image_root: str | Path,
        output_csv: str | Path,
        samples_per_class: int,
    ):
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError("ipywidgets is required for the quality-screen interface") from exc

        self.widgets = widgets
        self.screen = screen.copy().reset_index(drop=True)
        self.candidate_pool = candidate_pool.copy().reset_index(drop=True)
        self.image_root = Path(image_root)
        self.output_csv = Path(output_csv)
        self.samples_per_class = int(samples_per_class)
        self.class_order = sorted(self.screen["food_class"].astype(str).unique())

        start = self._next_required_position()
        self.position = start if start is not None else 0
        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="720px", max_height="540px"))
        options = [("Accept: valid single image", "accept")]
        options.extend((label, reason) for reason, label in REJECTION_REASON_LABELS.items())
        self.decision = widgets.RadioButtons(
            options=options,
            value=None,
            description="Decision",
            layout=widgets.Layout(width="100%"),
            style={"description_width": "80px"},
        )
        self.notes = widgets.Textarea(
            description="Notes",
            placeholder="Optional, except required when choosing Other.",
            layout=widgets.Layout(width="100%", height="70px"),
            style={"description_width": "80px"},
        )
        self.status = widgets.HTML()
        self.progress = widgets.HTML()
        self.message = widgets.HTML()
        button_layout = widgets.Layout(width="160px", height="40px", margin="0 8px 0 0")
        self.previous_button = widgets.Button(description="Previous reviewed", layout=button_layout)
        self.save_button = widgets.Button(
            description="Save + Next",
            button_style="success",
            layout=button_layout,
        )
        self.previous_button.on_click(self._previous_reviewed)
        self.save_button.on_click(self._save_and_next)

        self.container = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Quality screen only.</b> Accept one clear food photograph. "
                    "Reject collages, heavy overlays, class mismatches, and scenes without one primary dish."
                ),
                self.status,
                widgets.HBox(
                    [self.image],
                    layout=widgets.Layout(width="100%", justify_content="center", padding="8px 0"),
                ),
                self.decision,
                self.notes,
                widgets.HBox([self.previous_button, self.save_button]),
                self.message,
                self.progress,
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
            image.thumbnail((960, 720))
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()

    def _accepted_count(self, food_class: str, exclude_position: int | None = None) -> int:
        accepted = self.screen["food_class"].eq(food_class) & self.screen["decision"].eq("accept")
        if exclude_position is not None:
            accepted.iloc[exclude_position] = False
        return int(accepted.sum())

    def _next_required_position(
        self,
        preferred_class: str | None = None,
        after_rank: int = 0,
    ) -> int | None:
        classes = []
        if preferred_class is not None:
            classes.append(preferred_class)
        classes.extend(name for name in self.class_order if name != preferred_class)
        for food_class in classes:
            if self._accepted_count(food_class) >= self.samples_per_class:
                continue
            candidates = self.screen[
                self.screen["food_class"].eq(food_class) & self.screen["decision"].fillna("").eq("")
            ]
            minimum_rank = after_rank if food_class == preferred_class else 0
            later = candidates[candidates["candidate_rank"].astype(int) > minimum_rank]
            if not later.empty:
                return int(later.sort_values("candidate_rank").index[0])
            if not candidates.empty:
                return int(candidates.sort_values("candidate_rank").index[0])
        return None

    def _progress_html(self) -> str:
        progress = quality_screen_progress(self.screen, self.candidate_pool, self.samples_per_class)
        rows = "".join(
            f"<tr><td>{escape(name)}</td><td>{counts['accepted']}</td>"
            f"<td>{counts['rejected']}</td><td>{counts['needed']}</td></tr>"
            for name, counts in progress["class_counts"].items()
        )
        return (
            f"<div style='padding-top:8px'><b>Overall:</b> reviewed {progress['reviewed']}, "
            f"accepted {progress['accepted']}, rejected {progress['rejected']}, "
            f"remaining acceptances {progress['remaining_acceptances']}</div>"
            "<table style='margin-top:6px'><tr><th>Class</th><th>Accepted</th>"
            f"<th>Rejected</th><th>Needed</th></tr>{rows}</table>"
        )

    def _load_position(self) -> None:
        row = self.screen.iloc[self.position]
        self.image.value = self._image_bytes(self.image_root / row["relative_path"])
        if row["decision"] == "accept":
            self.decision.value = "accept"
        elif row["decision"] == "reject":
            self.decision.value = str(row["rejection_reason"])
        else:
            self.decision.value = None
        self.notes.value = str(row["notes"] or "")
        accepted = self._accepted_count(str(row["food_class"]))
        self.status.value = (
            f"<div style='font-size:1.05em;padding:6px 0'>"
            f"<b>Expected class:</b> {escape(str(row['food_class']))}<br>"
            f"Candidate rank {int(row['candidate_rank'])}; accepted {accepted}/{self.samples_per_class} "
            f"for this class</div>"
        )
        self.progress.value = self._progress_html()
        self.message.value = ""

    def _save_current(self) -> bool:
        selected = self.decision.value
        if selected is None:
            self.message.value = "<span style='color:red'>Choose Accept or one rejection reason.</span>"
            return False
        if selected == "other" and not self.notes.value.strip():
            self.message.value = "<span style='color:red'>Add a note for the Other rejection reason.</span>"
            return False

        row = self.screen.iloc[self.position]
        food_class = str(row["food_class"])
        if (
            selected == "accept"
            and self._accepted_count(food_class, exclude_position=self.position) >= self.samples_per_class
        ):
            self.message.value = (
                "<span style='color:red'>This class already has enough accepted images. "
                "Change an earlier acceptance first.</span>"
            )
            return False
        self.screen.at[self.position, "decision"] = "accept" if selected == "accept" else "reject"
        self.screen.at[self.position, "rejection_reason"] = "" if selected == "accept" else selected
        self.screen.at[self.position, "notes"] = self.notes.value.strip()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_csv.with_suffix(".tmp.csv")
        self.screen.to_csv(temporary, index=False)
        temporary.replace(self.output_csv)
        return True

    def _save_and_next(self, _button) -> None:
        current_class = str(self.screen.at[self.position, "food_class"])
        current_rank = int(self.screen.at[self.position, "candidate_rank"])
        if not self._save_current():
            return
        next_position = self._next_required_position(current_class, current_rank)
        if next_position is None:
            self.progress.value = self._progress_html()
            progress = quality_screen_progress(self.screen, self.candidate_pool, self.samples_per_class)
            if progress["complete"]:
                self.message.value = (
                    "<b>Quality screening is complete. Run the sealing/export cell below the interface.</b>"
                )
            else:
                self.message.value = (
                    "<span style='color:red'>The reserve pool was exhausted before every class reached "
                    "the target. Increase quality_candidates_per_class and create a new packet.</span>"
                )
            return
        self.position = next_position
        self._load_position()

    def _previous_reviewed(self, _button) -> None:
        reviewed = self.screen.index[
            self.screen["decision"].fillna("").ne("") & (self.screen.index < self.position)
        ].tolist()
        if not reviewed:
            self.message.value = "No earlier reviewed candidate."
            return
        self.position = reviewed[-1]
        self._load_position()
