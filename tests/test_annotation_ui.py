from pathlib import Path

import pandas as pd
from PIL import Image

from src.annotation_ui import AnnotationApp
from src.annotations import create_annotation_sheet
from src.ontology import IngredientOntology

ONTOLOGY = IngredientOntology.from_json(Path(__file__).parents[1] / "data" / "ontology" / "visible_ingredients.json")


def test_annotation_ui_uses_exclusive_states_and_saves(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (64, 48), color=(180, 120, 60)).save(image_path)
    manifest = pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "relative_path": [image_path.name],
            "food_class": ["hidden-from-interface"],
            "split": ["train"],
        }
    )
    sheet = create_annotation_sheet(manifest, "annotator_a")
    output_path = tmp_path / "annotations_annotator_a.csv"

    app = AnnotationApp(sheet, ONTOLOGY, tmp_path, output_path)

    assert len(app.label_panel.controls) == len(ONTOLOGY.ids) == 43
    assert len(app.label_panel.accordion.children) == 9

    app.label_panel.controls["rice"].value = "visible"
    app.label_panel.controls["chicken"].value = "uncertain"
    assert app._save_current()

    saved = pd.read_csv(output_path, keep_default_na=False)
    assert saved.loc[0, "visible_ingredients"] == "rice"
    assert saved.loc[0, "uncertain_ingredients"] == "chicken"

    app.special_status.value = "unreadable"
    assert app.label_panel.values() == (set(), set())
    assert all(control.disabled for control in app.label_panel.controls.values())
