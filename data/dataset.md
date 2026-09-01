# Dataset Documentation

## Data roles

| Dataset | Frozen Kaggle slug | Role |
|---|---|---|
| Food Ingredients and Recipe Dataset with Images | `pes12017000148/food-ingredients-and-recipe-dataset-with-images/versions/1` | Optional weak recipe-presence pretraining |
| Dataset Food Classification | `rizkyyk/dataset-food-classification/versions/5` | 260-image visible-component benchmark |
| Food Recipes Dataset | `albertnathaniel12/food-recipes-dataset/versions/2` | Optional development-only text evidence |

Only the target benchmark is final ground truth. The recipe dataset and web snippets are development inputs.

## Benchmark construction

- Verify every image with PIL.
- Record dimensions, bytes, SHA-256, and a 64-bit difference hash.
- Collapse exact and near-duplicate groups before sampling.
- Quarantine any exact/near copies found in the optional recipe-pretraining source before model fitting.
- Sample 20 unique images per class with seed 42.
- Seal 12 train, 4 validation, and 4 test images per class before annotation.
- Hash the manifest in `manifest_lock.json`.
- Obtain two independent visible-component annotations and adjudicate disagreement.

## Not included in Git

- raw images;
- annotation packets containing source paths;
- trained checkpoints;
- raw web evidence;
- raw VLM outputs;
- generated prediction tables.

Obtain datasets from original sources and inspect their current terms. A completed run manifest records versions and digests without redistributing data.

## Representativeness

The source contains 13 classes, including Burger, Pizza, and French Fries. It is not a comprehensive sample of Indonesian cuisine, regions, preparation methods, or serving contexts. Conclusions must remain benchmark-specific.

See `docs/DATA_CARD.md` and `docs/ANNOTATION_GUIDE.md`.
