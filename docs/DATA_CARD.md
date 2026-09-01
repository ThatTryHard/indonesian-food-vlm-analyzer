# Data Card

## Sources and roles

| Source | Version identifier | Role |
|---|---|---|
| Food Ingredients and Recipe Dataset with Images | Kaggle slug in `configs/project.json` | Weak recipe-presence pretraining |
| Dataset Food Classification | Kaggle slug in `configs/project.json` | 260-image target benchmark |
| Food Recipes Dataset | Kaggle slug in `configs/project.json` | Optional development-only web/text evidence |

Dataset files are not redistributed. Users must obtain them from the original source and review the current source license/terms before use. Kaggle dataset bundle/version IDs and content hashes are stored in each run manifest where available.

See `docs/LICENSE_INVENTORY.md` for the recorded source licenses and required re-verification step.

## Sampling

- Corrupt images are excluded and logged.
- Exact SHA-256 duplicates and difference-hash near-duplicate groups are collapsed globally, including cross-class copies.
- Twenty unique images are sampled per class with seed 42.
- Split assignment is sealed before annotation.

## Annotation

See `docs/ANNOTATION_GUIDE.md`. Two independent passes and human adjudication are required. Uncertain components are stored but excluded from primary positive ground truth. The explicit `no_visible_ontology_label` state preserves legitimate all-negative images without forcing a positive label.

## Representativeness limits

Directory classes do not cover Indonesia's regional cuisine, demographic groups, preparation variation, restaurant conditions, or modern fusion dishes. The dataset includes global fast-food classes. Results describe this benchmark only.

## Privacy and ethics

The intended inputs are food images. The inventory should be reviewed for incidental faces, receipts, addresses, or other personal information before publishing examples. Do not publish raw images unless the source license permits redistribution.
