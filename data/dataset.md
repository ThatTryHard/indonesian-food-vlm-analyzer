# Dataset Documentation

This repository does not include the full datasets used in the notebook. The project relies on public Kaggle food image datasets and web/text-mined weak labels, which should be downloaded or attached separately when reproducing the experiment.

## Why the datasets are not included

The raw datasets are excluded from this repository for three reasons:

1. **File size** — food image datasets are usually too large for a clean GitHub repository.
2. **Licensing** — datasets from Kaggle should be accessed from their original source so users follow the dataset owner's terms.
3. **Reproducibility** — keeping dataset access separate makes the repository easier to clone, inspect, and run in different environments.

The repository should contain the notebook, documentation, and code only. Raw images, processed files, model checkpoints, and generated outputs should stay outside Git tracking.

## Recommended workflow: Kaggle Notebook

The easiest way to run this project is through Kaggle Notebook because the project uses image datasets, PyTorch, and Vision-Language Model inference.

Recommended Kaggle settings:

```text
Accelerator: GPU
Internet: On
```

In Kaggle, attach the required datasets through the **Add Data** panel. After the datasets are attached, the notebook should resolve the dataset paths automatically or through the dataset path configuration cell.

## Datasets used

The notebook uses three main data sources:

| Dataset | Purpose in this project |
|---|---|
| Food Ingredients and Recipes with Images | Main dataset for CNN-based multi-label ingredient classification |
| Food Classification dataset | Indonesian food image dataset for transfer/evaluation |
| Food Recipes dataset | Recipe/text source for ingredient vocabulary and weak-label construction |
| Web-mined text snippets | Additional weak supervision for Indonesian ingredient labels |

The exact Kaggle dataset names or slugs may differ depending on the source used. If you fork or reproduce this project, use the notebook's dataset path resolver section to confirm that the expected folders are detected correctly.

## Local folder structure

If running locally, place the downloaded datasets under `data/raw/`.

Expected structure:

```text
data/
└── raw/
    ├── food-ingredients-and-recipes/
    ├── food-classification/
    └── food-recipes/
```

If your downloaded dataset folders have different names, either rename the folders or update the dataset path variables in the notebook.

## Files that should not be committed

The following files and folders should stay out of GitHub:

```text
data/raw/
data/processed/
models/
checkpoints/
outputs/
*.pt
*.pth
*.ckpt
*.pkl
```

These files are intentionally ignored through `.gitignore`.

## Reproducibility notes

To reproduce the notebook:

1. Clone the repository.
2. Open the notebook in Kaggle or a local Jupyter environment.
3. Attach or download the required Kaggle datasets.
4. Confirm that the dataset paths are detected correctly.
5. Run the CNN and weak-label sections first.
6. Run the VLM section only after GPU availability is confirmed.

For local execution, the notebook can still run parts of the pipeline on CPU, but the Vision-Language Model section is strongly recommended to run on GPU.

## Data usage disclaimer

The datasets are used for educational and portfolio purposes. Nutrition estimates produced by the model are approximate machine-generated outputs and should not be treated as medical, dietary, or professional nutrition advice.