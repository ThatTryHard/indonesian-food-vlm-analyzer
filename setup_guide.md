# Setup Guide

This guide explains how to run the **Indonesian Food VLM Analyzer** project and prepare the repository for GitHub.

This project is designed to be run primarily on **Kaggle Notebook with GPU acceleration**. The notebook includes CNN training, web/text mining, weak-label construction, and Vision-Language Model inference. Because the VLM section is resource-heavy, running the full notebook on a local CPU-only machine is not recommended.

---

## 1. Recommended Environment

### Main workflow: Kaggle Notebook

Use Kaggle when running the full notebook, especially the Qwen2-VL / Vision-Language Model section.

Recommended Kaggle settings:

```text
Accelerator: GPU T4 x2
Internet: On
Persistence: Files only, if needed
```

Why Kaggle is recommended:

* No manual CUDA installation is needed.
* GPU acceleration is available directly from the notebook settings.
* Kaggle datasets can be attached through the Kaggle interface.
* Large model inference is much more practical than running on a CPU-only laptop.
* The project can be reproduced without storing large datasets inside the GitHub repository.

### Optional: local machine

Local setup is only useful for editing the notebook, checking documentation, or running lighter preprocessing sections. The full VLM pipeline is not intended to be run locally unless the machine has a compatible NVIDIA GPU and CUDA-enabled PyTorch.

Local folders such as `.venv/`, `data/raw/`, `data/processed/`, and `models/` should remain outside Git tracking.

---

## 2. Repository Structure

The GitHub repository should contain only the notebook, documentation, dependency list, and lightweight project files.

Recommended GitHub structure:

```text
indonesian-food-vlm-analyzer/
├── README.md
├── indonesian-food-vlm-data-analyzer.ipynb
├── requirements.txt
├── .gitignore
├── setup_guide.md
└── data/
    └── dataset.md
```

Optional:

```text
assets/
└── sample screenshots, result previews, or diagrams
```

Do not commit raw datasets, processed datasets, model checkpoints, virtual environments, or generated experiment outputs.

These should stay local only:

```text
.venv/
data/raw/
data/processed/
models/
outputs/
checkpoints/
```

---

## 3. Running on Kaggle

### Step 1: Create or upload the notebook

Create a new Kaggle Notebook, then upload:

```text
indonesian-food-vlm-data-analyzer.ipynb
```

Alternatively, upload the notebook to GitHub first and import it from the GitHub repository into Kaggle.

### Step 2: Enable GPU and internet

Open the notebook settings and set:

```text
Accelerator → GPU T4 x2
Internet → On
```

The GPU is important for the VLM section. Without GPU, the CNN and data-processing sections may still run, but the Qwen2-VL section will be slow or impractical.

### Step 3: Attach the datasets

Attach the required Kaggle datasets from the **Add Data** panel.

The notebook expects datasets related to:

```text
food-classification
food-ingredients-and-recipes
food-recipes
```

The notebook contains path-resolving cells that check whether the dataset folders are available. If a path is not detected automatically, check the printed dataset paths in the early setup cells and adjust the path variables if needed.

### Step 4: Install missing packages if needed

Kaggle already includes many common machine learning packages. If a required package is missing, run the package installation cell near the top of the notebook.

Manual install command:

```python
!pip install -q transformers accelerate qwen-vl-utils peft torchmetrics ddgs kagglehub
```

Restart the session only if Kaggle asks you to do so after installation.

### Step 5: Confirm GPU availability

Before running the VLM section, check that PyTorch can detect the GPU:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU detected")
```

Expected result:

```text
True
Tesla T4
```

The exact GPU name may differ depending on Kaggle availability.

### Step 6: Run the notebook

Recommended run order:

1. Run environment setup and package installation cells.
2. Run dataset path detection cells.
3. Run CNN preprocessing, training, and evaluation cells.
4. Run web/text mining and weak-label construction cells.
5. Run vocabulary expansion cells.
6. Run VLM structured prompting cells after confirming GPU availability.
7. Run final model comparison, hallucination analysis, and summary sections.

---

## 4. Dataset Handling

Do not upload the full datasets to GitHub.

This project uses public Kaggle food image datasets and web/text-mined weak labels. The full datasets are excluded because they are large and may have their own licensing restrictions.

Your local project folder may contain:

```text
data/
├── dataset.md
├── raw/
│   ├── food-classification/
│   ├── food-ingredients-and-recipes/
│   └── food-recipes/
└── processed/
```

However, GitHub should only contain:

```text
data/
└── dataset.md
```

The following folders must be ignored by Git:

```text
data/raw/
data/processed/
```

When running on Kaggle, attach datasets through the Kaggle interface instead of uploading the dataset folders to GitHub.

---

## 5. Model and Output Handling

Do not upload trained model weights or generated outputs to GitHub.

The `models/` folder can exist locally, but it should not be committed. The same applies to generated checkpoints, experiment outputs, and saved model files.

These should be ignored:

```text
models/
checkpoints/
outputs/
*.pt
*.pth
*.ckpt
*.safetensors
*.pkl
*.joblib
```

If the `models/` folder is empty, it does not need to be included in the repository.

---

## 6. GitHub Upload

Before pushing, check the repository status:

```bash
git status
```

Only stage the files that should appear in GitHub:

```bash
git add README.md
git add indonesian-food-vlm-data-analyzer.ipynb
git add requirements.txt
git add .gitignore
git add setup_guide.md
git add data/dataset.md
```

Avoid using this unless you are completely sure `.gitignore` is correct:

```bash
git add .
```

Before committing, run:

```bash
git status
```

Make sure these folders are not staged:

```text
.venv/
data/raw/
data/processed/
models/
outputs/
checkpoints/
```

Then commit and push:

```bash
git commit -m "Prepare Kaggle-ready Indonesian food VLM portfolio project"
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main
```

If you add lightweight screenshots or diagrams later:

```bash
git add assets/
git commit -m "Add project visuals"
git push
```

---

## 7. Suggested Repository Description

```text
Multimodal food image analysis pipeline for Indonesian dishes using CNN, web-mined weak labels, and Vision-Language Models for structured ingredient and nutrition estimation.
```

---

## 8. Suggested GitHub Topics

```text
computer-vision
vision-language-models
food-analysis
multilabel-classification
web-mining
weak-supervision
nutrition-estimation
pytorch
kaggle
qwen-vl
```

---

## 9. Troubleshooting

### Kaggle does not detect GPU

Run:

```python
import torch
print(torch.cuda.is_available())
```

If the result is `False`, check the Kaggle notebook settings and make sure GPU acceleration is enabled.

### VLM section is slow or crashes

The VLM section is the heaviest part of the notebook. Use GPU acceleration and avoid running multiple large models at the same time. If memory issues occur, restart the session and run only the required VLM cells.

### Dataset path is not found

If a dataset path is not detected:

1. Check that the dataset is attached in the Kaggle sidebar.
2. Print the available input folders:

```python
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    print(dirname)
```

3. Update the dataset path variables in the notebook if the folder names are different.

### Missing package error

If Kaggle raises an import error, install the missing package in a notebook cell:

```python
!pip install -q package-name
```

For this project, the most likely missing packages are:

```python
!pip install -q transformers accelerate qwen-vl-utils peft torchmetrics ddgs kagglehub
```

### Accidentally staged dataset files

If `git status` shows files under `data/raw/` or `data/processed/`, unstage them:

```bash
git rm -r --cached data/raw
git rm -r --cached data/processed
```

Then check again:

```bash
git status
```

The dataset folders should no longer be staged.