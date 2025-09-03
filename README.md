# Deepfake-Detection-And-Prevention-Using-AI
AI-powered Deepfake Detection and Prevention system using Python, TensorFlow, and OpenCV.
# Deepfake Detection & Prevention

> A step-by-step guide to download, run, and get final results from the Deepfake Detection and Prevention repository.

---

## Table of Contents

* [About](#about)
* [Features](#features)
* [Project structure](#project-structure)
* [Prerequisites](#prerequisites)
* [Download the repository](#download-the-repository)
* [Install dependencies](#install-dependencies)
* [Prepare dataset and models](#prepare-dataset-and-models)
* [Run training (optional)](#run-training-optional)
* [Run inference (get final output)](#run-inference-get-final-output)
* [Run the Streamlit demo UI](#run-the-streamlit-demo-ui)
* [Expected outputs](#expected-outputs)
* [Troubleshooting & tips](#troubleshooting--tips)
* [Contributing & License](#contributing--license)
* [Ethics & Usage](#ethics--usage)

---

## About

This repository provides an end-to-end pipeline to **detect** and **help prevent** deepfake images and videos. It includes training code, inference scripts, visualization (Grad-CAM heatmaps), and a Streamlit demo app for quick testing.

---

## Features

* Frame-level and temporal (video) detection.
* Lightweight backbones (e.g., MobileNetV2) for fast inference.
* Explainability using Grad-CAM heatmaps.
* Simple Streamlit UI for uploading media and viewing results.
* Exportable detection reports.

---

## Project structure (example)

```
README.md
requirements.txt
app.py                     # Streamlit demo
src/
  ├─ train.py
  ├─ infer.py
  ├─ utils.py
  ├─ data_loader.py
  └─ models/
models/                    # saved weights (gitignored)
notebooks/                 # experiments and EDA
data/                      # (gitignored) local dataset folder
inputs/                    # sample inputs for quick test
outputs/                   # inference outputs (predictions, heatmaps, reports)
assets/                    # screenshots and GIFs
```

---

## Prerequisites

* Python 3.8 or newer
* At least 8 GB RAM (GPU recommended for training)
* Git
* (Optional) CUDA-compatible GPU + drivers for faster training/inference

---

## Download the repository

1. Open a terminal (Linux/macOS) or PowerShell (Windows).
2. Clone your repo:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

3. If you prefer ZIP: on GitHub click **Code ▸ Download ZIP**, then extract.

---

## Install dependencies

It is recommended to use a virtual environment.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` should include: `tensorflow` or `torch` (depending on your implementation), `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `streamlit`, and any other utilities.

---

## Prepare dataset and models

1. **Obtain dataset**: Download a public dataset such as FaceForensics++ or a DFDC subset (or use your own). Place dataset under `data/` with subfolders like `real/` and `fake/` or follow the loader's required format.

2. **Prepare dataset** (if required): run any preprocessing scripts to extract frames, align faces, and create train/val/test splits.

```bash
python src/data_loader.py --input data/raw --output data/processed --mode extract_frames
```

3. **(Optional) Download pre-trained weights**: If you provide example weights in `models/`, copy them there. If hosted externally, download and place into `models/`.

---

## Run training (optional)

If you want to train your own model:

```bash
python src/train.py \
  --dataset data/processed \
  --epochs 20 \
  --batch-size 32 \
  --backbone mobilenet_v2 \
  --save-dir models/run1
```

Notes:

* Use `--gpu` or environment variables to enable GPU if available.
* Check the `train.py` help (`-h`) for more flags (learning rate, augmentation, resume checkpoint, etc.).

---

## Run inference (get final output)

Use `infer.py` to produce the final prediction, heatmaps, and an optional report.

Example (image):

```bash
python src/infer.py \
  --input inputs/sample_image.jpg \
  --model models/run1/best_model.h5 \
  --output outputs/sample_image_results.json \
  --visualize outputs/heatmaps/
```
What these commands produce:

* `outputs/*.json` or `outputs/*.csv` — per-file/frame predictions and confidence scores.
* `outputs/heatmaps/` — saved Grad-CAM images for frames flagged as suspicious.
* `outputs/report_<input>.pdf` (optional) — a human-readable report combining visuals and summary stats.

---

## Run the Streamlit demo UI

Start the interactive demo to upload media and see results quickly:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The UI allows webcam or file upload, shows prediction probability, and displays Grad-CAM heatmaps.

---

## Expected final output

After running inference or the Streamlit UI you should have:

* A prediction label: `real` or `fake` with a probability score.
* Visual explanation images (heatmaps) showing suspicious regions.
* A report (JSON/CSV/PDF) summarizing the results and flagged timestamps/frames.

---

## Troubleshooting & tips

* **Model not loading**: confirm `--model` path and model format (.h5, .pt). Check package versions.
* **CUDA errors**: ensure GPU drivers, CUDA, and cuDNN match your TensorFlow/PyTorch version.
* **Missing dependencies**: reinstall with `pip install -r requirements.txt` and check for version conflicts.
* **Slow inference**: switch to a lightweight backbone (MobileNet) or run on GPU.

---

## Contributing & License

Contributions are welcome. Please open issues or PRs. Add `CONTRIBUTING.md` for contribution rules.

This project is released under the **MIT License** — see `LICENSE`.

---

## Ethics & Usage

This repository is intended for **research, education, and defensive** use only. Do not use it to create or distribute deepfakes, violate privacy, or make legal judgments. Include a dataset citation and follow licenses for any public datasets used.

---

If you want, I can also generate `CONTRIBUTING.md`, `requirements.txt`, example `train.py`/`infer.py` stubs, or a polished `app.py` Streamlit demo. Just tell me which file you want next.




https://github.com/user-attachments/assets/cc694114-e806-4585-a944-dd4aa8a297b1




