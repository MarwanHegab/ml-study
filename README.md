# Machine Learning Algorithms from Scratch — CS589 Final Project

A comparative study of supervised machine learning algorithms implemented **from scratch** in Python (NumPy/pandas) and evaluated across multiple real-world datasets. Built for COMPSCI 589 (Machine Learning) at UMass Amherst.

**Authors:** Marwan Hegab, Peter Ye, Max Goldman

## Overview

We implemented and rigorously benchmarked five classic algorithms without relying on prebuilt ML estimators, then compared their accuracy and weighted F1-score using **stratified 10-fold cross-validation** with hyperparameter tuning on each dataset.

### Algorithms implemented
- **k-Nearest Neighbors** (`kNN.py`)
- **Decision Tree** (`decisiontree.py`)
- **Random Forest** (`randomforest.py`)
- **Neural Network** (fully-connected, backprop from scratch) (`NN.py`)
- **Naive Bayes** (Gaussian + Multinomial) (`bayes.py`, `multinomialNB_marwan.py`)
- **Hybrid kNN + Decision Tree ensemble** (`dt-kNN_ensemble.py`)

### Datasets (`datasets/`)
- **Handwritten Digits** — 8×8 grayscale digit recognition (64 features)
- **Credit Approval** — binary classification
- **Parkinson's** — clinical voice-measure classification
- **Rice Grains** — morphological feature classification
- **GTZAN Music Genres** (extra) — 58 audio features across 10 genres

## Results

Each algorithm was evaluated with stratified 10-fold cross-validation, reporting accuracy and weighted F1-score. Full numerical results are in the `*_results.txt` files, plots are in `figures/`, and the complete write-up with analysis is in **[`report.pdf`](report.pdf)**.

Highlights:
- Built a hybrid ensemble of bootstrapped kNNs and Decision Trees, evaluated on the Rice Grains dataset across varying tree/kNN ratios.
- Extended Dataset 1 to a three-way comparison (kNN, Random Forest, Neural Network).

## Repository structure

```
.
├── kNN.py, decisiontree.py, randomforest.py, NN.py, bayes.py   # algorithm implementations
├── dt-kNN_ensemble.py                                          # hybrid ensemble
├── dataset1_analysis.py, dataset4_analysis.py, ...             # per-dataset experiments
├── evaluate*.py                                                # evaluation / CV harnesses
├── datasets/                                                   # CSV datasets
├── figures/                                                    # generated accuracy/cost plots
├── *_results.txt                                               # raw metric outputs
├── report.pdf / report.tex                                     # full project report
└── requirements.txt
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run a per-dataset experiment, e.g.:
python dataset1_analysis.py
```

## License

Released under the [MIT License](LICENSE).
