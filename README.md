# Deep Learning Project — Automated Classification of Colorectal Cancer Tissues

**Diplome Universitaire Sorbonne Data Analytics — Promotion 007 — Mars 2026**

## Context

Colorectal cancer is the third most common cancer worldwide. Histopathological analysis of tissue biopsies remains the gold standard for diagnosis, but it is time-consuming and subject to inter-observer variability. Automating the classification of tissue types from H&E-stained histology images could assist pathologists in screening large volumes of slides, prioritizing suspicious cases, and reducing diagnostic delays.

This project explores whether deep learning models can reliably distinguish 9 tissue types from low-resolution (28x28) colorectal histology patches, and evaluates the trade-offs between model complexity, diagnostic performance, and clinical deployability.

## Dataset

**PathMNIST** (MedMNIST benchmark) — 107,180 colorectal cancer histopathology patches (28x28, RGB), 9 tissue types (adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium).

- Train: 89,996 images (Hospital A — NCT-CRC-HE-100K)
- Val: 10,004 images (Hospital A)
- Test: 7,180 images (Hospital B — CRC-VAL-HE-7K)

The train/test split comes from two different hospitals, introducing a **domain shift** that is the central challenge of this project.

The dataset is automatically downloaded when running the notebooks.

## Notebooks

| Notebook | Content | Test accuracy | Recall cancer |
|----------|---------|--------------|---------------|
| NB1 — EDA | Data exploration, domain shift analysis, Q1.1 & Q1.2 | — | — |
| NB2 — MLP | Dense network baseline (2352-512-256-128-9) | 68.02% | 0.7745 |
| NB3 — CNN | CNN from scratch + augmentation iterations | 91.78% | 0.9570 |
| NB4 — ResNet-18 | Transfer learning: frozen (87.14%) vs fine-tuning | 91.77% | 0.9659 |
| NB5 — ViT | Vision Transformer from scratch, patch 7x7 | 81.98% | 0.8873 |
| NB6 — Grad-CAM | Interpretability: what the models look at | — | — |
| NB7 — Comparison | Final comparison, metrics analysis, clinical recommendations | — | — |

## Key findings

1. **CNN from scratch (91.78%) matches ResNet fine-tuning (91.77%) with 25x fewer parameters.** The key is ColorJitter augmentation, which simulates the H&E staining variations between hospitals and directly addresses the domain shift.

2. **Accuracy alone is misleading for clinical deployment.** The ResNet FT dominates on cancer detection (F1 = 0.9632 vs 0.9414 for CNN, precision 0.9605 vs 0.9262). In oncology, missing a cancer (false negative) is far more dangerous than a false alarm.

3. **Stroma is universally difficult** (recall 0.39-0.53 across all models). This is a physical limitation of the 28x28 resolution — connective tissue textures are indistinguishable at this scale.

4. **ViT without positional embeddings outperforms ViT with them** (83.40% vs 81.98%). In histology, tissue identity is in the texture, not the location — a finding that contrasts sharply with NLP where position is essential.

5. **CNN and ResNet use complementary visual strategies** (Grad-CAM analysis). An ensemble could potentially surpass both.

## Team

- **Xia Bizot** — NB1 (EDA), NB2 (MLP), NB3 (CNN), NB5 (ViT), NB7 (Comparison)
- **Camille** — NB4 (ResNet-18), NB6 (Grad-CAM)

## Prerequisites

- Python 3.12
- PyTorch >= 2.0 (CUDA recommended)
- MedMNIST >= 3.0

## Installation

```bash
pip install -r requirements.txt
```

## Run order

Notebooks must be run in order (NB1 to NB7). Each notebook saves its models and predictions for downstream notebooks.

```
NB1 (EDA) -> NB2 (MLP) -> NB3 (CNN) -> NB4 (ResNet) -> NB5 (ViT) -> NB6 (Grad-CAM) -> NB7 (Comparison)
```

## Reproducibility

All notebooks use SEED=42, torch.backends.cudnn.deterministic=True, and save their best checkpoints and predictions as pickle files. Normalization constants are computed in NB1 and reused across all subsequent notebooks.
