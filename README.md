---

## **DIPLOME UNIVERSITAIRE**
## **SORBONNE DATA ANALYTICS**
---
## **Projet Deep Learning**
## **Classification de tissus cancéreux colorectaux**

---
*Professeur* :

*Etudiants* :

Promotion 007

Mars 2026

---
**Jeu de données MedMNIST : PathMNIST — Colorectal Cancer Histology**

https://medmnist.com/

---

## Notebooks

| Notebook | Contenu |
|----------|---------|
| NB1 — EDA | Exploration du dataset, statistiques, visualisations, Q1.1 & Q1.2 |
| NB2 — MLP | Réseau dense baseline (63% test accuracy) |
| NB3 — CNN | CNN from scratch + comparaison augmentations (89% test accuracy) |
| NB4 — ResNet-18 | Transfer learning frozen vs fine-tuning (91% test accuracy) |
| NB5 — ViT | Vision Transformer from scratch (79% test accuracy) |
| NB6 — Grad-CAM | Visualisation des zones d'attention du modèle |

## Prérequis
- Python 3.12
- PyTorch 2.10
- MedMNIST 3.0
- GPU recommandé (CUDA)

## Installation
```bash
pip install torch torchvision medmnist scikit-learn seaborn scipy
