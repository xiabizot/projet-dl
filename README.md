# Cell.IA — Classification de tissus cancereux colorectaux

**Projet Deep Learning — DU Sorbonne Data Analytics — Promotion 007 — Mars 2026**

## Dataset
PathMNIST (benchmark MedMNIST) — 107 180 images d'histopathologie colorectale (28x28, RGB), 9 types de tissus.
Le dataset est telecharge automatiquement a l'execution des notebooks.

## Notebooks

| Notebook | Contenu | Test accuracy |
|----------|---------|--------------|
| NB1 — EDA | Exploration, statistiques, visualisations, Q1.1 & Q1.2 | — |
| NB2 — MLP | Reseau dense baseline | 68.02% |
| NB3 — CNN | CNN from scratch + iterations d'augmentation (v1 meilleur) | 91.78% |
| NB4 — ResNet-18 | Transfer learning : frozen vs fine-tuning | 91.77% |
| NB5 — ViT | Vision Transformer from scratch | 81.98% |
| NB6 — Grad-CAM | Interpretabilite : ou regardent les modeles | — |
| NB7 — Comparaison | Comparaison finale + recommandations cliniques | — |
| NB8 — Agent | Agent IA Cell.IA + export artefacts | — |

## Resultats cles
- Le CNN v1 from scratch (436K params) egale le ResNet fine-tuning (11.1M params) — le ColorJitter simule le domain shift
- Le ViT sans positional embeddings surpasse le ViT avec — l'histologie est translationnellement invariante
- Le domain shift entre hopitaux explique l'ecart val/test sur tous les modeles
- L'accuracy globale (91.78%) masque un F1 stroma de 0.66 — le recall par classe est la metrique clinique

## Application Cell.IA

**Classification Explicable de Lames par Learning — Intelligence Artificielle**

Application Streamlit deployee en ligne avec :
- Classification par CNN v1 et ensemble CNN + ResNet
- Monte Carlo Dropout pour l'estimation d'incertitude
- Grad-CAM comparatif CNN vs ResNet (glass-box)
- Recommandations par similarite cosine sur embeddings
- Visualisation t-SNE et UMAP des clusters
- Explication en langage naturel par agent IA (Claude API)

## Prerequis
- Python 3.12
- PyTorch 2.x (CUDA recommande)
- MedMNIST 3.0

## Installation
```bash
pip install -r requirements.txt
```

## Ordre d'execution
Les notebooks doivent etre executes dans l'ordre (NB1 → NB8). Chaque notebook sauvegarde ses modeles et resultats pour le suivant.

```
NB1 (EDA) → NB2 (MLP) → NB3 (CNN) → NB4 (ResNet) → NB5 (ViT) → NB6 (Grad-CAM) → NB7 (Comparaison) → NB8 (Agent)
```

## Lancer l'application
```bash
streamlit run app_streamlit_pathmnist.py
```
