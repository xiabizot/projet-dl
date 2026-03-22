# Projet Deep Learning — Classification automatisee de tissus cancereux colorectaux

**Diplome Universitaire Sorbonne Data Analytics — Promotion 007 — Mars 2026**

## Contexte

Le cancer colorectal est le troisieme cancer le plus frequent dans le monde. L'analyse histopathologique des biopsies tissulaires reste le standard diagnostique, mais elle est chronophage et sujette a la variabilite inter-observateurs. Automatiser la classification des types tissulaires a partir d'images d'histologie colorees H&E pourrait assister les pathologistes dans le tri de grands volumes de lames, la priorisation des cas suspects et la reduction des delais diagnostiques.

Ce projet explore si des modeles de deep learning peuvent distinguer de maniere fiable 9 types tissulaires a partir de patches d'histologie colorectale en basse resolution (28x28), et evalue les compromis entre complexite du modele, performance diagnostique et deployabilite clinique.

## Dataset

**PathMNIST** (benchmark MedMNIST) — 107 180 patches d'histopathologie de cancer colorectal (28x28, RGB), 9 types tissulaires (adipose, background, debris, lymphocytes, mucus, muscle lisse, muqueuse colique normale, stroma associe au cancer, epithelium d'adenocarcinome colorectal).

- Train : 89 996 images (Hopital A — NCT-CRC-HE-100K)
- Val : 10 004 images (Hopital A)
- Test : 7 180 images (Hopital B — CRC-VAL-HE-7K)

Le split train/test provient de deux hopitaux differents, introduisant un **domain shift** qui est le defi central de ce projet.

Le dataset est telecharge automatiquement a l'execution des notebooks.

## Notebooks

| Notebook | Contenu | Test accuracy | Recall cancer |
|----------|---------|--------------|---------------|
| NB1 — EDA | Exploration des donnees, analyse du domain shift, Q1.1 & Q1.2 | — | — |
| NB2 — MLP | Reseau dense baseline (2352-512-256-128-9) | 68.02% | 0.7745 |
| NB3 — CNN | CNN from scratch + iterations d'augmentation | 91.78% | 0.9570 |
| NB4 — ResNet-18 | Transfer learning : frozen (87.14%) vs fine-tuning | 91.77% | 0.9659 |
| NB5 — ViT | Vision Transformer from scratch, patch 7x7 | 81.98% | 0.8873 |
| NB6 — Grad-CAM | Interpretabilite : ou regardent les modeles | — | — |
| NB7 — Comparaison | Comparaison finale, analyse des metriques, recommandations cliniques | — | — |

## Resultats cles

1. **Le CNN from scratch (91.78%) egale le ResNet fine-tuning (91.77%) avec 25 fois moins de parametres.** La cle est l'augmentation ColorJitter, qui simule les variations de coloration H&E entre hopitaux et adresse directement le domain shift.

2. **L'accuracy seule est trompeuse pour le deploiement clinique.** Le ResNet FT domine sur la detection du cancer (F1 = 0.9632 vs 0.9414 pour le CNN, precision 0.9605 vs 0.9262). En oncologie, manquer un cancer (faux negatif) est bien plus grave qu'une fausse alerte.

3. **Le stroma est universellement difficile** (recall 0.39-0.53 pour tous les modeles). C'est une limite physique de la resolution 28x28 — les textures des tissus conjonctifs sont indistinguables a cette echelle.

4. **Le ViT sans positional embeddings surpasse le ViT avec** (83.40% vs 81.98%). En histologie, l'identite du tissu est dans la texture, pas dans la localisation — un resultat qui contraste fortement avec le NLP ou la position est essentielle.

5. **Le CNN et le ResNet utilisent des strategies visuelles complementaires** (analyse Grad-CAM). Un ensemble des deux pourrait potentiellement surpasser chacun individuellement.

## Equipe

- **Xia Bizot** — NB1 (EDA), NB2 (MLP), NB3 (CNN), NB5 (ViT), NB7 (Comparaison)
- **Camille** — NB4 (ResNet-18), NB6 (Grad-CAM)

## Prerequis

- Python 3.12
- PyTorch >= 2.0 (CUDA recommande)
- MedMNIST >= 3.0

## Installation

```bash
pip install -r requirements.txt
```

## Ordre d'execution

Les notebooks doivent etre executes dans l'ordre (NB1 a NB7). Chaque notebook sauvegarde ses modeles et predictions pour les notebooks suivants.

```
NB1 (EDA) -> NB2 (MLP) -> NB3 (CNN) -> NB4 (ResNet) -> NB5 (ViT) -> NB6 (Grad-CAM) -> NB7 (Comparaison)
```

## Reproductibilite

Tous les notebooks utilisent SEED=42, torch.backends.cudnn.deterministic=True, et sauvegardent leurs meilleurs checkpoints et predictions en fichiers pickle. Les constantes de normalisation sont calculees dans NB1 et reutilisees dans tous les notebooks suivants.
