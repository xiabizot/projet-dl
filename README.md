# Projet Deep Learning - Classification de tissus cancereux colorectaux

**Diplome Universitaire Sorbonne Data Analytics - Promotion 007 - Mars 2026**


https://github.com/xiabizot/projet-dl


## Contexte

Le cancer est un problème de santé publique. Le cancer colorectal est le troisieme cancer le plus frequent dans le monde. L'analyse des biopsies reste le standard diagnostique, mais elle est chronophage et sujette a la variabilite inter-observateurs. Le deep learning accompagne les domaines de la santé dans ses enjeux technologiques et peut par exemple aider a trier les lames plus vite, a reperer les cas suspects en priorite, et a rendre les criteres de classification plus homogenes à grande échelle.

Ce projet explore si des modeles de deep learning peuvent distinguer de manière fiable et efficace 9 types de tissus a partir de lames d'histologie colorectale en basse resolution (28x28), et evalue les compromis entre complexite du modele, performance diagnostique et deployabilite clinique.

## Problematiques

**1. Pourquoi le deep learning ?** Classer des tissus sur des lames d'histologie, c'est un travail d'expert qui prend du temps. Les pathologistes font face a des volumes croissants de biopsies et la lecture reste subjective (deux medecins peuvent diverger sur le meme echantillon). Le deep learning peut automatiser le tri, standardiser les criteres et detecter des patterns subtils. Mais ca souleve des questions concretes : est-ce que ca marche quand les donnees viennent d'un autre hopital ? Et quelle metrique regarder pour savoir si on peut faire confiance au modele ?

**2. Le domain shift** Nos modeles sont entraines sur les images d'un hopital A mais evalues sur celles d'un hopital B. Les protocoles de coloration, les scanners, la preparation des lames different d'un site a l'autre, la population differe d'un site à l'autre. Resultat : un modele qui atteint 99% en validation peut chuter de 7 a 10 points sur le test. C'est le probleme central du projet - un modele qui ne generalise pas entre hopitaux a peu d'utilité clinique.

**3. Quelle metrique pour le diagnostic ?** L'accuracy globale donne une vision trop optimiste. Un modele a 92% peut manquer 1 cancer sur 10 si ses erreurs se concentrent sur les classes critiques. En oncologie, rater un cancer (faux negatif) est bien plus grave qu'une fausse alerte (faux positif = examen supplementaire). On s'interesse donc au recall sur le cancer, mais aussi a la precision et au F1 score par classe pour avoir une image complete.

**4. Resolution vs complexite** A 28x28 pixels, on perd beaucoup de details. Certains tissus (stroma, muscle lisse, debris) se ressemblent trop a cette echelle pour qu'un modele les distingue. La question est de savoir si des architectures plus puissantes (transfer learning, Transformers) compensent cette perte, ou si la resolution impose un plafond indepassable.

## Dataset

**PathMNIST** (benchmark MedMNIST) 107 180 patches d'histopathologie de cancer colorectal (28x28, RGB), 9 types de tissus (adipose, background, debris, lymphocytes, mucus, muscle lisse, muqueuse colique normale, stroma associe au cancer, epithelium d'adenocarcinome colorectal).

- Train : 89 996 images (Hopital A — NCT-CRC-HE-100K)
- Val : 10 004 images (Hopital A)
- Test : 7 180 images (Hopital B — CRC-VAL-HE-7K)

Le split train/test provient de deux hopitaux differents, introduisant un **domain shift** qui est l'un des défis de ce projet.

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

4. **Le ViT sans positional embeddings surpasse le ViT avec** (83.40% vs 81.98%). En histologie, l'identite du tissu est dans la texture, pas dans la localisation — un resultat qui contraste avec le NLP ou la position est essentielle.

5. **Le CNN et le ResNet utilisent des strategies visuelles complementaires** (analyse Grad-CAM). Un ensemble des deux pourrait potentiellement surpasser chacun individuellement.

## Prerequis

- Python 3.12
- PyTorch >= 2.0 (CUDA recommande)
- MedMNIST >= 3.0

## Installation

```bash
pip install -r requirements.txt
```

## Ordre d'execution

Les notebooks doivent etre executes dans l'ordre (NB1 a NB7). Chaque notebook sauvegarde ses modeles et predictions pour les notebooks suivants, jusqu'au notebook de comparaison finale.

```
NB1 (EDA) -> NB2 (MLP) -> NB3 (CNN) -> NB4 (ResNet) -> NB5 (ViT) -> NB6 (Grad-CAM) -> NB7 (Comparaison)
```

## Reproductibilite

Tous les notebooks utilisent SEED=42, torch.backends.cudnn.deterministic=True, et sauvegardent leurs meilleurs checkpoints et predictions en fichiers pickle. Les constantes de normalisation sont calculees dans NB1 et reutilisees dans tous les notebooks suivants.

## Application Cell.IA

**C**lassification **E**xplicable de **L**ames par **L**earning — **I**ntelligence **A**rtificielle

Application Streamlit deployee en ligne : https://cell-ia.streamlit.app

Cell.IA met en production les modeles entraines dans les notebooks sous forme d'un agent IA interactif. L'application est concue pour etre consultable sur mobile et propose deux niveaux d'explication (standard et junior pour les adolescents).

### Les 4 onglets

**1. Classification**
Clique sur un microbe kawaii (une image aleatoire de la classe est chargee depuis le test set), utilise l'analyse aleatoire, ou glisse ta propre image. Cell.IA classifie le tissu et affiche :
- La classe predite avec un badge colore
- Un signal clinique : vert (tissu sain), jaune (classe a faible recall, supervision recommandee), rouge (cancer detecte)
- La confiance du modele (softmax)
- L'incertitude estimee par **Monte Carlo Dropout** : 20 forward passes avec dropout active a l'inference. Si les 20 predictions convergent, le modele est stable. Si elles divergent, le modele hesite et une verification humaine est recommandee
- Les metriques de la classe predite (precision, recall, F1-score) et les moyennes globales (accuracy, macro F1, weighted F1)
- La distribution des probabilites sur les 9 classes
- Deux boutons d'explication par agent IA : "Cell.IA explique" (technique, pour le pathologiste) et "Cell.IA version junior" (accessible, pour les adolescents)

**2. Grad-CAM**
De la **black-box** a la **glass-box**. Le Grad-CAM montre ou chaque modele regarde pour prendre sa decision :
- 3 colonnes : image originale, heatmap CNN v1, heatmap ResNet FT
- Score de concordance (cosine similarity entre les deux heatmaps) : si les modeles regardent au meme endroit, la prediction est fiable
- Comparaison avec l'image la plus similaire du test set et son Grad-CAM
- Explication par agent IA du Grad-CAM (avec envoi de l'image et de la heatmap a l'API Claude pour analyse visuelle)
- Exigences cliniques : Trust (le medecin comprend), Safety (bon recall + bon endroit), Liability (raisonnement explicable)

**3. Embeddings**
Visualisation de l'espace des representations apprises par le CNN v1 :
- 5 images les plus similaires (cosine similarity sur les embeddings 128D) avec nom de classe et pourcentage de similarite
- Carte **t-SNE** : 7180 images projetees de 128D vers 2D, preservant les voisinages locaux. Les clusters bien separes correspondent aux classes a fort recall
- Carte **UMAP** : meme projection mais preservant aussi la structure globale. Les distances entre clusters sont plus significatives
- Position de l'image analysee (etoile rouge) sur la carte t-SNE avec ses 5 plus proches voisins

**4. Resultats**
Tableau comparatif des 7 modeles (MLP, CNN sans aug, CNN v1, ResNet frozen, ResNet FT, ViT, ViT sans pos) avec test accuracy, precision cancer, recall cancer, F1 cancer, parametres et ecart par rapport au meilleur modele. Classification reports complets CNN v1 vs ResNet FT cote a cote avec colonne delta F1. Tableau des moyennes globales (macro avg, weighted avg, accuracy).

### Lancer l'application en local

```bash
streamlit run app_streamlit_pathmnist.py
```

Les modeles sont telecharges automatiquement depuis Google Drive au premier lancement.