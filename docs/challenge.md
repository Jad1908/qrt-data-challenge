# Challenge : Prédiction des performances d'allocations d'actifs
> **Mantra du challenge :** Faire confiance, ou parier contre ?

Dans le monde du trading systématique, les allocations d'actifs sont omniprésentes — mais la qualité des signaux fait toute la différence. 

Chaque jour, les traders sont submergés d'allocations candidates : des constructions de portefeuille fondées sur des signaux prédictifs récents, des flux de liquidité ou des schémas historiques. Certaines de ces allocations seront profitables lors de la prochaine session de trading. D'autres sous-performeront ou, pire, seront perdantes de manière si consistante que parier contre elles serait le choix le plus profitable.

Ce challenge se centre autour d'une question simple mais cruciale : **Pouvez-vous prédire si une allocation d'actifs donnée mérite d'être suivie ou s'il vaut mieux parier contre ?**

---

## 💡 Qu'est-ce qu'une allocation d'actifs ?

Une allocation d'actifs peut être définie comme une **méthode systématique de construction de portefeuille** utilisant des signaux ou des règles prédéfinies. 

Dans ce challenge :
* Chaque allocation est définie par un **vecteur de poids** (positifs ou négatifs), fixés chaque jour et tenus pour toute une session de trading.
* D'un jour à l'autre, une allocation peut rééquilibrer ses poids selon une certaine proportion, appelée **« turnover »**.
* Les performances journalières d'une allocation d'actifs représentent les performances agrégées des positions pondérées et rebalancées chaque jour.

### Définition Mathématique

Pour un jour $t$, une allocation $S$, et $M$ actifs dans un univers de trading, soient :

1. **Les poids de l'allocation $S$ au jour $t$ :**
   $$w_{S,t} = (w_{S,t,1}, w_{S,t,2}, ..., w_{S,t,M})$$

2. **Le rendement d'un actif $i$ du jour $t$ au jour $t+1$ :**
   $$r_{i,t+1}$$

Le **rendement réalisé** de l'allocation $S$ à $t+1$ est donné par :
$$r_{S,t+1} = \sum_{i=1}^{M} w_{S,t,i} \times r_{i,t+1}$$

---

## 🎯 Objectif du Challenge

Chaque ligne du dataset représente une journée et une allocation d'actifs, matérialisée par son portefeuille de poids construit et rebalancé ce jour-là. Les données fournissent l'empreinte historique des **20 jours précédents** (performances passées, volumes pondérés et turnover médian).

L'objectif est d'utiliser cette empreinte pour **prédire le signe du rendement futur** de cette allocation ($TARGET$) :
* **Modèle prédit une performance positive ($1$) :** Faire confiance à l'allocation et conserver ses poids pour la prochaine session.
* **Modèle prédit une performance négative ($0$) :** Parier contre l'allocation et jouer l'inverse de ses poids.

---

## 📊 Métrique d'évaluation

La métrique choisie est l'**Accuracy** (Précision), qui détermine à quel point un modèle prédit correctement la direction (le signe) de la performance future, indépendamment de sa magnitude.

Pour chaque ligne $i$ indexée par un timestamp $t$ et une allocation $S$, il est attendu :
* **$1$** si la prédiction est positive ($x > 0$).
* **$0$** si la prédiction est négative ($x \le 0$).

$$Accuracy = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}_{[\text{sign}(\hat{r}_i) = \text{sign}(r_i)]}$$

$$Accuracy = \frac{1}{T \times M} \sum_{t=1}^{T} \sum_{S=1}^{M} \mathbf{1}_{[\text{sign}(\hat{r}_{S,t+1}) = \text{sign}(r_{S,t+1})]}$$

**Où :**
* $N$ : le nombre de lignes du dataset.
* $\text{sign}(x) = 1$ si $x > 0$, sinon $0$.
* $\mathbf{1}_{[\dots]}$ : la fonction indicatrice (vaut 1 si la condition est vraie, 0 sinon).
* $r_i = r_{S,t+1}$ : le vrai rendement futur.
* $\hat{r}_i = \hat{r}_{S,t+1}$ : le rendement futur prédit.

---

## 🗄️ Description des Données

Le dataset se présente sous la forme de **séries temporelles** avec un multi-index `(date, allocation)`. 

### Liste des Colonnes

| Nom de la colonne | Description |
| :--- | :--- |
| `ROW_ID` | Identifiant unique (permet de mapper $X$ et $Y$). |
| `TS` | Timestamp du snapshot (dates anonymisées, ex: `DATE_0001`, `DATE_0002`). |
| `ALLOCATION` | Nom de l'allocation (identifiant stable dans le temps, ex: `ALLOCATION_01`). |
| `RET_{i}` | Rendement de l'allocation au jour passé $i$ (avec $i \in [1, \dots, 20]$). |
| `SIGNED_VOLUME_{i}`| Volume signé de l'allocation au jour passé $i$ (avec $i \in [1, \dots, 20]$). |
| `MEDIAN_DAILY_TURNOVER` | Turnover médian de l'allocation sur les 20 derniers jours. |
| `GROUP` | Groupe d'appartenance anonymisé de l'allocation. |
| `TARGET` | **Variable cible** : Le rendement futur de l'allocation sur la prochaine session. |

### Détails Techniques (Volumes & Turnover)

À chaque jour $t$, chaque allocation $S$ respecte la contrainte de normalisation suivante :
$$\forall t , \forall S : \sum_{i=1}^{M} |w_{S,i,t}| = 1$$

* **Volume Signé (`SIGNED_VOLUME`) :** Calculé à partir du volume total échangé sur le marché ($V_{i,t}$) pour chaque instrument $i$, rescalé de manière « roulante » pour assurer la comparabilité :
  $$V_{S,t} = \sum_{i=1}^{M} w_{S,t,i} \times V_{i,t}$$

* **Turnover Journalier Median (`MEDIAN_DAILY_TURNOVER`) :** Mesure l'intensité du rééquilibrage.
  $$TO_{S,t} = \sum_{i=1}^{M} |w_{S,t,i} - w_{S,t-1,i}|$$
  $$MDT_{S,t} = \text{median}(TO_{S,t}, TO_{S,t-1}, \dots, TO_{S,t-20})$$

---

## 📂 Fichiers mis à disposition

* **`X_train.csv`** : Les données d'entraînement (527 073 observations).
* **`y_train.csv`** : La target d'entraînement associée.
* **`X_test.csv`** : Les données de test (31 870 observations).
* **`sample_submission.csv`** : Un exemple de fichier de soumission aléatoire au bon format.
* **`benchmark_submission.ipynb`** : Un notebook de référence.

---

## 🚀 Description du Baseline Benchmark

Le notebook `benchmark_submission.ipynb` fourni contient l'ingénierie des caractéristiques (*feature engineering*) et les modèles de base suivants :

1. **Features additionnelles créées :**
   * Moyenne journalière des performances passées des allocations à différents horizons.
   * Moyenne journalière des performances passées de *toutes* les allocations (effet marché).
   * Volatilité passée des allocations sur les 20 derniers jours.
   * Moyenne des volatilités passées de l'ensemble des allocations.

2. **Modèles testés :**
   * Un modèle **Ridge Regression** calibré sur toutes les caractéristiques.
   * Un modèle **LightGBM** optimisé avec une stratégie de validation croisée (*Cross-Validation*).

> 📊 **Score de référence (Public Leaderboard) :** Le modèle LightGBM atteint une accuracy de **0.5079**.