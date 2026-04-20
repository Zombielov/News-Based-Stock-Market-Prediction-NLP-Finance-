News-Based Stock Market Prediction



---

# 1. Project Information

Fill in the following information.

- **Project Title:** Analyse prédictive du Dow Jones (DJIA) par le Traitement du Langage Naturel (NLP) : Approches TF-IDF et Word2Vec sur les actualités Reddit.
- **Group Name:** Group 5
- **Group Members:**  
  - Student 1 – Alexia Jonquet
  - Student 2 – Arooba Mushtaq
  - Student 3 – Sania Mestari
  - Student 4 – Vinosha Rajathurai


- **Course Name:** AI In Finance
- **Instructor:** Nicolas De Roux & Mohamed EL FAKIR
- **Submission Date:** 20/04/2026

---

# 2. Project Description

📌 **Instructions:**  
Le problème central repose sur l'Hypothèse d'Efficience du Marché, qui suggère que toute information publique est déjà intégrée dans les cours. Ce projet explore si les techniques modernes de NLP peuvent néanmoins extraire un « signal » prédictif dans le bruit médiatique pour obtenir un avantage concurrentiel. Cette question est cruciale pour les traders quantitatifs, les analystes financiers et les data scientists, car une amélioration même légère de la précision peut se traduire par des gains économiques significatifs lors d'une application à grande échelle.

Include:

- The context of the problem : L'utilisation des 25 meilleurs titres d'actualité quotidiens pour prévoir la clôture du marché.
- Why the problem is interesting or important : Tester l'efficience du marché face aux nouvelles publiques et la capacité des modèles (Random Forest, TF-IDF, Word2Vec) à extraire un signal exploitable.
- Who might benefit from solving it : Acteurs de la finance (investisseurs, quants) et chercheurs en IA appliquée à l'économie.

✏️ **Write your description below:**

Ce projet explore l'utilisation du Traitement du Langage Naturel (NLP) et du Machine Learning pour prédire les mouvements quotidiens de l'indice boursier Dow Jones Industrial Average (DJIA). L'objectif principal est de déterminer si les informations textuelles contenues dans les titres de l'actualité mondiale peuvent servir d'indicateurs fiables pour anticiper la clôture des marchés financiers.

---

# 3. Project Goal

📌 **Instructions:**  
Ce projet a pour objectif de développer un système de classification binaire capable de prédire la direction quotidienne de l'indice boursier Dow Jones (DJIA) à partir de l'analyse textuelle des actualités mondiales.

Voici les composantes clés de cet objectif :

Ce que le système prédit : À partir des 25 titres les plus populaires du sous-reddit r/worldnews, le modèle détermine si le marché finira en hausse ou stable (Label 1) ou en baisse (Label 0) pour la séance suivante.

Analyse technique : Le projet compare l'efficacité de différentes méthodes d'extraction de caractéristiques (TF-IDF vs Word2Vec) couplées à des algorithmes de Machine Learning (Random Forest, SVM, Régression Logistique).

Définition d'une solution réussie : Un modèle performant doit non seulement surpasser le hasard (précision > 50 %) de manière statistiquement significative, mais aussi démontrer sa viabilité via un backtest réaliste, en prouvant que le signal capté peut générer une stratégie de trading positive après déduction des frais de transaction.


✏️ **Write your project goal below:**

L’objectif est de prédire si le Dow Jones va monter ou baisser le lendemain à partir des news quotidiennes, et d’évaluer si ces informations contiennent un signal prédictif utile.

---

# 4. Task Definition

📌 **Instructions:**  
Define the **machine learning or data analysis task**.

Specify:

- **Task Type:** (classification, regression, clustering, etc.)
- **Input:** What data is used as input
- **Output:** What the model predicts
- **Evaluation Metric:** How performance will be measured

✏️ **Fill in the following:**

- **Task Type:**  Classification binaire (supervised learning)
- **Input Variables:**  Les données d’entrée sont les 25 headlines Reddit par jour, prétraitées puis transformées en représentations numériques :
soit via Word2Vec (moyenne des embeddings des mots),
soit via TF-IDF (vecteurs de fréquences pondérées des mots et bigrammes)
- **Target Variable:**  Une variable binaire appelée Target_tomorrow, qui indique la direction du marché le lendemain :
1 → le marché monte (Close(t+1) > Close(t))
0 → le marché baisse
- **Evaluation Metric(s):** Plusieurs métriques sont utilisées pour évaluer les performances du modèle :
Accuracy (métrique principale, baseline ≈ 50%)
Balanced Accuracy (corrige un éventuel déséquilibre)
Precision / Recall / F1-score
ROC-AUC (capacité de discrimination)
MCC (Matthews Correlation Coefficient) 
PR-AUC et Brier Score (qualité probabiliste)

---

# 5. Dataset Description

📌 **Instructions:**  
Describe the dataset used in the project.

Le dataset utilisé dans ce projet est composé de deux sources de données synchronisées dans le temps :

Données textuelles (news)
Chaque observation correspond à un jour de trading et contient :
une date
un label binaire indiquant la direction du marché (0 = baisse, 1 = hausse)
25 titres d’actualité (headlines) issus de Reddit (subreddit WorldNews)

Ces 25 titres représentent l’ensemble des informations marquantes du jour. Ils sont concaténés pour former un document unique, qui sert d’entrée au modèle.

Données financières (marché)
Pour chaque date, on dispose des informations du Dow Jones :
prix d’ouverture (Open)
prix de clôture (Close)
plus haut (High), plus bas (Low)
volume

Ces données permettent de construire la variable cible (direction du marché).
---

## Dataset Overview

Provide general information about the dataset.

Fill in:

- **Number of samples:**  Environ 1989 observations (jours)
- **Number of features:**  25 variables textuelles (Top1 à Top25 : headlines quotidiennes)
variables financières (Open, Close, High, Low, Volume)
variables construites (tokens, TF-IDF, embeddings Word2Vec)
- **Target variable:**  Target_tomorrow (variable binaire) :
1 → le marché monte le lendemain
0 → le marché baisse le lendemain
- **Data source:** Dataset Kaggle :
https://www.kaggle.com/datasets/aaron7sun/stocknews

---

## Feature Description

📌 **Instructions:**  
List and describe the most important variables.

Example table:

| Feature | Description | Type |
|------|------|------|
| age | Age of individual | Numerical |
| income | Annual income | Numerical |
| gender | Gender category | Categorical |

## Feature Description

| Feature            | Description                                                                 | Type                   |
|--------------------|-----------------------------------------------------------------------------|------------------------|
| Date               | Date de l’observation (jour de trading)                                    | Temporelle             |
| Top1 – Top25       | 25 titres d’actualité Reddit du jour                                       | Texte                  |
| corpus_raw         | Texte combiné des headlines (Word2Vec)                                     | Texte                  |
| corpus_stemmed     | Texte combiné avec stemming (TF-IDF)                                       | Texte                  |
| tokens_raw         | Liste de mots nettoyés                                                     | Texte (liste)          |
| tokens_stemmed     | Liste de mots avec stemming                                                | Texte (liste)          |
| Open               | Prix d’ouverture du Dow Jones                                              | Numérique              |
| Close              | Prix de clôture du Dow Jones                                               | Numérique              |
| High               | Prix maximum de la journée                                                 | Numérique              |
| Low                | Prix minimum de la journée                                                 | Numérique              |
| Volume             | Volume de transactions                                                     | Numérique              |
| Return_tomorrow    | Rendement du marché le lendemain                                           | Numérique              |
| Target_tomorrow    | Direction du marché (1 = hausse, 0 = baisse)                               | Catégorielle (binaire) |

## Target Variable

📌 **Instructions:**  
Explain what the model is trying to predict.

Include:

- Variable name
- Meaning
- Possible values (if classification)

La variable cible utilisée dans ce projet est appelée Target_tomorrow.

Elle représente la direction du marché boursier (Dow Jones) le lendemain par rapport au jour courant.

Plus précisément, cette variable est définie à partir du rendement du marché entre deux jours consécutifs :

Si le prix de clôture du lendemain est supérieur à celui du jour courant → Target_tomorrow = 1 (marché en hausse)
Sinon → Target_tomorrow = 0 (marché en baisse)

Il s’agit donc d’un problème de classification binaire, avec deux classes possibles :

1 : Up (hausse du marché)
0 : Down (baisse du marché)

---

## Data Types

📌 **Instructions:**  
Describe the types of variables present in the dataset.

Examples:

- Numerical
- Categorical
- Ordinal
- Text
- Time-series


- **Variables temporelles (Time-series)**
  - **Date** : représente le jour de trading.

- **Variables textuelles (Text)**
  - **Top1 à Top25** : titres d’actualité Reddit  
  - **corpus_raw / corpus_stemmed** : texte combiné des headlines  
  - **tokens_raw / tokens_stemmed** : listes de mots après preprocessing  

- **Variables numériques (Numerical)**
  - **Open, Close, High, Low** : prix du Dow Jones  
  - **Volume** : volume de transactions  
  - **Return_tomorrow** : rendement du marché le lendemain  

- **Variable catégorielle (Categorical / binaire)**
  - **Target_tomorrow** : direction du marché  
    - **1** = hausse  
    - **0** = baisse  



---

## Data Distribution

📌 **Instructions:**  
Describe important distribution characteristics.

Examples:

- Class balance or imbalance
- Skewed numerical variables
- Range of key features

✏️ **Describe the data distribution here**

---

## Data Quality

📌 **Instructions:**  
Mention any issues found in the dataset.

Examples:

- Missing values
- Outliers
- Imbalanced classes
- Duplicate entries

✏️ **Describe any data quality issues here**

---

# 6. Data Preprocessing

📌 **Instructions:**  
Explain the preprocessing steps applied before modeling.

Examples:

- Handling missing values
- Removing duplicates
- Encoding categorical variables
- Normalizing or scaling features
- Feature engineering

For each step briefly explain **why it was necessary**.

✏️ **Describe your preprocessing steps here**

---

# 7. Modeling Approach

📌 **Instructions:**  
Explain how you solved the problem.

---

## Chosen Models

List the models or algorithms used.

Examples:

- Linear Regression
- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Networks

✏️ **List and describe the models used**

---

## Modeling Strategy

📌 **Instructions:**  
Explain:

- Why you selected these models
- Whether you used a baseline model
- If hyperparameter tuning was performed
- Whether cross-validation was used

✏️ **Explain your modeling strategy**

---

## Evaluation Metrics

📌 **Instructions:**  
Specify the metrics used to evaluate model performance.

Examples:

- Accuracy
- Precision / Recall
- F1-score
- ROC-AUC
- Mean Absolute Error
- RMSE

Also explain **why these metrics are appropriate**.

✏️ **Describe your evaluation metrics**

---

# 8. Project Structure

📌 **Instructions:**  
Explain how the repository is organized.


If you added additional folders, explain them.

---

# 9. Installation

📌 **Instructions:**  
Explain how to install project dependencies.

Example:

```bash
pip install -r requirements.txt
