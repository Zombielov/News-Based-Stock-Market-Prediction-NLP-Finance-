News-Based Stock Market Prediction



---

# 1. Project Information

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

Le problème central repose sur l'Hypothèse d'Efficience du Marché, qui suggère que toute information publique est déjà intégrée dans les cours. Ce projet explore si les techniques modernes de NLP peuvent néanmoins extraire un « signal » prédictif dans le bruit médiatique pour obtenir un avantage concurrentiel. Cette question est cruciale pour les traders quantitatifs, les analystes financiers et les data scientists, car une amélioration même légère de la précision peut se traduire par des gains économiques significatifs lors d'une application à grande échelle.

Include:

- The context of the problem : L'utilisation des 25 meilleurs titres d'actualité quotidiens pour prévoir la clôture du marché.
- Why the problem is interesting or important : Tester l'efficience du marché face aux nouvelles publiques et la capacité des modèles (Random Forest, TF-IDF, Word2Vec) à extraire un signal exploitable.
- Who might benefit from solving it : Acteurs de la finance (investisseurs, quants) et chercheurs en IA appliquée à l'économie.

---

# 3. Project Goal

Ce projet a pour objectif de développer un système de classification binaire capable de prédire la direction quotidienne de l'indice boursier Dow Jones (DJIA) à partir de l'analyse textuelle des actualités mondiales.

Voici les composantes clés de cet objectif :

Ce que le système prédit : À partir des 25 titres les plus populaires du sous-reddit r/worldnews, le modèle détermine si le marché finira en hausse ou stable (Label 1) ou en baisse (Label 0) pour la séance suivante.

Analyse technique : Le projet compare l'efficacité de différentes méthodes d'extraction de caractéristiques (TF-IDF vs Word2Vec) couplées à des algorithmes de Machine Learning (Random Forest, SVM, Régression Logistique).

Définition d'une solution réussie : Un modèle performant doit non seulement surpasser le hasard (précision > 50 %) de manière statistiquement significative, mais aussi démontrer sa viabilité via un backtest réaliste, en prouvant que le signal capté peut générer une stratégie de trading positive après déduction des frais de transaction.

---

# 4. Task Definition

Define the **machine learning or data analysis task**.

Specify:

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

- **Équilibre des classes (Target_tomorrow)**
  - ~53 % de jours de hausse (1)
  - ~47 % de jours de baisse (0)
  - Dataset globalement équilibré → baseline ≈ 50 %

- **Distribution temporelle**
  - Période : 2008 – 2016
  - Inclut plusieurs régimes de marché (crise, reprise, stabilité)
  - Données séquentielles → dépendances temporelles

- **Variables numériques**
  - Prix (Open, Close, High, Low) : valeurs élevées (indice Dow Jones)
  - Volume : très variable
  - Return_tomorrow : faible amplitude, centré autour de 0

- **Données textuelles**
  - Headlines très variées et bruitées
  - Vocabulaire riche (~18 000 mots)
  - Signal faible et difficile à exploiter

---

## Data Quality

- **Valeurs manquantes**
  - Quelques valeurs manquantes dans certaines colonnes de headlines (Top23 à Top25)
  - Nombre très faible (moins de 10 au total)
  - Traitement : remplacement par des chaînes vides

- **Doublons**
  - Aucun doublon détecté dans les données (news et prix)

- **Cohérence des données**
  - Forte cohérence entre le label et la direction du marché (~98%)
  - Quelques incohérences (~32 observations) dues à :
    - décalages temporels
    - bruit dans les données

- **Équilibre des classes**
  - Dataset légèrement déséquilibré mais proche de l’équilibre :
    - ~53% hausse
    - ~47% baisse
  - Pas de problème majeur de déséquilibre

- **Données textuelles bruitées**
  - Headlines hétérogènes (politique, économie, événements divers)
  - Certaines informations peu pertinentes pour le marché
  - Présence de bruit, redondance et langage non structuré

- **Outliers**
  - Pas d’outliers majeurs détectés dans les variables numériques
  - Les variations extrêmes peuvent être liées à des événements de marché réels
---

# 6. Data Preprocessing

Plusieurs étapes de preprocessing ont été appliquées afin de préparer les données avant la modélisation :

- **Gestion des valeurs manquantes**
  - Remplacement des valeurs manquantes dans les headlines par des chaînes vides
  - Nécessaire pour éviter des erreurs lors du traitement du texte

- **Vérification des doublons**
  - Aucun doublon détecté dans les datasets (news et prix)
  - Permet de garantir la qualité et l’unicité des observations

- **Nettoyage du texte**
  - Conversion en minuscules
  - Suppression de la ponctuation et des caractères spéciaux
  - Permet de standardiser le texte et réduire le bruit

- **Suppression des stopwords**
  - Suppression des mots fréquents non informatifs (ex : "the", "is")
  - Améliore la qualité du signal utile pour le modèle

- **Tokenisation**
  - Transformation du texte en liste de mots (tokens)
  - Nécessaire pour appliquer les méthodes NLP

- **Stemming**
  - Réduction des mots à leur racine (ex : "running" → "run")
  - Permet de regrouper les variantes d’un même mot

- **Création de deux représentations du texte**
  - **Sans stemming (tokens_raw)** → utilisé pour Word2Vec  
  - **Avec stemming (tokens_stemmed)** → utilisé pour TF-IDF  
  - Adaptation du preprocessing selon la méthode NLP utilisée

- **Construction de la variable cible**
  - Création de `Target_tomorrow` basée sur le rendement du lendemain
  - Permet de prédire le futur et éviter le look-ahead bias

- **Split temporel des données**
  - Train : données avant 2015  
  - Test : données après 2015  
  - Respect de l’ordre temporel (pas de fuite d’information)

- **Feature engineering (NLP)**
  - TF-IDF (vecteurs de mots pondérés)
  - Word2Vec (embeddings de mots)
  - Transformation du texte en variables numériques exploitables

- **Standardisation (Word2Vec)**
  - Application d’un StandardScaler avant la régression logistique
  - Améliore la convergence et les performances du modèle
---

# 7. Modeling Approach

Pour résoudre le problème de prédiction de la direction du marché, nous avons adopté une approche de **classification supervisée** combinant des techniques de NLP et des modèles de machine learning.

Le texte ne pouvant pas être utilisé directement, il a été transformé en variables numériques via deux approches :

- **TF-IDF**
  - Représentation basée sur la fréquence des mots et des bigrammes
  - Capture les cooccurrences locales du vocabulaire

- **Word2Vec**
  - Représentation dense des mots (embeddings)
  - Chaque document est représenté par la moyenne des vecteurs de ses mots
  - Capture les relations sémantiques entre les mots
---

## Chosen Models

Plusieurs modèles de machine learning ont été utilisés afin de comparer leurs performances sur les données textuelles :

- **Logistic Regression**
  - Modèle linéaire de classification
  - Sert de baseline solide pour les problèmes NLP
  - Interprétable et rapide à entraîner

- **Random Forest**
  - Ensemble de plusieurs arbres de décision
  - Capable de capturer des relations non linéaires
  - Robuste au bruit mais peut surapprendre

- **Multinomial Naive Bayes**
  - Modèle probabiliste adapté aux données textuelles (TF-IDF)
  - Rapide et efficace sur des données de grande dimension
  - Hypothèse d’indépendance des mots

- **Linear SVM (Support Vector Machine)**
  - Modèle linéaire performant pour les données à haute dimension
  - Souvent efficace avec des représentations TF-IDF
  - Bonne capacité de généralisation

| Modèle                         | Représentation | Accuracy | Balanced Acc | F1-score | MCC   | ROC-AUC | Commentaire |
|--------------------------------|---------------|----------|---------------|----------|-------|---------|-------------|
| Naive Baseline                 | —             | ~50,66 % | ~50 %         | —        | 0     | 0.50    | Référence minimale (classe majoritaire) |
| Word2Vec + Logistic Regression | W2V           | ~53,05 % | ~52,82 %      | ~60,22 % | 0.06  | ~0.53   | Léger signal, bon rappel mais précision faible |
| Word2Vec + Random Forest       | W2V           | ~54,38 % | ~54,12 %      | ~61,95 % | 0.09  | ~0.53   | Meilleur modèle W2V, mais gain limité |
| TF-IDF + Logistic Regression   | TF-IDF        | ~52,52 % | ~52,30 %      | ~59,41 % | 0.05  | ~0.53   | Modèle stable, bon compromis global |
| TF-IDF + Multinomial NB        | TF-IDF        | ~50,66 % | ~50,00 %      | ~67,25 % | 0.00  | ~0.53   | Prédit presque toujours "Up" → recall élevé mais peu informatif |
| TF-IDF + Linear SVM            | TF-IDF        | ~52–53 % | ~52 %         | ~59–60 % | ~0.05 | ~0.52   | Performances proches de LogReg |
| TF-IDF + Random Forest         | TF-IDF        | ~51–52 % | ~51 %         | ~58–60 % | ~0.03 | ~0.51   | Moins efficace que sur W2V |


---

## Modeling Strategy

Un modèle de référence (**baseline**) a été utilisé :

- **Dummy Classifier (classe majoritaire)**
  - Prédit toujours la classe la plus fréquente
  - Performance ≈ **50% d’accuracy**

Permet de vérifier que les modèles apprennent mieux que le hasard.

---

### Hyperparameter tuning

- Aucun tuning complexe n’a été réalisé
- Des paramètres standards ont été utilisés (ex : nombre d’arbres pour Random Forest, régularisation pour Logistic Regression)

Choix volontaire pour :
- garder une approche simple
- éviter l’overfitting
- se concentrer sur la comparaison des modèles

---

### Validation croisée

- Utilisation de **TimeSeriesSplit**
- Validation croisée adaptée aux séries temporelles

Important car :
- respecte l’ordre chronologique
- évite la fuite d’information (data leakage)
---

## Evaluation Metrics

Plusieurs métriques ont été utilisées afin d’évaluer les performances des modèles de manière complète :

- **Accuracy**
  - Proportion de bonnes prédictions
  - Simple à interpréter et utile comme première référence

- **Balanced Accuracy**
  - Moyenne des taux de bonne classification par classe
  - Permet de corriger un éventuel déséquilibre entre les classes

- **Precision**
  - Proportion de prédictions positives correctes
  - Important pour éviter les faux signaux de hausse

- **Recall**
  - Capacité à détecter les jours où le marché monte
  - Utile pour ne pas manquer des opportunités

- **F1-score**
  - Moyenne harmonique entre précision et rappel
  - Donne un bon compromis entre les deux

- **ROC-AUC**
  - Mesure la capacité du modèle à discriminer entre les classes
  - Indépendant du seuil de classification

- **PR-AUC**
  - Aire sous la courbe précision-rappel
  - Pertinent pour évaluer les performances sur les classes positives

- **MCC (Matthews Correlation Coefficient)**
  - Mesure robuste prenant en compte toutes les erreurs
  - Adapté aux problèmes de classification binaire

- **Brier Score**
  - Mesure la qualité des probabilités prédites
  - Évalue la calibration du modèle
 

| Métrique              | Valeur (meilleur modèle test) | Justification |
|----------------------|------------------------------|---------------|
| Accuracy             | 51,72 % (TF-IDF + RF)        | Métrique principale — baseline naïve ≈ 50,66 % |
| Balanced Accuracy    | 51,46 %                      | Corrige le léger déséquilibre de classes |
| Precision / Recall   | 51,70 % / 71,73 %            | Biais du modèle vers la détection des hausses |
| F1-score             | 60,09 %                      | Compromis précision / rappel |
| MCC (Matthews CC)    | 0,032                        | Métrique robuste aux déséquilibres — valeur proche de 0 indique peu de signal |
| ROC-AUC              | 51,78 %                      | Capacité de discrimination — à peine au-dessus du hasard (50 %) |
| PR-AUC               | 51,56 %                      | Qualité probabiliste sur la classe positive |
| Brier Score          | 25,42 %                      | Calibration probabiliste (0 = parfait, 0.25 = baseline) |

---

# 8. Project Structure

- Notebook principal contenant :
  - chargement des données  
  - preprocessing  
  - feature engineering (TF-IDF, Word2Vec)  
  - modélisation  
  - évaluation des performances  

- **daily-news-for-stock-market-prediction.ipynb**    

- **djia-stock-nlp-forecast-using-news.ipynb**  

---

# 9. Installation

import nltk
nltk.download("stopwords")

import kagglehub
dataset_path = kagglehub.dataset_download("aaron7sun/stocknews")

```bash
pip install gensim scikit-learn tqdm nltk
