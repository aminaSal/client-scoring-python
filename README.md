# Bank Churn Scoring

bank-churn-scoring/
│
├── data/
│   ├── bank.csv                  # Dataset brut (séparateur ;)
│   └── bank_engineered.csv       # Dataset après feature engineering
│
├── eda_visualisations.py         # Exploration & visualisations
├── feature_engineering.py        # Transformation des features
├── modelisation.py               # Entraînement & comparaison des modèles
├── scoring_clients.py            # Génération des scores par client
│
├── outputs/
│   ├── eda_numeriques.png
│   ├── eda_categorielles.png
│   ├── comparaison_modeles.png
│   └── scores_clients.csv
│
└── README.md

📊 Dataset

Source : Bank Marketing Dataset — UCI / Kaggle
4 521 clients, 16 features + 1 cible (y : souscription → proxy du churn)
Déséquilibre des classes : 88.5% non-churners / 11.5% churners


⚙️ Feature Engineering
TransformationVariable(s)RaisonSuppressiondurationData leak — inconnue avant l'appelBinaire 0/1default, housing, loan, y2 modalités uniquementOrdinaleducationOrdre naturel primary < secondary < tertiaryCyclique sin/cosmonthContinuité décembre → janvierOne-Hotjob, marital, contact, poutcomePas d'ordre entre modalitésNouvelle featurewas_contactedExtrait le sens de pdays = -1

🤖 Modèles
Deux modèles entraînés avec class_weight='balanced' pour gérer le déséquilibre des classes.
MétriqueLogistic RegressionRandom ForestAUC-ROC0.71520.7242Avg Precision0.31330.3163Recall0.5770.135Precision0.1890.636F1-score0.2850.222
Modèle retenu : Logistic Regression — meilleur recall (4× plus de churners détectés), interprétable, et adapté à un contexte où rater un churner coûte plus cher qu'une fausse alarme.

🎯 Scoring client
Chaque client reçoit un score de probabilité de churn (moyenne LR + RF), segmenté en 3 niveaux :
SegmentSeuilAction recommandée🔴 Risque élevé≥ 0.60Appel commercial prioritaire🟠 Risque modéré0.35 – 0.60Suivi & offre ciblée🟢 Risque faible< 0.35Pas d'action immédiate

🚀 Lancer le projet
bash# 1. Installer les dépendances
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Explorer les données
python eda_visualisations.py

# 3. Préparer les features
python feature_engineering.py

# 4. Entraîner et comparer les modèles
python modelisation.py

# 5. Générer les scores clients
python scoring_clients.py

🔭 Pistes d'amélioration

Optimisation du seuil de décision via la courbe Précision-Rappel
Interprétabilité avec SHAP values
Test de modèles boostés (XGBoost, LightGBM)
Rééchantillonnage avec SMOTE pour mieux gérer le déséquilibre
