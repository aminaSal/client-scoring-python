#SCORING CLIENTS                        

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Chargement des données
# ════════════════════════════════════════════════════════════════════════════

# Dataset brut pour récupérer les infos lisibles (job, age, balance…)
df_raw = pd.read_csv('bank.csv', sep=';')

# Dataset transformé (feature engineering déjà appliqué)
df = pd.read_csv('bank_engineered.csv')

X = df.drop(columns=['churn'])
y = df['churn']

# Split stratifié avec le même random_state qu'à l'entraînement
# → garantit exactement les mêmes index train/test à chaque exécution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Entraînement des modèles
# ════════════════════════════════════════════════════════════════════════════

# Normalisation pour la Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("✅ Modèles entraînés")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Génération des scores par client
# ════════════════════════════════════════════════════════════════════════════

# On récupère les index du test set pour relier les scores
# aux infos lisibles du dataset brut (job, age, balance…)
test_idx = X_test.index

# predict_proba()[:, 1] = probabilité d'appartenir à la classe 1 (churner)
# C'est le "score de churn" : plus il est proche de 1, plus le client est à risque.
scores = pd.DataFrame({
    'age':        df_raw.loc[test_idx, 'age'].values,
    'job':        df_raw.loc[test_idx, 'job'].values,
    'marital':    df_raw.loc[test_idx, 'marital'].values,
    'balance':    df_raw.loc[test_idx, 'balance'].values,
    'churn_reel': y_test.values,
    'score_lr':   lr.predict_proba(X_test_sc)[:, 1].round(4),
    'score_rf':   rf.predict_proba(X_test)[:, 1].round(4),
})

# Score moyen des deux modèles — simple façon de combiner les prédictions
# (en production, on pourrait utiliser un stacking ou un vote pondéré)
scores['score_moyen'] = ((scores['score_lr'] + scores['score_rf']) / 2).round(4)


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Segmentation en niveaux de risque
# ════════════════════════════════════════════════════════════════════════════

# Les seuils (0.35 et 0.60) sont des choix métier à adapter selon :
#   - le coût d'une action de rétention (cibler trop large = budget gaspillé)
#   - le coût de la perte d'un client (rater un churner = revenu perdu)
# On peut optimiser ces seuils avec une courbe Précision-Rappel.
def segment_risque(score):
    if score >= 0.60:   return 'Risque élevé'    # Action prioritaire
    elif score >= 0.35: return 'Risque modéré'   # Surveiller
    else:               return 'Risque faible'   # Pas d'action immédiate

scores['segment_risque'] = scores['score_moyen'].apply(segment_risque)

# Tri par score décroissant : les clients les plus à risque en premier
# C'est la liste de priorité pour les équipes commerciales/rétention
scores = scores.sort_values('score_moyen', ascending=False).reset_index(drop=True)
scores.index += 1  # Rang commence à 1 pour la lisibilité


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Affichage et export
# ════════════════════════════════════════════════════════════════════════════

print("\n📊 Top 20 clients à risque :")
print(scores.head(20).to_string())

print("\n📈 Répartition des segments :")
print(scores['segment_risque'].value_counts().to_string())

# Taux de détection réelle par segment
print("\n🎯 Taux de vrais churners par segment :")
print(scores.groupby('segment_risque')['churn_reel'].mean().round(3).to_string())
# → Valide que le modèle envoie bien les vrais churners dans les segments élevés

# Export CSV — prêt à être partagé avec les équipes métier
scores.to_csv('scores_clients.csv', index=True, index_label='rang')
print("\n💾 Fichier exporté → scores_clients.csv")
