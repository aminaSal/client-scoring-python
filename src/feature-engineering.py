# FEATURE ENGINEERING
# Objective: transform raw data into features usable by a Machine Learning model

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# ── Chargement ───────────────────────────────────────────────────────────────
df = pd.read_csv('bank.csv', sep=';')
print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Suppression des colonnes à problème
# ════════════════════════════════════════════════════════════════════════════

# ⚠ 'duration' = DATA LEAK : durée de l'appel connue uniquement APRÈS le
# contact. En production, on ne connaît pas cette valeur à l'avance.
# L'inclure ferait "tricher" le modèle → on la supprime.
df.drop(columns=['duration'], inplace=True)
print("🗑  'duration' supprimée (data leak)\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Feature engineering sur 'pdays'
# ════════════════════════════════════════════════════════════════════════════

# 'pdays' = nb de jours depuis le dernier contact d'une campagne précédente.
# La valeur -1 signifie "jamais contacté auparavant".
# Problème : mélanger -1 (absence d'info) et des vraies valeurs numériques
# dans la même colonne brouille le signal pour le modèle.

# → Créer une variable binaire qui capte l'information principale :
#   "ce client a-t-il déjà été contacté ?"
df['was_contacted'] = (df['pdays'] != -1).astype(int)
# was_contacted = 1 : client déjà dans une campagne précédente
# was_contacted = 0 : nouveau client (jamais contacté)

# → Remplacer les -1 par 0 pour que pdays reste utilisable comme numérique
#   (les clients non contactés auront juste pdays = 0)
df['pdays'] = df['pdays'].replace(-1, 0)

print("✅ 'pdays' : valeurs -1 remplacées par 0")
print(f"✅ 'was_contacted' créée — répartition :\n{df['was_contacted'].value_counts().to_string()}\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Encodage des variables binaires (yes/no)
# ════════════════════════════════════════════════════════════════════════════

# Ces colonnes n'ont que 2 modalités → simple mapping 0/1
# Pas besoin de LabelEncoder, un map() suffit et est plus lisible.
binary_cols = ['default', 'housing', 'loan', 'y']

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
    print(f"✅ '{col}' encodée : no→0, yes→1")

# Renommer la cible pour plus de clarté
df.rename(columns={'y': 'churn'}, inplace=True)
print()


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Encodage ordinal de 'education'
# ════════════════════════════════════════════════════════════════════════════

# 'education' a un ordre naturel : primary < secondary < tertiary
# → Encodage ordinal pour que le modèle capte cette hiérarchie.
# 'unknown' est traité comme une valeur médiane (secondary) car
# on ne peut pas supposer le niveau.
education_order = ['unknown', 'primary', 'secondary', 'tertiary']
edu_map = {val: idx for idx, val in enumerate(education_order)}
df['education'] = df['education'].map(edu_map)

print("✅ 'education' encodée ordinalement :")
print(f"   {edu_map}\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Encodage du mois (cyclique)
# ════════════════════════════════════════════════════════════════════════════

# Le mois est une variable cyclique : décembre est "proche" de janvier.
# Un simple entier (1→12) ne capte pas cette circularité.
# Astuce : encoder avec sin/cos pour que le modèle comprenne la continuité.
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
df['month_num'] = df['month'].map(month_map)
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

# Supprimer les colonnes intermédiaires devenues inutiles
df.drop(columns=['month', 'month_num'], inplace=True)
print("✅ 'month' encodé cycliquement → 'month_sin' + 'month_cos'")
print("   (sin/cos préservent la continuité entre déc. et jan.)\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — One-Hot Encoding des variables nominales
# ════════════════════════════════════════════════════════════════════════════

# Les variables nominales n'ont pas d'ordre → One-Hot Encoding.
# drop_first=True évite la multicolinéarité parfaite (dummy variable trap) :
# si on a N modalités, N-1 colonnes suffisent à tout représenter.
nominal_cols = ['job', 'marital', 'contact', 'poutcome']

df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
print(f"✅ One-Hot Encoding appliqué à : {nominal_cols}")
print(f"   Nouvelles colonnes créées par get_dummies :")
ohe_cols = [c for c in df.columns if any(c.startswith(f"{col}_") for col in nominal_cols)]
print(f"   {ohe_cols}\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Récapitulatif & export
# ════════════════════════════════════════════════════════════════════════════

print("═" * 60)
print(f"📐 Shape finale : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n📋 Colonnes finales :")
for col in df.columns:
    dtype = str(df[col].dtype)
    print(f"   {col:<30} {dtype}")

print(f"\n🎯 Distribution de la cible 'churn' :")
vc = df['churn'].value_counts()
for val, cnt in vc.items():
    print(f"   {val} → {cnt} ({cnt/len(df)*100:.1f}%)")

# Séparation features / cible
X = df.drop(columns=['churn'])
y = df['churn']

# Split train / test stratifié (stratify=y garantit les mêmes proportions
# churners/non-churners dans les deux ensembles → important sur données déséquilibrées)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️  Split train/test (80/20, stratifié) :")
print(f"   X_train : {X_train.shape} | y_train churn rate : {y_train.mean()*100:.1f}%")
print(f"   X_test  : {X_test.shape}  | y_test  churn rate : {y_test.mean()*100:.1f}%")

# Export du dataset transformé pour la suite
df.to_csv('bank_engineered.csv', index=False)
print(f"\n💾 Dataset transformé sauvegardé → 'bank_engineered.csv'")
print("═" * 60)
