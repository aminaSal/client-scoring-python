"""
╔══════════════════════════════════════════════════════════════════╗
║         CLIENT SCORING — Bank Churn                             ║
║  Goal: generate a churn probability score for each client       ║
║  and segment them by risk level                                 ║
║  Output: client_scores.csv                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ════════════════════════════════════════════════════════════════════════════

# Raw dataset to retrieve readable client info (job, age, balance…)
df_raw = pd.read_csv('bank.csv', sep=';')

# Transformed dataset (feature engineering already applied)
df = pd.read_csv('bank_engineered.csv')

X = df.drop(columns=['churn'])
y = df['churn']

# Stratified split with the same random_state as training
# → guarantees identical train/test indices on every run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train models
# ════════════════════════════════════════════════════════════════════════════

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("✅ Models trained")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Generate per-client scores
# ════════════════════════════════════════════════════════════════════════════

# We use test set indices to link scores back to
# readable fields from the raw dataset (job, age, balance…)
test_idx = X_test.index

# predict_proba()[:, 1] = probability of belonging to class 1 (churner).
# This is the "churn score": the closer to 1, the higher the risk.
scores = pd.DataFrame({
    'age':         df_raw.loc[test_idx, 'age'].values,
    'job':         df_raw.loc[test_idx, 'job'].values,
    'marital':     df_raw.loc[test_idx, 'marital'].values,
    'balance':     df_raw.loc[test_idx, 'balance'].values,
    'actual_churn': y_test.values,
    'score_lr':    lr.predict_proba(X_test_sc)[:, 1].round(4),
    'score_rf':    rf.predict_proba(X_test)[:, 1].round(4),
})

# Average score across both models — simple ensemble combination.
# In production, consider stacking or weighted voting instead.
scores['avg_score'] = ((scores['score_lr'] + scores['score_rf']) / 2).round(4)


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Risk segmentation
# ════════════════════════════════════════════════════════════════════════════

# Thresholds (0.35 and 0.60) are business decisions to tune based on:
#   - cost of a retention action (targeting too broadly = wasted budget)
#   - cost of losing a client (missing a churner = lost revenue)
# These can be optimised using the Precision-Recall curve.
def risk_segment(score):
    if score >= 0.60:    return 'High Risk'      # Priority action
    elif score >= 0.35:  return 'Moderate Risk'  # Monitor
    else:                return 'Low Risk'        # No immediate action

scores['risk_segment'] = scores['avg_score'].apply(risk_segment)

# Sort by descending score: highest-risk clients first.
# This is the priority list for commercial/retention teams.
scores = scores.sort_values('avg_score', ascending=False).reset_index(drop=True)
scores.index += 1  # Rank starts at 1 for readability


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Display & export
# ════════════════════════════════════════════════════════════════════════════

print("\n📊 Top 20 highest-risk clients:")
print(scores.head(20).to_string())

print("\n📈 Segment distribution:")
print(scores['risk_segment'].value_counts().to_string())

# True churn rate per segment — validates that actual churners
# land in the high-risk segments as expected
print("\n🎯 Actual churn rate per segment:")
print(scores.groupby('risk_segment')['actual_churn'].mean().round(3).to_string())

# Export CSV — ready to share with business/retention teams
scores.to_csv('client_scores.csv', index=True, index_label='rank')
print("\n💾 File exported → client_scores.csv")
