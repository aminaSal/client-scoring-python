"""
╔══════════════════════════════════════════════════════════════════╗
║         FEATURE ENGINEERING — Bank Churn                        ║
║  Goal: transform raw data into features usable by an ML model   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('bank.csv', sep=';')
print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Drop problematic columns
# ════════════════════════════════════════════════════════════════════════════

# ⚠ 'duration' = DATA LEAK: call duration is only known AFTER the contact.
# In production, this value is unavailable at prediction time.
# Including it would cause the model to "cheat" → drop it.
df.drop(columns=['duration'], inplace=True)
print("🗑  'duration' dropped (data leak)\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature engineering on 'pdays'
# ════════════════════════════════════════════════════════════════════════════

# 'pdays' = number of days since the last contact from a previous campaign.
# The value -1 means "never contacted before".
# Problem: mixing -1 (absence of info) and real numeric values in the same
# column confuses the model signal.

# → Create a binary variable capturing the key information:
#   "has this client been contacted before?"
df['was_contacted'] = (df['pdays'] != -1).astype(int)
# was_contacted = 1 → client already in a previous campaign
# was_contacted = 0 → new client (never contacted)

# → Replace -1 with 0 so pdays remains usable as a numeric feature
df['pdays'] = df['pdays'].replace(-1, 0)

print("✅ 'pdays': -1 values replaced with 0")
print(f"✅ 'was_contacted' created — distribution:\n{df['was_contacted'].value_counts().to_string()}\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Encoding binary variables (yes/no)
# ════════════════════════════════════════════════════════════════════════════

# These columns have only 2 modalities → simple 0/1 mapping.
# No need for LabelEncoder; a map() is cleaner and more readable.
binary_cols = ['default', 'housing', 'loan', 'y']

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
    print(f"✅ '{col}' encoded: no→0, yes→1")

# Rename target for clarity
df.rename(columns={'y': 'churn'}, inplace=True)
print()


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Ordinal encoding of 'education'
# ════════════════════════════════════════════════════════════════════════════

# 'education' has a natural order: primary < secondary < tertiary.
# → Ordinal encoding lets the model capture this hierarchy.
# 'unknown' is treated as an intermediate value (secondary) since
# we cannot assume the actual level.
education_order = ['unknown', 'primary', 'secondary', 'tertiary']
edu_map = {val: idx for idx, val in enumerate(education_order)}
df['education'] = df['education'].map(edu_map)

print("✅ 'education' ordinally encoded:")
print(f"   {edu_map}\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Cyclical encoding of 'month'
# ════════════════════════════════════════════════════════════════════════════

# Month is a cyclical variable: December is "close" to January.
# A simple integer (1→12) misses this circularity.
# Trick: encode with sin/cos so the model understands the continuity.
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
df['month_num'] = df['month'].map(month_map)
df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

# Drop intermediate columns no longer needed
df.drop(columns=['month', 'month_num'], inplace=True)
print("✅ 'month' cyclically encoded → 'month_sin' + 'month_cos'")
print("   (sin/cos preserve continuity between Dec. and Jan.)\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — One-Hot Encoding for nominal variables
# ════════════════════════════════════════════════════════════════════════════

# Nominal variables have no order → One-Hot Encoding.
# drop_first=True avoids perfect multicollinearity (dummy variable trap):
# N modalities can be fully represented with N-1 columns.
nominal_cols = ['job', 'marital', 'contact', 'poutcome']

df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
print(f"✅ One-Hot Encoding applied to: {nominal_cols}")
ohe_cols = [c for c in df.columns if any(c.startswith(f"{col}_") for col in nominal_cols)]
print(f"   New columns created: {ohe_cols}\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Summary & export
# ════════════════════════════════════════════════════════════════════════════

print("═" * 60)
print(f"📐 Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Final columns:")
for col in df.columns:
    print(f"   {col:<30} {str(df[col].dtype)}")

print(f"\n🎯 Target 'churn' distribution:")
vc = df['churn'].value_counts()
for val, cnt in vc.items():
    print(f"   {val} → {cnt} ({cnt/len(df)*100:.1f}%)")

# Features / target split
X = df.drop(columns=['churn'])
y = df['churn']

# Stratified train/test split — stratify=y ensures the same churn ratio
# in both sets → critical on imbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️  Train/test split (80/20, stratified):")
print(f"   X_train: {X_train.shape} | y_train churn rate: {y_train.mean()*100:.1f}%")
print(f"   X_test:  {X_test.shape}  | y_test  churn rate: {y_test.mean()*100:.1f}%")

# Export transformed dataset for next steps
df.to_csv('bank_engineered.csv', index=False)
print(f"\n💾 Transformed dataset saved → 'bank_engineered.csv'")
print("═" * 60)
