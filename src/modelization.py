"""
╔══════════════════════════════════════════════════════════════════╗
║         MODELLING — Bank Churn                                  ║
║  Models: Logistic Regression vs Random Forest                   ║
║  Output: model_comparison.png                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              precision_recall_curve, average_precision_score,
                              classification_report)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & split
# ════════════════════════════════════════════════════════════════════════════

# Start from the dataset already transformed by feature_engineering.py
df = pd.read_csv('bank_engineered.csv')

X = df.drop(columns=['churn'])
y = df['churn']

# Stratified split: stratify=y ensures the churn proportion (11.5%)
# is identical in both train and test sets.
# Without stratification, the test set might have too few churners.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
print(f"   Churn rate — train: {y_train.mean()*100:.1f}% | test: {y_test.mean()*100:.1f}%\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Scaling (Logistic Regression only)
# ════════════════════════════════════════════════════════════════════════════

# Logistic Regression is sensitive to feature scales: a feature ranging
# 0–71000 (balance) would dominate a binary feature (0–1).
# StandardScaler centres (mean=0) and scales (std=1) each feature.
# ⚠ IMPORTANT: fit_transform on TRAIN only, then transform on TEST.
#   Fitting on test data would leak test statistics into training (data leak).
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Random Forest is scale-invariant → no scaling needed.


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train models
# ════════════════════════════════════════════════════════════════════════════

# ── Logistic Regression ──────────────────────────────────────────────────────
# class_weight='balanced': automatically compensates for class imbalance
#   by giving more weight to churners (minority) during optimisation.
# Without it, the model would always predict 0 and reach 88.5% accuracy
#   by never detecting a single churner.
# max_iter=1000: increased from default (100) to ensure convergence.
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_sc, y_train)
print("✅ Logistic Regression trained")

# ── Random Forest ────────────────────────────────────────────────────────────
# n_estimators=200: 200 decision trees → more stable than default (100)
#   without excessive computational cost.
# class_weight='balanced': same rationale as for LR.
# n_jobs=-1: uses all available CPU cores → parallel training.
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("✅ Random Forest trained\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Predictions
# ════════════════════════════════════════════════════════════════════════════

# predict() → binary class (0 or 1), useful for standard metrics
# predict_proba()[:, 1] → probability of being a churner (column 1 = positive class)
#   richer output: allows ROC/PR curves and threshold tuning
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Metrics
# ════════════════════════════════════════════════════════════════════════════

def get_metrics(y_true, y_pred, y_proba, name):
    """
    Compute all relevant metrics for a binary classification model.

    Why so many metrics?
    ────────────────────
    On imbalanced data, accuracy alone is misleading.
    A model that always predicts "non-churner" reaches 88.5% accuracy!

    - AUC-ROC      : overall ability to separate classes, threshold-independent.
                     0.5 = random, 1.0 = perfect.
    - Avg Precision: summarises the Precision-Recall curve → more robust than
                     AUC-ROC on heavily imbalanced datasets.
    - Precision    : of all predicted churners, how many truly are?
                     (avoid false alarms = cost of unnecessary retention actions)
    - Recall       : of all actual churners, how many are detected?
                     (missing a churner = lost revenue)
    - F1-score     : harmonic mean of Precision/Recall → balances both.
    """
    cm  = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    return {
        'name':          name,
        'auc_roc':       roc_auc_score(y_true, y_proba),
        'avg_precision': average_precision_score(y_true, y_proba),
        'precision_1':   rep['1']['precision'],
        'recall_1':      rep['1']['recall'],
        'f1_1':          rep['1']['f1-score'],
        'accuracy':      rep['accuracy'],
        'cm':            cm,
        'fpr': fpr, 'tpr': tpr,
        'prec_curve': prec_c, 'rec_curve': rec_c,
    }

m_lr = get_metrics(y_test, lr_pred, lr_proba, 'Logistic Regression')
m_rf = get_metrics(y_test, rf_pred, rf_proba, 'Random Forest')

for m in [m_lr, m_rf]:
    print(f"{'='*42}\n{m['name']}")
    print(f"  AUC-ROC       : {m['auc_roc']:.4f}")
    print(f"  Avg Precision : {m['avg_precision']:.4f}")
    print(f"  Precision     : {m['precision_1']:.4f}")
    print(f"  Recall        : {m['recall_1']:.4f}")
    print(f"  F1-score      : {m['f1_1']:.4f}")
    print(f"  Accuracy      : {m['accuracy']:.4f}")

fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\n🌳 Top 10 features (Random Forest):")
print(fi.head(10).to_string())


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Comparison visualisation
# ════════════════════════════════════════════════════════════════════════════

colors_m = {'Logistic Regression': '#5DADE2', 'Random Forest': '#F39C12'}

sns.set_theme(style='whitegrid', font_scale=0.9)
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
                     'axes.edgecolor': '#DEE2E6', 'grid.color': '#E9ECEF'})

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Model Comparison — Logistic Regression vs Random Forest',
             fontsize=14, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Summary table ────────────────────────────────────────────────────────────
# A recap table at the top gives an immediate read of relative performance
# without having to read every chart individually.
ax0 = fig.add_subplot(gs[0, :])
ax0.axis('off')
rows = [
    ['AUC-ROC',       f"{m_lr['auc_roc']:.4f}",       f"{m_rf['auc_roc']:.4f}",       '🌳 RF' if m_rf['auc_roc'] > m_lr['auc_roc'] else '📈 LR'],
    ['Avg Precision', f"{m_lr['avg_precision']:.4f}", f"{m_rf['avg_precision']:.4f}", '🌳 RF' if m_rf['avg_precision'] > m_lr['avg_precision'] else '📈 LR'],
    ['Precision',     f"{m_lr['precision_1']:.4f}",   f"{m_rf['precision_1']:.4f}",   '🌳 RF' if m_rf['precision_1'] > m_lr['precision_1'] else '📈 LR'],
    ['Recall',        f"{m_lr['recall_1']:.4f}",      f"{m_rf['recall_1']:.4f}",      '🌳 RF' if m_rf['recall_1'] > m_lr['recall_1'] else '📈 LR'],
    ['F1-score',      f"{m_lr['f1_1']:.4f}",          f"{m_rf['f1_1']:.4f}",          '🌳 RF' if m_rf['f1_1'] > m_lr['f1_1'] else '📈 LR'],
    ['Accuracy',      f"{m_lr['accuracy']:.4f}",      f"{m_rf['accuracy']:.4f}",      '🌳 RF' if m_rf['accuracy'] > m_lr['accuracy'] else '📈 LR'],
]
table = ax0.table(cellText=rows,
                  colLabels=['Metric', 'Logistic Regression', 'Random Forest', 'Best'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
for j in range(4):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows)+1):
    for j in range(4):
        table[i, j].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
ax0.set_title('Metrics Summary (churner class = 1)', fontweight='bold', pad=10)

# ── ROC Curve ────────────────────────────────────────────────────────────────
# Shows the TPR/FPR trade-off for every possible threshold.
# The dashed diagonal represents a random classifier (AUC = 0.5).
ax1 = fig.add_subplot(gs[1, 0])
for m in [m_lr, m_rf]:
    ax1.plot(m['fpr'], m['tpr'], label=f"{m['name']} (AUC={m['auc_roc']:.3f})",
             color=colors_m[m['name']], linewidth=2)
ax1.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax1.fill_between(m_rf['fpr'], m_rf['tpr'], alpha=0.07, color=colors_m['Random Forest'])
ax1.set_title('ROC Curve', fontweight='bold')
ax1.set_xlabel('False Positive Rate (FPR)')
ax1.set_ylabel('True Positive Rate (TPR)')
ax1.legend(fontsize=8)

# ── Precision-Recall Curve ───────────────────────────────────────────────────
# More informative than ROC on imbalanced data.
# Baseline = churn rate (11.5%): anything above adds value over random.
ax2 = fig.add_subplot(gs[1, 1])
for m in [m_lr, m_rf]:
    ax2.plot(m['rec_curve'], m['prec_curve'],
             label=f"{m['name']} (AP={m['avg_precision']:.3f})",
             color=colors_m[m['name']], linewidth=2)
ax2.axhline(y_test.mean(), color='k', linestyle='--', linewidth=1,
            alpha=0.5, label=f'Baseline ({y_test.mean():.2f})')
ax2.set_title('Precision-Recall Curve', fontweight='bold')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.legend(fontsize=8)

# ── Side-by-side bar chart ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
mnames = ['AUC-ROC', 'Avg Prec.', 'Precision', 'Recall', 'F1']
lr_v = [m_lr['auc_roc'], m_lr['avg_precision'], m_lr['precision_1'], m_lr['recall_1'], m_lr['f1_1']]
rf_v = [m_rf['auc_roc'], m_rf['avg_precision'], m_rf['precision_1'], m_rf['recall_1'], m_rf['f1_1']]
x = np.arange(len(mnames)); w = 0.35
ax3.bar(x-w/2, lr_v, w, label='LR', color=colors_m['Logistic Regression'], edgecolor='white')
ax3.bar(x+w/2, rf_v, w, label='RF', color=colors_m['Random Forest'], edgecolor='white')
ax3.set_xticks(x); ax3.set_xticklabels(mnames, fontsize=9)
ax3.set_ylim(0, 1.05)
ax3.set_title('Metrics Side-by-Side', fontweight='bold')
ax3.legend()

# ── Confusion matrices ───────────────────────────────────────────────────────
# Reading guide:
#   [0,0] = True Negatives  (TN): non-churners correctly classified
#   [0,1] = False Positives (FP): non-churners predicted as churners
#   [1,0] = False Negatives (FN): missed churners ← worst case
#   [1,1] = True Positives  (TP): churners correctly detected
for idx, (m, pos) in enumerate([(m_lr, gs[2,0]), (m_rf, gs[2,1])]):
    ax = fig.add_subplot(pos)
    sns.heatmap(m['cm'], annot=True, fmt='d',
                cmap='Blues' if idx == 0 else 'Oranges',
                ax=ax, xticklabels=['Predicted 0','Predicted 1'],
                yticklabels=['Actual 0','Actual 1'],
                cbar=False, linewidths=0.5, annot_kws={'size': 12})
    ax.set_title(f'Confusion Matrix\n{m["name"]}', fontweight='bold')

# ── RF Feature importances ───────────────────────────────────────────────────
# Feature importance in RF = average decrease in Gini impurity
# brought by that feature across all trees.
# Useful for feature selection and model interpretability.
ax_fi = fig.add_subplot(gs[2, 2])
fi15 = fi.head(15)
ax_fi.barh(fi15.index[::-1], fi15.values[::-1],
           color=colors_m['Random Forest'], edgecolor='white')
ax_fi.set_title('Top 15 Features (Random Forest)', fontweight='bold')
ax_fi.set_xlabel('Importance')
ax_fi.tick_params(axis='y', labelsize=8)

fig.tight_layout()
fig.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Figure saved → model_comparison.png")
