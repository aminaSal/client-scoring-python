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
# ÉTAPE 1 — Chargement & split
# ════════════════════════════════════════════════════════════════════════════

# On repart du dataset déjà transformé par le script feature_engineering.py
df = pd.read_csv('bank_engineered.csv')

X = df.drop(columns=['churn'])
y = df['churn']

# Split stratifié : stratify=y garantit que les proportions de churners (11.5%)
# sont les mêmes dans le train ET le test.
# Sans stratification, on risque un test set sans (ou avec trop peu de) churners.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train : {X_train.shape} | Test : {X_test.shape}")
print(f"   Taux churn train : {y_train.mean()*100:.1f}% | test : {y_test.mean()*100:.1f}%\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Normalisation (pour la Logistic Regression uniquement)
# ════════════════════════════════════════════════════════════════════════════

# La Logistic Regression est sensible aux échelles : une feature avec des
# valeurs 0-71000 (balance) écrase une feature 0-1 (binary).
# Le StandardScaler centre (moyenne=0) et réduit (écart-type=1) chaque feature.
# ⚠ IMPORTANT : fit_transform sur le TRAIN, transform seul sur le TEST.
#   Autrement, on "contaminerait" le test avec des infos du train (data leak).
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Le Random Forest est insensible aux échelles → pas besoin de normaliser.


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement des modèles
# ════════════════════════════════════════════════════════════════════════════

# ── Logistic Regression ──────────────────────────────────────────────────────
# class_weight='balanced' : compense automatiquement le déséquilibre des classes
#   en donnant plus de poids aux churners (minorité) lors de l'optimisation.
# Sans ce paramètre, le modèle ignorerait les churners et prédirait toujours 0
#   pour atteindre une accuracy de 88.5% en ne se trompant jamais !
# max_iter=1000 : augmenté depuis la valeur par défaut (100) pour garantir
#   la convergence avec des données normalisées.
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_sc, y_train)
print("✅ Logistic Regression entraînée")

# ── Random Forest ────────────────────────────────────────────────────────────
# n_estimators=200 : 200 arbres de décision → plus stable que la valeur
#   par défaut (100), sans coût computationnel excessif.
# class_weight='balanced' : même logique que pour la LR.
# n_jobs=-1 : utilise tous les cœurs CPU disponibles → entraînement parallèle.
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("✅ Random Forest entraîné\n")


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Prédictions
# ════════════════════════════════════════════════════════════════════════════

# predict() → classe binaire (0 ou 1), utile pour les métriques classiques
# predict_proba()[:, 1] → probabilité d'être churner (colonne 1 = classe positive)
#   plus riche : permet de tracer les courbes ROC/PR et d'ajuster le seuil de décision
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Calcul des métriques
# ════════════════════════════════════════════════════════════════════════════

def get_metrics(y_true, y_pred, y_proba, name):
    """
    Calcule toutes les métriques utiles pour un modèle de classification binaire.

    Pourquoi autant de métriques ?
    ─────────────────────────────
    Sur des données déséquilibrées, l'accuracy seule est trompeuse.
    Un modèle qui prédit toujours "non-churner" atteint 88.5% d'accuracy !

    - AUC-ROC      : capacité globale à séparer les classes, indépendamment
                     du seuil de décision. 0.5 = aléatoire, 1.0 = parfait.
    - Avg Precision: résume la courbe Précision-Rappel → plus robuste que
                     l'AUC-ROC sur datasets très déséquilibrés.
    - Precision    : parmi les clients prédits churners, combien le sont vraiment ?
                     (éviter les fausses alarmes = coût d'actions marketing inutiles)
    - Recall       : parmi les vrais churners, combien sont détectés ?
                     (ne pas rater de churners = coût de perte client)
    - F1-score     : moyenne harmonique Precision/Recall → équilibre les deux.
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

# Affichage console
for m in [m_lr, m_rf]:
    print(f"{'='*42}\n{m['name']}")
    print(f"  AUC-ROC       : {m['auc_roc']:.4f}")
    print(f"  Avg Precision : {m['avg_precision']:.4f}")
    print(f"  Precision     : {m['precision_1']:.4f}")
    print(f"  Recall        : {m['recall_1']:.4f}")
    print(f"  F1-score      : {m['f1_1']:.4f}")
    print(f"  Accuracy      : {m['accuracy']:.4f}")

# Top features du Random Forest
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\n🌳 Top 10 features (Random Forest) :")
print(fi.head(10).to_string())


# ════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Visualisation de la comparaison
# ════════════════════════════════════════════════════════════════════════════

colors_m = {'Logistic Regression': '#5DADE2', 'Random Forest': '#F39C12'}

sns.set_theme(style='whitegrid', font_scale=0.9)
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
                     'axes.edgecolor': '#DEE2E6', 'grid.color': '#E9ECEF'})

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Comparaison — Logistic Regression vs Random Forest',
             fontsize=14, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Tableau de synthèse ──────────────────────────────────────────────────────
# Un tableau récapitulatif en haut de figure offre une lecture immédiate
# des performances relatives, sans avoir à lire chaque graphique séparément.
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
                  colLabels=['Métrique', 'Logistic Regression', 'Random Forest', 'Meilleur'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
for j in range(4):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows)+1):
    for j in range(4):
        table[i, j].set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
ax0.set_title('Récapitulatif des métriques (classe churner = 1)', fontweight='bold', pad=10)

# ── Courbe ROC ──────────────────────────────────────────────────────────────
# La courbe ROC montre le compromis TPR/FPR pour tous les seuils possibles.
# L'AUC (aire sous la courbe) résume la performance en un seul chiffre.
# La diagonale pointillée représente un classifieur aléatoire (AUC = 0.5).
ax1 = fig.add_subplot(gs[1, 0])
for m in [m_lr, m_rf]:
    ax1.plot(m['fpr'], m['tpr'], label=f"{m['name']} (AUC={m['auc_roc']:.3f})",
             color=colors_m[m['name']], linewidth=2)
ax1.plot([0,1],[0,1], 'k--', linewidth=1, alpha=0.5, label='Aléatoire')
ax1.fill_between(m_rf['fpr'], m_rf['tpr'], alpha=0.07, color=colors_m['Random Forest'])
ax1.set_title('Courbe ROC', fontweight='bold')
ax1.set_xlabel('Taux de faux positifs (FPR)')
ax1.set_ylabel('Taux de vrais positifs (TPR)')
ax1.legend(fontsize=8)

# ── Courbe Précision-Rappel ──────────────────────────────────────────────────
# Plus informative que la ROC sur données déséquilibrées.
# La baseline = taux de churn (11.5%) : c'est ce qu'atteindrait un classifieur
# aléatoire. Tout ce qui est au-dessus apporte de la valeur.
ax2 = fig.add_subplot(gs[1, 1])
for m in [m_lr, m_rf]:
    ax2.plot(m['rec_curve'], m['prec_curve'],
             label=f"{m['name']} (AP={m['avg_precision']:.3f})",
             color=colors_m[m['name']], linewidth=2)
ax2.axhline(y_test.mean(), color='k', linestyle='--', linewidth=1,
            alpha=0.5, label=f'Baseline ({y_test.mean():.2f})')
ax2.set_title('Courbe Précision-Rappel', fontweight='bold')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.legend(fontsize=8)

# ── Comparaison barres ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
mnames = ['AUC-ROC', 'Avg Prec.', 'Precision', 'Recall', 'F1']
lr_v = [m_lr['auc_roc'], m_lr['avg_precision'], m_lr['precision_1'], m_lr['recall_1'], m_lr['f1_1']]
rf_v = [m_rf['auc_roc'], m_rf['avg_precision'], m_rf['precision_1'], m_rf['recall_1'], m_rf['f1_1']]
x = np.arange(len(mnames)); w = 0.35
ax3.bar(x-w/2, lr_v, w, label='LR', color=colors_m['Logistic Regression'], edgecolor='white')
ax3.bar(x+w/2, rf_v, w, label='RF', color=colors_m['Random Forest'], edgecolor='white')
ax3.set_xticks(x); ax3.set_xticklabels(mnames, fontsize=9)
ax3.set_ylim(0, 1.05)
ax3.set_title('Métriques côte à côte', fontweight='bold')
ax3.legend()

# ── Matrices de confusion ────────────────────────────────────────────────────
# Lecture :
#   [0,0] = Vrais Négatifs  (TN) : non-churners bien classés
#   [0,1] = Faux Positifs   (FP) : non-churners prédits churners (fausses alarmes)
#   [1,0] = Faux Négatifs   (FN) : churners non détectés (le pire cas !)
#   [1,1] = Vrais Positifs  (TP) : churners bien détectés
for idx, (m, pos) in enumerate([(m_lr, gs[2,0]), (m_rf, gs[2,1])]):
    ax = fig.add_subplot(pos)
    sns.heatmap(m['cm'], annot=True, fmt='d',
                cmap='Blues' if idx == 0 else 'Oranges',
                ax=ax, xticklabels=['Prédit 0','Prédit 1'],
                yticklabels=['Réel 0','Réel 1'],
                cbar=False, linewidths=0.5, annot_kws={'size': 12})
    ax.set_title(f'Matrice confusion\n{m["name"]}', fontweight='bold')

# ── Feature importances RF ───────────────────────────────────────────────────
# L'importance d'une feature dans un RF = diminution moyenne de l'impureté
# (Gini) apportée par cette feature à travers tous les arbres.
# Utile pour la sélection de features et l'interprétabilité du modèle.
ax_fi = fig.add_subplot(gs[2, 2])
fi15 = fi.head(15)
ax_fi.barh(fi15.index[::-1], fi15.values[::-1],
           color=colors_m['Random Forest'], edgecolor='white')
ax_fi.set_title('Top 15 features (Random Forest)', fontweight='bold')
ax_fi.set_xlabel('Importance')
ax_fi.tick_params(axis='y', labelsize=8)

fig.tight_layout()
fig.savefig('comparaison_modeles.png', dpi=150, bbox_inches='tight')
print("\n✅ Figure sauvegardée → comparaison_modeles.png")
