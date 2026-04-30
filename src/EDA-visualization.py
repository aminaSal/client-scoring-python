# EDA VISUALIZATIONS — Banking Churn
# Objective: visually explore the features in relation to the target variable (churn: yes/no)
# Outputs: eda_numerical.png + eda_categorical.png


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Chargement ───────────────────────────────────────────────────────────────
df = pd.read_csv('bank.csv', sep=';')

# Conversion de la cible en binaire pour faciliter les calculs de taux
df['churn'] = (df['y'] == 'yes').astype(int)
print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ── Paramètres visuels ───────────────────────────────────────────────────────
# Palette cohérente sur tout le notebook : vert = non-churner, rouge = churner
palette  = {'no': '#1D9E75', 'yes': '#D85A30'}
colors   = ['#1D9E75', '#D85A30']

sns.set_theme(style='whitegrid', font_scale=0.9)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F9FA',
    'axes.edgecolor':   '#DEE2E6',
    'grid.color':       '#E9ECEF',
    'font.family':      'DejaVu Sans',
})


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Features numériques vs Churn
# ════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(3, 3, figsize=(15, 12))
fig1.suptitle('EDA — Features numériques vs Churn', fontsize=14, fontweight='bold', y=1.01)

# ── 1.1 Distribution de la cible ────────────────────────────────────────────
# Visualiser le déséquilibre des classes dès le départ est essentiel :
# un fort déséquilibre (ici ~88/12) impacte le choix des métriques et
# des techniques de rééchantillonnage (SMOTE, class_weight, etc.)
ax = axes[0, 0]
counts = df['y'].value_counts()
bars = ax.bar(['Non-churner\n(no)', 'Churner\n(yes)'], counts.values,
              color=colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Distribution de la cible (y)', fontweight='bold')
ax.set_ylabel('Nombre de clients')

# ── 1.2 KDE Âge par classe ──────────────────────────────────────────────────
# La KDE (Kernel Density Estimate) lisse la distribution comme un histogramme
# continu. Superposer les deux classes permet de voir si l'âge discrimine.
ax = axes[0, 1]
for label, grp in df.groupby('y'):
    grp['age'].plot.kde(ax=ax, label=label, color=palette[label], linewidth=2)
ax.set_title('Âge par classe', fontweight='bold')
ax.set_xlabel('Âge')
ax.legend(title='Churn')
ax.set_xlim(15, 90)
# → Les churners ont tendance à être légèrement plus âgés

# ── 1.3 KDE Solde bancaire par classe ───────────────────────────────────────
ax = axes[0, 2]
for label, grp in df.groupby('y'):
    grp['balance'].plot.kde(ax=ax, label=label, color=palette[label], linewidth=2)
ax.set_title('Solde bancaire (balance) par classe', fontweight='bold')
ax.set_xlabel('Balance (€)')
ax.legend(title='Churn')
ax.set_xlim(-5000, 20000)
# → La balance seule discrimine peu, mais combinée à d'autres features
#   elle peut apporter de l'information au modèle

# ── 1.4 Boxplot durée d'appel ───────────────────────────────────────────────
# Le boxplot montre la médiane, les quartiles et les outliers.
# ⚠ Cette feature est un DATA LEAK : elle n'est connue qu'après l'appel.
#   On la visualise ici pour comprendre les données, mais elle sera
#   supprimée lors du feature engineering.
ax = axes[1, 0]
df.boxplot(column='duration', by='y', ax=ax, patch_artist=True,
           boxprops=dict(facecolor='#AED6F1'),
           medianprops=dict(color='#D85A30', linewidth=2))
ax.set_title("Durée d'appel (duration) ⚠ data leak", fontweight='bold')
ax.set_xlabel('Churn')
ax.set_ylabel('Secondes')
plt.sca(ax)
plt.title('')  # retire le titre automatique généré par boxplot

# ── 1.5 Distribution nb de contacts (campagne actuelle) ─────────────────────
# On plafonne à 10 pour la lisibilité : les valeurs > 10 sont rares
# mais tirent l'axe vers la droite et écrasent la lecture principale.
ax = axes[1, 1]
df_c = df.copy()
df_c['campaign_capped'] = df_c['campaign'].clip(upper=10)
for label, grp in df_c.groupby('y'):
    grp['campaign_capped'].value_counts().sort_index().plot(
        kind='bar', ax=ax, alpha=0.7, label=label,
        color=palette[label], width=0.4,
        position=0 if label == 'no' else 1)
ax.set_title('Nb de contacts (campagne actuelle)', fontweight='bold')
ax.set_xlabel('Contacts (plafonné à 10)')
ax.set_ylabel('Nombre de clients')
ax.legend(title='Churn')
# → Plus on contacte un client, moins il souscrit → signe de lassitude

# ── 1.6 Taux de churn selon l'historique de contact ─────────────────────────
# pdays = -1 signifie "jamais contacté avant cette campagne".
# On crée ici une variable binaire pour visualiser son impact sur le churn.
ax = axes[1, 2]
df['contacted_before'] = (df['pdays'] != -1).map(
    {True: 'Déjà contacté', False: 'Jamais contacté'})
churn_by_contact = df.groupby('contacted_before')['churn'].mean().reset_index()
bars = ax.bar(churn_by_contact['contacted_before'],
              churn_by_contact['churn'] * 100,
              color=['#5DADE2', '#F0A500'], edgecolor='white', width=0.4)
for bar, val in zip(bars, churn_by_contact['churn'] * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Taux de churn : contact antérieur ?', fontweight='bold')
ax.set_ylabel('Taux de churn (%)')
ax.set_ylim(0, 25)
# → Les clients déjà contactés lors d'une campagne précédente churnent plus souvent

# ── 1.7 Heatmap de corrélation ──────────────────────────────────────────────
# La heatmap montre les corrélations linéaires entre toutes les features
# numériques. Utile pour détecter les redondances (multicolinéarité)
# et identifier les variables les plus liées à la cible (churn).
ax = axes[2, 0]
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'churn']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
ax.set_title('Corrélation features numériques', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
# → duration est la plus corrélée à churn (mais c'est un data leak)
# → previous a une corrélation positive notable avec churn

# ── 1.8 Scatter âge vs balance ──────────────────────────────────────────────
# Ce scatter plot colore chaque point par classe. S'il n'y a pas de
# séparation claire entre les deux nuages, un modèle linéaire sera limité.
ax = axes[2, 1]
for label, grp in df.groupby('y'):
    ax.scatter(grp['age'], grp['balance'],
               alpha=0.25, s=12, color=palette[label], label=label)
ax.set_title('Âge vs Balance par classe', fontweight='bold')
ax.set_xlabel('Âge')
ax.set_ylabel('Balance (€)')
ax.legend(title='Churn')
# → Pas de séparation linéaire évidente → justifie des modèles non-linéaires
#   comme le Random Forest ou le XGBoost

# ── 1.9 Distribution contacts antérieurs (previous > 0) ─────────────────────
# On filtre les clients jamais contactés (previous = 0) pour ne visualiser
# que ceux qui ont un historique de contact, et voir si le nb de contacts
# antérieurs influe sur le churn.
ax = axes[2, 2]
df_prev = df[df['previous'] > 0]
for label, grp in df_prev.groupby('y'):
    grp['previous'].clip(upper=8).value_counts().sort_index().plot(
        kind='bar', ax=ax, alpha=0.7, label=label, color=palette[label])
ax.set_title('Contacts antérieurs (previous > 0)', fontweight='bold')
ax.set_xlabel('Nb de contacts précédents')
ax.set_ylabel('Nb clients')
ax.legend(title='Churn')

fig1.tight_layout()
fig1.savefig('eda_numeriques.png', dpi=150, bbox_inches='tight')
print("✅ Figure 1 sauvegardée → eda_numeriques.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Features catégorielles vs Churn
# ════════════════════════════════════════════════════════════════════════════
# Pour chaque variable catégorielle, on calcule le taux de churn par modalité
# et on le compare à la moyenne globale (ligne pointillée).
# C'est plus informatif que de simplement compter les effectifs, car ça
# montre directement quelle modalité est "à risque".
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
fig2.suptitle('EDA — Features catégorielles vs Churn', fontsize=14, fontweight='bold', y=1.01)

cat_features = ['job', 'marital', 'education', 'default',
                'housing', 'loan', 'contact', 'month', 'poutcome']

# Commentaires par feature :
# job      → retraités et étudiants ont le taux de churn le plus élevé
# marital  → les célibataires churnent légèrement plus
# education→ niveau primaire : taux plus élevé ; tertiary : plus faible
# default  → peu d'impact (effectifs très faibles pour "yes")
# housing  → prêt immobilier = moins de churn (client plus engagé dans la banque)
# loan     → prêt perso = légère hausse du churn
# contact  → "unknown" (mode de contact non renseigné) = fort taux de churn
# month    → décembre et mars ressortent avec des taux élevés
# poutcome → 'success' est très prédictif : un client converti avant churne peu

for idx, feat in enumerate(cat_features):
    ax = axes2[idx // 3][idx % 3]

    # Taux de churn moyen par modalité, trié décroissant pour plus de lisibilité
    churn_rate = df.groupby(feat)['churn'].mean().sort_values(ascending=False) * 100
    bars = ax.bar(
        churn_rate.index, churn_rate.values,
        color=plt.cm.RdYlGn_r(np.linspace(0.1, 0.7, len(churn_rate))),
        edgecolor='white'
    )

    # Ligne de référence = taux de churn global → permet de voir
    # quelles modalités sont au-dessus ou en dessous de la moyenne
    global_rate = df['churn'].mean() * 100
    ax.axhline(global_rate, color='#333', linestyle='--',
               linewidth=1, alpha=0.6, label=f'Moy. {global_rate:.1f}%')

    # Annotations des barres
    for bar, val in zip(bars, churn_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_title(f'Taux churn par {feat}', fontweight='bold', fontsize=10)
    ax.set_ylabel('Taux de churn (%)')
    ax.set_ylim(0, churn_rate.max() * 1.3)
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.legend(fontsize=7)

fig2.tight_layout()
fig2.savefig('eda_categorielles.png', dpi=150, bbox_inches='tight')
print("✅ Figure 2 sauvegardée → eda_categorielles.png")

