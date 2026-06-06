# Bank Churn Scoring

Predicting customer churn for a bank using marketing campaign data. The project covers data exploration, feature engineering, and a comparison of two classification models.

---

## 📁 Project Structure

```
bank-churn-scoring/
│
├── data/
│   ├── bank.csv                  # Raw dataset (semicolon-separated)
│   └── bank_engineered.csv       # Dataset after feature engineering
│
├── src/
│   ├── eda_visualisations.py      # Exploration & visualisations
│   └── feature_engineering.py     # Feature transformation
│   └── modelling.py               # Model training & comparison
│   └── client_scoring.py          # Per-client churn score generation
│
├── outputs/
│   ├── eda_numeric.png
│   ├── eda_categorical.png
│   ├── model_comparison.png
│   └── client_scores.csv
│
└── README.md
```

---

## Dataset

- **Source**: [Bank Marketing Dataset — UCI / Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- **4,521 clients**, 16 features + 1 target (`y`: subscription → churn proxy)
- **Class imbalance**: 88.5% non-churners / 11.5% churners

---

## Feature Engineering

| Transformation | Variable(s) | Reason |
|---|---|---|
| Drop | `duration` | Data leak — unknown before the call |
| Binary 0/1 | `default`, `housing`, `loan`, `y` | Only 2 modalities |
| Ordinal | `education` | Natural order: primary < secondary < tertiary |
| Cyclical sin/cos | `month` | Continuity between December and January |
| One-Hot | `job`, `marital`, `contact`, `poutcome` | No order between modalities |
| New feature | `was_contacted` | Extracts meaning from `pdays = -1` |

---

## Models

Both models trained with `class_weight='balanced'` to handle class imbalance.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| AUC-ROC | 0.7152 | 0.7242 |
| Avg Precision | 0.3133 | 0.3163 |
| Recall | **0.577** | 0.135 |
| Precision | 0.189 | **0.636** |
| F1-score | **0.285** | 0.222 |

**Selected model: Logistic Regression** — 4× better recall, interpretable coefficients, and well-suited for a context where missing a churner is more costly than a false alarm.

---

## Client Scoring

Each client receives a churn probability score (average of LR + RF), segmented into 3 risk levels:

| Segment | Threshold | Recommended Action |
|---|---|---|
| 🔴 High Risk | ≥ 0.60 | Priority commercial call |
| 🟠 Moderate Risk | 0.35 – 0.60 | Targeted follow-up & offer |
| 🟢 Low Risk | < 0.35 | No immediate action |

---

## Getting Started

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Explore the data
python eda_visualisations.py

# 3. Prepare features
python feature_engineering.py

# 4. Train and compare models
python modelling.py

# 5. Generate client scores
python client_scoring.py


