# Credit Card Default Prediction — ML Assignment 2

> **Before you submit:** replace every `<...>` placeholder, check every number against
> the table printed by `python train_models.py`, and rewrite the observations in your own words.

## a. Problem statement

Banks lose money when credit-card customers stop paying. The goal of this project is to
predict, from a customer's profile and last six months of billing and payment history,
whether that customer will **default on next month's payment** (1 = default, 0 = no default).
This is a **binary classification** problem. Five classifiers are trained on the same data,
compared on six metrics, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Default of Credit Card Clients (Yeh & Lien, 2009) |
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| Instances | 30,000 customers (Taiwan, April–September 2005) |
| Features | 23 input features (after dropping the `ID` column) |
| Target | `default` (renamed from `default payment next month`): 1 = default, 0 = no default |
| Class balance | 77.9% no default, 22.1% default (imbalanced) |
| Train / test split | 80% / 20% stratified (24,000 / 6,000 rows), `random_state=42` |

**Feature groups**

| Features | Meaning |
|---|---|
| `LIMIT_BAL` | Credit limit (NT dollars) |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographics (coded as numbers) |
| `PAY_0`, `PAY_2` … `PAY_6` | Repayment status for Sep → Apr (-1 = paid on time, 1–9 = months of delay) |
| `BILL_AMT1` … `BILL_AMT6` | Bill statement amount for Sep → Apr |
| `PAY_AMT1` … `PAY_AMT6` | Amount actually paid for Sep → Apr |

Preprocessing: `ID` column dropped; features standardised (`StandardScaler`) inside a
pipeline for Logistic Regression, KNN, Naive Bayes and SVM; the Decision Tree uses raw values.

## c. GitHub repository link

<paste your repo link, e.g. https://github.com/your-username/credit-default-ml>

Live Streamlit app: <paste your app link, e.g. https://your-app-name.streamlit.app>

**Repository structure**

```
├── app.py                  # Streamlit app (loads saved models, no retraining)
├── train_models.py         # trains the 5 models, prints metrics, saves .pkl files
├── requirements.txt        # pinned library versions
├── README.md
├── test_data.csv           # 20% hold-out test set (upload this in the app)
├── data/                   # raw UCI .xls file
├── model/                  # 5 trained models (.pkl) + metrics_summary.csv
└── .streamlit/config.toml  # custom app theme
```

## d. Models used

Five classifiers, all trained on the same 24,000 training rows and evaluated on the same
6,000 test rows: Logistic Regression, Decision Tree (max_depth=6, min_samples_leaf=20),
K-Nearest Neighbours (k=15), Gaussian Naive Bayes, and Support Vector Machine (RBF kernel).

### Comparison table (test set, 6,000 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.8165 | 0.7479 | 0.6677 | 0.3391 | 0.4498 | 0.3827 |
| KNN | 0.8065 | 0.7327 | 0.6153 | 0.3338 | 0.4328 | 0.3507 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Support Vector Machine | 0.8160 | 0.7210 | 0.6652 | 0.3384 | 0.4486 | 0.3809 |

*Baseline for context: always predicting "no default" would score 0.779 accuracy but
0 recall, so accuracy alone is misleading on this imbalanced dataset; F1 and MCC are more informative.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.687) but lowest recall (0.240): when it flags a defaulter it is usually right, but it misses about 3 in 4 actual defaulters. Its straight-line decision boundary cannot capture the non-linear effect of payment delays, so it has the lowest F1 (0.355) and MCC (0.324). |
| Decision Tree | Best accuracy (0.8165), AUC (0.748) and MCC (0.383). Its rules lean heavily on recent repayment status (`PAY_0` alone carries about 66% of feature importance), which matches how default really happens. Limiting depth to 6 kept it from overfitting. |
| KNN | Middle of the pack (accuracy 0.807, MCC 0.351). It needs scaled features to work and keeps all 24,000 training rows in memory, so its model file is the largest after SVM and predictions are slower. Recall stays low because defaulters are a minority among the 15 nearest neighbours. |
| Naive Bayes | Lowest accuracy (0.753) but highest recall (0.554) and highest F1 (0.498): it catches the most defaulters at the cost of more false alarms. Its independence assumption is violated here (the six `BILL_AMT` columns correlate 0.80–0.95), which skews its probabilities. |
| Support Vector Machine | Nearly ties the Decision Tree on accuracy (0.816) and MCC (0.381), but has a lower AUC (0.721) and took by far the longest to train (~80 s vs under 1 s for the others). The extra cost buys no gain on this dataset. |
| **Overall, winner for this dataset?** | **Decision Tree** — top accuracy, AUC and MCC, it trains fast, and its rules are easy to explain to a credit team. If the business priority is catching as many defaulters as possible (recall), Naive Bayes is the better pick despite lower accuracy. |

## How to run locally

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python train_models.py           # retrains and re-saves the models (optional)
streamlit run app.py
```
