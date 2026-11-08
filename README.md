# Credit Card Default Prediction — ML Assignment 2

**M.Tech (AIML) · Machine Learning · BITS Pilani WILP**

**Name:** Sunny Pawar | **BITS ID:** 2026K05098

**GitHub repository:** https://github.com/sunnypawar-92/Machine-Learning-Assignment

**Live Streamlit app:** https://credit-default-sunnypawar.streamlit.app/

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

https://github.com/sunnypawar-92/Machine-Learning-Assignment


**Repository structure**

```
Machine-Learning-Assignment/
├── .streamlit/
│   └── config.toml               # custom app theme (colours, font)
├── data/
│   └── default of credit card clients.xls   # raw UCI dataset (30,000 rows, 23 features)
├── model/                        # saved trained models - loaded by the app, never retrained
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── metrics_summary.csv       # test-set metrics for all 5 models
├── .gitignore                    # keeps venv/ and cache files out of the repo
├── README.md                     # this file
├── app.py                        # Streamlit app: upload CSV, pick model, view metrics
├── requirements.txt              # pinned library versions for deployment
├── test_data.csv                 # 20% stratified hold-out set (6,000 rows) - upload this in the app
└── train_models.py               # trains the 5 models, prints metrics, saves .pkl files
```

*Preprocessing:* the `StandardScaler` is saved inside each model's pipeline (all except the Decision Tree), so no separate preprocessing file is needed.

## d. Models used

All five classifiers were trained on the same 24,000 training rows and evaluated on the same 6,000 test rows (80/20 stratified split, `random_state=42`).

| Model | Preprocessing | Hyper-parameters used | Why these values |
|---|---|---|---|
| Logistic Regression | StandardScaler | `max_iter=1000`, `C=1.0` (default), `penalty='l2'` (default), `solver='lbfgs'` (default) | `max_iter` raised from 100 so the solver converges on scaled data |
| Decision Tree | None (trees do not need scaling) | `max_depth=6`, `min_samples_leaf=20`, `criterion='gini'` (default) | Depth and leaf-size limits stop the tree from memorising noise (overfitting) |
| K-Nearest Neighbours | StandardScaler | `n_neighbors=15`, `weights='uniform'` (default), Euclidean distance (default) | k = 15 smooths out noisy neighbours compared with the default k = 5 |
| Gaussian Naive Bayes | StandardScaler | `var_smoothing=1e-9` (default) | No key hyper-parameters; defaults used |
| Support Vector Machine | StandardScaler | `kernel='rbf'`, `C=1.0` (default), `gamma='scale'` (default), `probability=True` | RBF kernel allows non-linear boundaries; `probability=True` is needed to calculate AUC |

*Hyper-parameters were chosen manually from the reasoning above, not by an automated search such as GridSearchCV. `random_state=42` is fixed for every model that uses randomness, so results are reproducible.*

### Model Comparison table (test set, 6,000 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.8165 | 0.7479 | 0.6677 | 0.3391 | 0.4498 | 0.3827 |
| KNN | 0.8065 | 0.7328 | 0.6153 | 0.3338 | 0.4328 | 0.3507 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Support Vector Machine | 0.8160 | 0.7210 | 0.6652 | 0.3384 | 0.4486 | 0.3809 |

*Baseline for context: always predicting "no default" would score 0.779 accuracy but
0 recall, so accuracy alone is misleading on this imbalanced dataset; F1 and MCC are more informative.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.6868) but lowest recall (0.2396): when it flags a defaulter it is usually right, but it misses about 3 in 4 actual defaulters. Its straight-line decision boundary cannot capture the non-linear effect of payment delays, so it has the lowest F1 (0.3553) and MCC (0.3244). |
| Decision Tree | Best accuracy (0.8165), AUC (0.7479) and MCC (0.3827). Its rules lean heavily on recent repayment status (`PAY_0` alone carries about 66% of feature importance), which matches how default really happens. Limiting depth to 6 kept it from overfitting. |
| KNN | Middle of the pack (accuracy 0.8065, MCC 0.3507). It needs scaled features to work and keeps all 24,000 training rows in memory, so its model file is the largest after SVM and predictions are slower. Recall stays low because defaulters are a minority among the 15 nearest neighbours. |
| Naive Bayes | Lowest accuracy (0.7525) but highest recall (0.5539) and highest F1 (0.4975): it catches the most defaulters at the cost of more false alarms. Its independence assumption is violated here (the six `BILL_AMT` columns correlate 0.80–0.95), which skews its probabilities. |
| Support Vector Machine | Nearly ties the Decision Tree on accuracy (0.8165) and MCC (0.3809), but has a lower AUC (0.7210) and took by far the longest to train (~80 s vs under 1 s for the others). The extra cost buys no gain on this dataset. |
| **Overall, winner for this dataset?** | **Decision Tree** — top accuracy, AUC and MCC, it trains fast, and its rules are easy to explain to a credit team. If the business priority is catching as many defaulters as possible (recall), Naive Bayes is the better pick despite lower accuracy. |

## e. Streamlit App Features
**1. Dataset upload (CSV) -** Users can upload a test dataset containing the 23 input features and the target column (default). The app validates the uploaded file and checks for missing columns.

**2. Model selection dropdown -** Users can select one of the five pre-trained machine learning models:

```
Logistic Regression
Decision Tree
K-Nearest Neighbours (KNN)
Naive Bayes
Support Vector Machine (SVM)
```

**3. Evaluation metrics display -** The selected model is evaluated using six performance metrics displayed as metric cards:

```
Accuracy
AUC
Precision
Recall
F1-Score
Matthews Correlation Coefficient (MCC)
```

**4. Confusion matrix and classification report -** The app displays a confusion matrix heatmap and a detailed classification report showing Precision, Recall, F1-Score, Support, Accuracy, Macro Average and Weighted Average values.

### Additional Features

**5. Sample test dataset download -** A sample test_data.csv file can be downloaded for quick testing of the application.

**6. Dataset summary -** Displays the number of uploaded records and the percentage of defaulting customers in the test dataset.

**7. Model comparison table -** All five models are evaluated on the uploaded dataset and compared side-by-side using the six evaluation metrics.

**8. Best performance highlighting -** The highest value in each metric column is highlighted, making it easier to identify the best-performing model for each evaluation criterion.

**9. F1 and MCC comparison chart -** A bar chart visually compares F1-Score and MCC across all five models.

**10. Prediction results table -** Displays the predicted class, probability of default, and prediction correctness for each record.

**11. Results download -** Users can download:

    Model comparison results as a CSV file.
    Individual prediction results as a CSV file.

**12. Interactive explanations -** Expandable sections provide guidance on interpreting performance metrics, confusion matrices, classification reports, and model comparison results.

**Note:** The application only loads pre-trained models stored in the model/ folder and performs prediction on uploaded test data. No model retraining is performed within the Streamlit application.

## f. How to run locally

```
pip install -r requirements.txt
python train_models.py          # retrains and re-saves the models (optional)
streamlit run app.py
```
