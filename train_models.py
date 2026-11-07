"""
train_models.py
---------------
ML Assignment 2 - Credit Card Default Prediction (UCI dataset)

What this script does, in order:
    1. Loads the raw UCI "Default of Credit Card Clients" Excel file.
    2. Cleans it lightly (drops the ID column, renames the target column).
    3. Splits it into 80% training data and 20% test data.
    4. Saves the 20% test data as test_data.csv (this is the file you upload
       in the Streamlit app).
    5. Trains 5 classifiers on the 80% training data:
           Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM
    6. Calculates 6 metrics for each model on the test data:
           Accuracy, AUC, Precision, Recall, F1, MCC
    7. Saves every trained model as a .pkl file inside the model/ folder,
       and saves the metrics table as model/metrics_summary.csv.

Run it from the project folder with:
    python train_models.py
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os            # to create the model/ folder
import time          # to show how long each model takes to train
import joblib        # to save (and later load) trained models as .pkl files
import pandas as pd  # to work with tables (DataFrames)

# scikit-learn: the machine learning library
from sklearn.model_selection import train_test_split  # splits data into train/test
from sklearn.preprocessing import StandardScaler      # rescales features
from sklearn.pipeline import Pipeline                  # chains scaler + model

# The 5 classifiers required by the assignment
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# The 6 evaluation metrics required by the assignment
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# ---------------------------------------------------------------------------
# 2. Settings (change these in one place if needed)
# ---------------------------------------------------------------------------
# BASE_DIR = the folder this script lives in. All paths are built from it, so
# the script reads and writes the right files whichever folder you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "default of credit card clients.xls")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_COL = "default"   # our short name for the column we want to predict
RANDOM_SEED = 42         # fixes the randomness so results are repeatable
TEST_SHARE = 0.20        # 20% of rows go to the test set


# ---------------------------------------------------------------------------
# 3. Load and clean the data
# ---------------------------------------------------------------------------
def load_dataset(path):
    """Read the UCI Excel file and return a clean DataFrame."""
    # The UCI file has an extra title row on top, so the real column names
    # are on the 2nd row -> header=1 tells pandas to use row index 1.
    df = pd.read_excel(path, header=1)

    # 'ID' is just a row number. It carries no information about the customer,
    # so keeping it could only confuse the models.
    df = df.drop(columns=["ID"])

    # The original target name has spaces, which is awkward to type.
    df = df.rename(columns={"default payment next month": TARGET_COL})
    return df


# ---------------------------------------------------------------------------
# 4. Define the 5 models
# ---------------------------------------------------------------------------
def build_models():
    """
    Return a dictionary: {model name: untrained model}.

    Why a Pipeline with StandardScaler?
    - Features here have very different ranges: LIMIT_BAL goes up to 1,000,000
      while SEX is 1 or 2. Models that use distances (KNN, SVM) or gradient-based
      weights (Logistic Regression) treat big numbers as more important unless
      we rescale every column to mean 0 and standard deviation 1.
    - Putting the scaler INSIDE the pipeline means the scaler is saved together
      with the model in one .pkl file. The Streamlit app can then feed raw test
      data straight in, and the same scaling learned from training is applied.
    - Decision Trees split on thresholds, so scaling does not change them;
      we leave the tree without a scaler.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            # max_iter raised from 100 so the solver has time to converge
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,            # limit depth so the tree does not memorise noise
            min_samples_leaf=20,    # each leaf must hold at least 20 customers
            random_state=RANDOM_SEED,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            # look at the 15 most similar customers and take a majority vote
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            # Gaussian NB suits continuous numeric features like ours
            ("clf", GaussianNB()),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            # RBF kernel lets the SVM draw curved boundaries.
            # probability=True lets us get probabilities for the AUC score
            # (it makes training slower - expect a few minutes for this model).
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
        ]),
    }
    return models


# ---------------------------------------------------------------------------
# 5. Calculate the 6 metrics for one model
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Return a dict with the 6 required metrics for a trained model."""
    # predict() gives the class label (0 or 1) for each row
    y_pred = model.predict(X_test)

    # predict_proba() gives the probability of each class; column 1 is the
    # probability of "default = 1". AUC needs these scores, not hard labels.
    y_score = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        # zero_division=0 avoids a crash if a model never predicts class 1
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


# ---------------------------------------------------------------------------
# 6. Main program
# ---------------------------------------------------------------------------
def main():
    # --- Load data ---------------------------------------------------------
    data = load_dataset(RAW_DATA_PATH)
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    # X = all feature columns, y = the target column
    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]
    print(f"Number of features: {X.shape[1]}")
    print("Class balance (share of each class):")
    print(y.value_counts(normalize=True).round(3).to_string(), "\n")

    # --- Split into train and test ----------------------------------------
    # stratify=y keeps the same share of defaulters (~22%) in both parts,
    # so the test set is a fair sample of the whole dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SHARE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    # --- Save the test set as CSV (features + target together) -------------
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(TEST_CSV_PATH, index=False)
    print(f"Saved test data to {TEST_CSV_PATH}\n")

    # --- Train, evaluate and save each model -------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)   # create model/ if it is missing
    results = []                            # one row of metrics per model

    for name, model in build_models().items():
        start = time.time()
        model.fit(X_train, y_train)          # learning happens here
        seconds = time.time() - start

        scores = evaluate_model(model, X_test, y_test)
        scores = {"Model": name, **scores}   # put the model name first
        results.append(scores)

        # File name: "Logistic Regression" -> "logistic_regression.pkl"
        file_name = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, os.path.join(MODEL_DIR, file_name))
        print(f"{name:<20} trained in {seconds:6.1f}s -> saved {file_name}")

    # --- Show and save the comparison table --------------------------------
    summary = pd.DataFrame(results).set_index("Model").round(4)
    summary.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"))

    print("\n=== Test-set metrics (copy this table into README.md) ===")
    print(summary.to_markdown())   # markdown format pastes straight into README


# This line means: only run main() when the file is run directly
# (python train_models.py), not when it is imported by another file.
if __name__ == "__main__":
    main()
