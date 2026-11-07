"""
app.py
------
Streamlit web app for ML Assignment 2 - Credit Card Default Prediction.

What the app does:
    1. Lets the user upload a test CSV (features + 'default' target column).
    2. Lets the user pick one of the 5 saved models from a dropdown.
    3. Loads that already-trained model from the model/ folder
       (NO retraining happens here - the assignment requires this).
    4. Shows the 6 evaluation metrics, a confusion matrix and a
       classification report for the chosen model.
    5. Shows a comparison table of all 5 models on the uploaded data.

Run locally with:
    streamlit run app.py
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# 2. Settings
# ---------------------------------------------------------------------------
# BASE_DIR = the folder this app.py file lives in. Building every path from it
# means the app finds its files no matter which folder you start Streamlit from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_CSV = os.path.join(BASE_DIR, "test_data.csv")
TARGET_COL = "default"

# Dropdown label -> file name inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl",
}

# The 23 input columns the models were trained on, in the same order.
FEATURE_COLS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


# ---------------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource  # load each model from disk only once, then reuse it
def load_model(model_name):
    """Load a saved, already-trained model from the model/ folder."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


def compute_metrics(y_true, y_pred, y_score):
    """Return the 6 assignment metrics as a dictionary."""
    # AUC needs both classes (0 and 1) present in the uploaded data.
    # If a tiny upload has only one class, show "n/a" instead of crashing.
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def run_model(model, X, y):
    """Predict with a saved model and return (predictions, scores, metrics)."""
    y_pred = model.predict(X)                 # class labels 0/1
    y_score = model.predict_proba(X)[:, 1]    # probability of default
    return y_pred, y_score, compute_metrics(y, y_pred, y_score)


def plot_confusion_matrix(y_true, y_pred, title):
    """Draw a labelled confusion matrix heatmap and return the figure."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["No default (0)", "Default (1)"],
        yticklabels=["No default (0)", "Default (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Default Model Lab", page_icon="💳", layout="wide")

st.title("💳 Credit Card Default — Model Lab")
st.caption(
    "Compare five pre-trained classifiers on Taiwan credit-card customers "
    "(UCI dataset, 23 features). Upload test data, pick a model, read the results."
)

with st.expander("ℹ️ About this app", expanded=True):
    st.markdown(
        """
**Purpose.** This app predicts whether a credit-card customer will **default on next
month's payment** and lets you evaluate how well five machine-learning models do it.

**How it works**
1. **Upload** a CSV containing the 23 customer features (credit limit, demographics,
   6 months of repayment status, bill amounts and payment amounts) plus the true
   `default` label (1 = defaulted, 0 = paid).
2. **Choose** one of five pre-trained models: Logistic Regression, Decision Tree,
   KNN, Naive Bayes or SVM. Models were trained offline with `train_models.py`;
   nothing is retrained here.
3. **Review** the results: six metrics (Accuracy, AUC, Precision, Recall, F1, MCC),
   a confusion matrix, a per-class report and a comparison of all five models.
4. **Download** the predictions for every row and the model comparison table as CSV.

**Reading the metrics.** Defaulters are a minority class, so Accuracy alone can be
misleading. **F1** and **MCC** give a fairer picture of how well a model catches
defaulters without raising too many false alarms.
        """
    )

# ----- Sidebar: inputs -----------------------------------------------------
with st.sidebar:
    st.header("1 · Upload test data")
    uploaded = st.file_uploader("CSV with the 23 features + 'default' column", type="csv")

    # Offer the project's own test file so an evaluator can try the app fast
    if os.path.exists(SAMPLE_CSV):
        with open(SAMPLE_CSV, "rb") as f:
            st.download_button("Download sample test_data.csv", f, file_name="test_data.csv")

    st.header("2 · Choose a model")
    chosen = st.selectbox("Model", list(MODEL_FILES.keys()))

    st.divider()
    st.markdown(
        "**Target:** `default` — 1 = customer missed next month's payment, 0 = paid.  \n"
        "Models were trained once offline (`train_models.py`); this app only loads them."
    )

# ----- Stop early if the trained models are not there ----------------------
# The models are created by train_models.py. If they are missing, explain how
# to fix it instead of showing a confusing "No such file" error.
missing_models = [f for f in MODEL_FILES.values()
                  if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing_models:
    st.error(
        f"Trained model files not found in `{MODEL_DIR}`: {missing_models}  \n"
        "Run `python train_models.py` in the project folder first, "
        "then refresh this page. (On Streamlit Cloud: make sure the `model/` "
        "folder is committed and pushed to GitHub.)"
    )
    st.stop()

# ----- Stop early until a file is uploaded ---------------------------------
if uploaded is None:
    st.info("⬅️ Upload a test CSV in the sidebar to begin. "
            "No file? Use **Download sample test_data.csv** first.")
    st.stop()

# ----- Read and check the uploaded file ------------------------------------
# sep=None + engine="python" lets pandas detect the separator itself
# (Excel in some regions saves CSVs with ';' instead of ',').
# encoding="utf-8-sig" removes an invisible "BOM" character Excel may add.
test_df = pd.read_csv(uploaded, sep=None, engine="python", encoding="utf-8-sig")

# Remove stray spaces around column names, e.g. "default " -> "default"
test_df.columns = test_df.columns.str.strip()

# Accept the target column under its original UCI or Kaggle names too
test_df = test_df.rename(columns={
    "default payment next month": TARGET_COL,   # original UCI name
    "default.payment.next.month": TARGET_COL,   # Kaggle CSV name
})

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in test_df.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    # Show what the file actually contains, to make the problem easy to spot
    st.write("Columns found in your file:", list(test_df.columns))
    st.stop()

X_test = test_df[FEATURE_COLS]   # keep only the 23 features, in training order
y_test = test_df[TARGET_COL]

st.success(f"Loaded {len(test_df):,} rows. "
           f"Defaulters in this file: {y_test.mean():.1%}")

# ---------------------------------------------------------------------------
# 5. Results for the chosen model
# ---------------------------------------------------------------------------
model = load_model(chosen)
y_pred, y_score, metrics = run_model(model, X_test, y_test)

st.subheader(f"Results — {chosen}")

# Six metric tiles in one row
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, "n/a" if pd.isna(value) else f"{value:.3f}")

# Confusion matrix on the left, classification report on the right
left, right = st.columns([1, 1.2])
with left:
    st.markdown("**Confusion matrix**")
    cm_fig = plot_confusion_matrix(y_test, y_pred, chosen)
    st.pyplot(cm_fig)
    plt.close(cm_fig)  # free memory; Streamlit reruns the script on every interaction
with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_test, y_pred, target_names=["No default (0)", "Default (1)"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ---------------------------------------------------------------------------
# 6. All five models side by side on the same uploaded data
# ---------------------------------------------------------------------------
st.divider()
st.subheader("All models on this test data")

rows = []
with st.spinner("Scoring all models..."):
    for name in MODEL_FILES:
        # Reuse the chosen model's metrics instead of predicting twice
        m = metrics if name == chosen else run_model(load_model(name), X_test, y_test)[2]
        rows.append({"Model": name, **m})
comparison = pd.DataFrame(rows).set_index("Model").round(4)

# Highlight the best score in each column so the winner stands out
styled = comparison.style.highlight_max(axis=0, color="#cfe8ff").format("{:.4f}")
st.dataframe(styled, width="stretch")
st.download_button(
    "⬇️ Download model comparison (CSV)",
    comparison.to_csv().encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv",
)

# Bar chart of F1 and MCC - the two metrics that matter most with imbalanced classes.
# Drawn with matplotlib (not st.bar_chart) because st.bar_chart relies on the
# altair library, whose version 5.5 crashes on Python 3.14 with
# "TypeError: _TypedDictMeta.__new__() got an unexpected keyword argument 'closed'".
fig, ax = plt.subplots(figsize=(7, 3))
comparison[["F1", "MCC"]].plot.bar(ax=ax, rot=0, color=["#0078D6", "#7B30E5"])
ax.set_ylabel("Score")
ax.set_ylim(0, comparison[["F1", "MCC"]].max().max() * 1.3)  # headroom so the legend never covers a bar
ax.set_xlabel("")
ax.set_title("F1 and MCC by model", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Individual predictions + download
# ---------------------------------------------------------------------------
st.divider()
st.subheader(f"Predictions — {chosen}")

results = test_df.copy()
results["predicted"] = y_pred
results["prob_default"] = y_score.round(4)
results["correct"] = (results["predicted"] == results[TARGET_COL]).astype(int)

st.download_button(
    f"⬇️ Download all {len(results):,} predictions (CSV)",
    results.to_csv(index=False).encode("utf-8"),
    file_name=f"predictions_{MODEL_FILES[chosen].removesuffix('.pkl')}.csv",
    mime="text/csv",
    type="primary",
)

with st.expander("See the first 20 predictions"):
    st.dataframe(results.head(20), width="stretch")

st.divider()
st.caption(
    "ML Assignment 2 · Credit Card Default Prediction · "
    "Data: UCI *Default of Credit Card Clients* (Taiwan, 2005)"
)
