import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

INPUT = "outputs/spotify_golden_annotation_context.csv"

df = pd.read_csv(INPUT)

# Golden Set
X = df["clean_text"].fillna("").astype(str)
y = df["intent"].astype(str)

print("=" * 75)
print("HIVER BASELINE — TF-IDF + LOGISTIC REGRESSION")
print("=" * 75)

print(f"Evaluation examples: {len(df)}")
print(f"Number of intents: {y.nunique()}")

print("\nIntent distribution:")
print(y.value_counts().to_string())

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

# ---------------------------------------------------------
# Stratified cross-validation
# ---------------------------------------------------------
#
# Some classes have very few examples.
# FAMILY_DUO has only 1 example, so normal stratified
# 5-fold CV is impossible.
#
# We therefore evaluate on the classes that have at least
# 2 examples and separately report the excluded class.
# ---------------------------------------------------------

class_counts = y.value_counts()

valid_classes = class_counts[class_counts >= 2].index

mask = y.isin(valid_classes)

X_eval = X[mask]
y_eval = y[mask]

excluded_classes = sorted(
    set(y.unique()) - set(valid_classes)
)

print(f"\nExamples used for CV: {len(X_eval)}")
print(f"Excluded classes: {excluded_classes if excluded_classes else 'NONE'}")

min_class_count = y_eval.value_counts().min()

n_splits = min(5, min_class_count)

print(f"Cross-validation folds: {n_splits}")

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

y_pred = cross_val_predict(
    model,
    X_eval,
    y_eval,
    cv=cv
)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_eval, y_pred)

macro_f1 = f1_score(
    y_eval,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_eval,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 75)
print("RESULTS")
print("=" * 75)

print(f"Accuracy   : {accuracy:.4f}")
print(f"Macro F1   : {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_eval,
        y_pred,
        zero_division=0
    )
)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

labels = sorted(y_eval.unique())

cm = confusion_matrix(
    y_eval,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm_df.to_string())

# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

results = df[mask].copy()
results["predicted_intent"] = y_pred

output = "outputs/spotify_tfidf_baseline_predictions.csv"

results.to_csv(
    output,
    index=False
)

print("\nSaved predictions:")
print(output)

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)