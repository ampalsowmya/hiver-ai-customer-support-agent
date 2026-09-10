import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT = "outputs/spotify_golden_annotation_context.csv"

df = pd.read_csv(INPUT)

# Evaluation ground truth
y_true = df["intent"].astype(str)

# Most common intent in the Golden Set
majority_class = df["intent"].value_counts().idxmax()

# Predict the same class for every example
y_pred = [majority_class] * len(df)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print("=" * 75)
print("HIVER BASELINE — MAJORITY CLASS")
print("=" * 75)

print(f"Evaluation examples : {len(df)}")
print(f"Majority class      : {majority_class}")
print(f"Accuracy            : {accuracy:.4f}")
print(f"Macro F1            : {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

print("=" * 75)