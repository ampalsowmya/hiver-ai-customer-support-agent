from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "outputs" / "spotify_retrieval_v2_predictions.csv"
OUTPUT = ROOT / "outputs" / "retrieval_relevance_annotation.csv"

df = pd.read_csv(INPUT)

# Keep the top 5 retrieved cases for every Golden Set query.
df = df[df["rank"] <= 5].copy()

df["relevance"] = ""
df["annotation_notes"] = ""

# Human-friendly annotation ID.
df.insert(
    0,
    "judgment_id",
    range(1, len(df) + 1)
)

columns = [
    "judgment_id",
    "golden_id",
    "golden_intent",
    "golden_query",
    "rank",
    "similarity",
    "historical_customer",
    "historical_support",
    "relevance",
    "annotation_notes",
]

df[columns].to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 75)
print("RETRIEVAL RELEVANCE ANNOTATION SET")
print("=" * 75)

print(f"Golden queries: {df['golden_id'].nunique()}")
print(f"Retrieved cases: {len(df)}")
print(f"Cases per query: {df.groupby('golden_id').size().min()}–{df.groupby('golden_id').size().max()}")

print()
print("Annotation values:")
print("  RELEVANT")
print("  NOT_RELEVANT")
print("  UNCLEAR")

print()
print(f"Saved: {OUTPUT}")
print("=" * 75)