import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_FILE = Path("outputs/spotify_intent_discovery_sample.csv")
KEYWORD_OUTPUT = Path("outputs/spotify_intent_keywords.csv")
CLUSTER_OUTPUT = Path("outputs/spotify_intent_clusters.csv")

RANDOM_STATE = 42

print("=" * 70)
print("HIVER - SPOTIFY INTENT DISCOVERY ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load sample
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["clean_text"] = (
    df["clean_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["clean_text"].ne("")].copy()

print(f"Messages loaded: {len(df):,}")

# ---------------------------------------------------------
# 2. TF-IDF representation
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.90,
    max_features=5000,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["clean_text"])

print(f"TF-IDF features: {X.shape[1]:,}")

# ---------------------------------------------------------
# 3. Extract important keywords/phrases
# ---------------------------------------------------------

terms = vectorizer.get_feature_names_out()
scores = X.mean(axis=0).A1

keyword_df = pd.DataFrame({
    "term": terms,
    "tfidf_score": scores
})

keyword_df = keyword_df.sort_values(
    "tfidf_score",
    ascending=False
)

keyword_df.to_csv(
    KEYWORD_OUTPUT,
    index=False
)

# ---------------------------------------------------------
# 4. Try several cluster counts
# ---------------------------------------------------------

print()
print("=" * 70)
print("CLUSTER QUALITY")
print("=" * 70)

results = []

for k in range(6, 13):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels,
        sample_size=min(1000, X.shape[0]),
        random_state=RANDOM_STATE
    )

    results.append({
        "k": k,
        "silhouette_score": score
    })

    print(
        f"k={k:2d} | silhouette={score:.4f}"
    )

# ---------------------------------------------------------
# 5. Select best cluster count
# ---------------------------------------------------------

best = max(
    results,
    key=lambda x: x["silhouette_score"]
)

best_k = best["k"]

print()
print(f"Selected cluster count: {best_k}")

# ---------------------------------------------------------
# 6. Final clustering
# ---------------------------------------------------------

model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

df["cluster"] = model.fit_predict(X)

# ---------------------------------------------------------
# 7. Display top terms per cluster
# ---------------------------------------------------------

print()
print("=" * 70)
print("TOP TERMS BY CLUSTER")
print("=" * 70)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

for cluster_id in range(best_k):

    top_terms = [
        terms[index]
        for index in order_centroids[cluster_id, :10]
    ]

    count = int(
        (df["cluster"] == cluster_id).sum()
    )

    print()
    print(
        f"CLUSTER {cluster_id} "
        f"({count} messages)"
    )

    print(
        "  " + ", ".join(top_terms)
    )

# ---------------------------------------------------------
# 8. Save clustered messages
# ---------------------------------------------------------

df = df.sort_values(
    ["cluster", "created_at"],
    kind="stable"
)

df.to_csv(
    CLUSTER_OUTPUT,
    index=False
)

print()
print("=" * 70)
print("INTENT DISCOVERY COMPLETE")
print("=" * 70)

print(f"Saved:")
print(f"  {KEYWORD_OUTPUT}")
print(f"  {CLUSTER_OUTPUT}")