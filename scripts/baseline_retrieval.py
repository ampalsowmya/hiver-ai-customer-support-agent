import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

GOLDEN_SET = "outputs/spotify_golden_annotation_context.csv"
CORPUS = "data/spotifycares_clean.csv"
OUTPUT = "outputs/spotify_retrieval_baseline_predictions.csv"

TOP_K = 5


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 75)
print("HIVER BASELINE — TF-IDF HISTORICAL CASE RETRIEVAL")
print("=" * 75)

golden = pd.read_csv(GOLDEN_SET)
corpus = pd.read_csv(CORPUS)

print(f"Golden Set rows       : {len(golden)}")
print(f"Historical corpus rows: {len(corpus)}")


# =========================================================
# PREPARE CUSTOMER-ONLY RETRIEVAL CORPUS
# =========================================================

# Historical customer messages are the retrieval candidates.
corpus = corpus[
    corpus["inbound"] == True
].copy()

corpus["clean_text"] = (
    corpus["clean_text"]
    .fillna("")
    .astype(str)
)

golden["clean_text"] = (
    golden["clean_text"]
    .fillna("")
    .astype(str)
)

print(f"Historical customer messages: {len(corpus)}")


# =========================================================
# REMOVE GOLDEN SET FROM RETRIEVAL CORPUS
# =========================================================

golden_ids = set(
    golden["tweet_id"]
    .dropna()
    .astype(str)
)

corpus["tweet_id_str"] = (
    corpus["tweet_id"]
    .astype(str)
)

before = len(corpus)

corpus = corpus[
    ~corpus["tweet_id_str"].isin(golden_ids)
].copy()

after = len(corpus)

print(f"Removed Golden Set cases: {before - after}")
print(f"Final retrieval corpus : {after}")


# =========================================================
# REMOVE EXACT DUPLICATE TEXT
# =========================================================

corpus = corpus[
    corpus["clean_text"].str.strip() != ""
].drop_duplicates(
    subset=["clean_text"]
).reset_index(drop=True)

print(f"Unique retrieval cases : {len(corpus)}")


# =========================================================
# TF-IDF
# =========================================================

print("\nBuilding TF-IDF index...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=50000
)

corpus_matrix = vectorizer.fit_transform(
    corpus["clean_text"]
)

golden_matrix = vectorizer.transform(
    golden["clean_text"]
)

print(f"TF-IDF matrix shape: {corpus_matrix.shape}")


# =========================================================
# RETRIEVAL
# =========================================================

print("\nRunning retrieval...")

similarities = cosine_similarity(
    golden_matrix,
    corpus_matrix
)


# =========================================================
# SAVE TOP-K RESULTS
# =========================================================

rows = []

for i in range(len(golden)):

    scores = similarities[i]

    # Top K indices
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    for rank, idx in enumerate(top_indices, start=1):

        rows.append({
            "annotation_id": golden.iloc[i]["annotation_id"],
            "query_tweet_id": golden.iloc[i]["tweet_id"],
            "query_text": golden.iloc[i]["clean_text"],
            "query_intent": golden.iloc[i]["intent"],
            "rank": rank,
            "retrieved_tweet_id": corpus.iloc[idx]["tweet_id"],
            "retrieved_text": corpus.iloc[idx]["clean_text"],
            "similarity_score": float(scores[idx]),
        })


results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT,
    index=False
)


# =========================================================
# RETRIEVAL METRICS
# =========================================================

print("\n" + "=" * 75)
print("RETRIEVAL EVALUATION")
print("=" * 75)

# For each Golden Set example, check whether at least one
# retrieved historical case has the same human-labeled intent.

recall_at_1 = 0
recall_at_3 = 0
recall_at_5 = 0

mrr_scores = []

for annotation_id, group in results.groupby(
    "annotation_id",
    sort=False
):

    true_intent = group["query_intent"].iloc[0]

    relevant = (
        group["retrieved_tweet_id"]
        .map(
            corpus.set_index("tweet_id")["tweet_id"]
            if False else
            lambda x: False
        )
    )

    # Retrieve intents directly from corpus
    retrieved_ids = group["retrieved_tweet_id"].tolist()

    retrieved_intents = []

    for tweet_id in retrieved_ids:
        match = corpus[
            corpus["tweet_id"].astype(str)
            == str(tweet_id)
        ]

        if len(match) > 0:
            # The historical corpus itself does not have
            # human intent labels, so we cannot use it as
            # ground truth.
            retrieved_intents.append(None)

    # MRR cannot honestly be computed from intent labels
    # because historical corpus cases are unlabeled.
    mrr_scores.append(0.0)


# =========================================================
# IMPORTANT: UNSUPERVISED RETRIEVAL METRIC
# =========================================================

# Since historical cases do not have human intent labels,
# we cannot honestly claim Recall@K/MRR for semantic
# relevance without additional human relevance judgments.
#
# Instead, calculate retrieval similarity statistics.

top1_scores = (
    results[results["rank"] == 1]["similarity_score"]
)

top5_scores = (
    results[results["rank"] <= 5]
    .groupby("annotation_id")["similarity_score"]
    .max()
)

print(f"Golden Set queries: {len(golden)}")

print(
    f"Mean Top-1 similarity: "
    f"{top1_scores.mean():.4f}"
)

print(
    f"Median Top-1 similarity: "
    f"{top1_scores.median():.4f}"
)

print(
    f"Mean Top-5 best similarity: "
    f"{top5_scores.mean():.4f}"
)

print(
    f"Median Top-5 best similarity: "
    f"{top5_scores.median():.4f}"
)


# =========================================================
# HIGH-SIMILARITY COVERAGE
# =========================================================

for threshold in [0.20, 0.30, 0.40, 0.50]:

    coverage = (
        top5_scores >= threshold
    ).mean()

    print(
        f"Top-5 similarity >= {threshold:.2f}: "
        f"{coverage:.2%}"
    )


# =========================================================
# SAMPLE RETRIEVALS
# =========================================================

print("\n" + "=" * 75)
print("SAMPLE RETRIEVALS")
print("=" * 75)

sample_ids = golden["annotation_id"].head(5).tolist()

for annotation_id in sample_ids:

    query = golden[
        golden["annotation_id"] == annotation_id
    ].iloc[0]

    retrieved = results[
        results["annotation_id"] == annotation_id
    ].sort_values("rank")

    print("\n" + "-" * 75)

    print(
        f"Golden ID: {annotation_id}"
    )

    print(
        f"Intent: {query['intent']}"
    )

    print(
        f"Customer: {query['clean_text']}"
    )

    for _, row in retrieved.iterrows():

        print(
            f"\n  Rank {int(row['rank'])}"
            f" | similarity={row['similarity_score']:.4f}"
        )

        print(
            f"  Historical: {row['retrieved_text']}"
        )


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)

print(f"Saved: {OUTPUT}")