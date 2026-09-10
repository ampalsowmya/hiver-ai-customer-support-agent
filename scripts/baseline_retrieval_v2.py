from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = ROOT / "outputs" / "spotify_golden_annotation_context.csv"
CONV_PATH = ROOT / "outputs" / "reconstructed_conversations.jsonl"
OUT_PATH = ROOT / "outputs" / "spotify_retrieval_v2_predictions.csv"


def normalize(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_conversations(path):
    conversations = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                conversations.append(record)

            except json.JSONDecodeError:
                continue

    return conversations


def parse_conversation(record):
    """
    The reconstruction file stores the actual conversation
    as a JSON-encoded string inside record["conversation"].
    """

    raw_conversation = record.get("conversation", "")

    if not raw_conversation:
        return []

    if isinstance(raw_conversation, str):

        try:
            messages = json.loads(raw_conversation)

        except json.JSONDecodeError:
            return []

    elif isinstance(raw_conversation, list):

        messages = raw_conversation

    else:
        return []

    if not isinstance(messages, list):
        return []

    return messages


print("=" * 75)
print("HIVER BASELINE — HISTORICAL SUPPORT CASE RETRIEVAL V2")
print("=" * 75)


# ---------------------------------------------------------------------
# 1. Load Golden Set
# ---------------------------------------------------------------------

golden = pd.read_csv(GOLDEN_PATH)

golden["tweet_id"] = golden["tweet_id"].astype(str)

golden_ids = set(golden["tweet_id"])

print(f"Golden Set rows: {len(golden)}")


# ---------------------------------------------------------------------
# 2. Load reconstructed conversations
# ---------------------------------------------------------------------

records = load_conversations(CONV_PATH)

print(f"Reconstructed conversations loaded: {len(records)}")


# ---------------------------------------------------------------------
# 3. Filter to SpotifyCares
# ---------------------------------------------------------------------

spotify_records = [
    r for r in records
    if str(r.get("brand", "")).lower() == "spotifycares"
]

print(f"SpotifyCares conversations: {len(spotify_records)}")


# ---------------------------------------------------------------------
# 4. Convert conversations into historical support cases
# ---------------------------------------------------------------------

cases = []

for record in spotify_records:

    messages = parse_conversation(record)

    if not messages:
        continue

    customer_parts = []
    support_parts = []

    tweet_ids = []

    for message in messages:

        if not isinstance(message, dict):
            continue

        text = normalize(message.get("text", ""))

        if not text:
            continue

        speaker = str(
            message.get("speaker", "")
        ).lower()

        tweet_id = message.get("tweet_id")

        if tweet_id is not None:
            tweet_ids.append(str(tweet_id))

        if speaker == "customer":
            customer_parts.append(text)

        elif speaker == "support":
            support_parts.append(text)

    customer_text = " ".join(customer_parts)
    support_text = " ".join(support_parts)

    if not customer_text:
        continue

    cases.append({
        "support_tweet_id": str(
            record.get("support_tweet_id", "")
        ),
        "tweet_ids": "|".join(tweet_ids),
        "customer_text": customer_text,
        "support_text": support_text,
        "case_text": normalize(
            customer_text + " " + support_text
        )
    })


cases_df = pd.DataFrame(cases)

print(f"Usable Spotify support cases: {len(cases_df)}")


# ---------------------------------------------------------------------
# 5. Remove Golden Set conversations
# ---------------------------------------------------------------------

def contains_golden_id(tweet_ids):
    ids = set(tweet_ids.split("|"))

    return bool(ids.intersection(golden_ids))


before = len(cases_df)

cases_df = cases_df[
    ~cases_df["tweet_ids"].apply(contains_golden_id)
].copy()

print(
    f"Golden Set cases removed: "
    f"{before - len(cases_df)}"
)


# ---------------------------------------------------------------------
# 6. Remove duplicate cases
# ---------------------------------------------------------------------

cases_df = cases_df.drop_duplicates(
    subset=["case_text"]
).reset_index(drop=True)

print(
    f"Final unique historical cases: "
    f"{len(cases_df)}"
)


if cases_df.empty:

    raise RuntimeError(
        "No historical Spotify support cases remain."
    )


# ---------------------------------------------------------------------
# 7. Build TF-IDF index
#
# Important:
# We index CUSTOMER PROBLEM TEXT.
# The associated SUPPORT RESPONSE is returned with the case.
# ---------------------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=50000
)


case_matrix = vectorizer.fit_transform(
    cases_df["customer_text"]
)


query_matrix = vectorizer.transform(
    golden["clean_text"]
    .fillna("")
    .map(normalize)
)


print(
    f"TF-IDF matrix: "
    f"{case_matrix.shape}"
)


# ---------------------------------------------------------------------
# 8. Retrieve Top 5
# ---------------------------------------------------------------------

similarities = cosine_similarity(
    query_matrix,
    case_matrix
)


rows = []


for i, golden_row in golden.iterrows():

    scores = similarities[i]

    top_indices = np.argsort(scores)[::-1][:5]

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        case = cases_df.iloc[idx]

        rows.append({

            "golden_id":
                golden_row["annotation_id"],

            "golden_intent":
                golden_row["intent"],

            "golden_query":
                golden_row["clean_text"],

            "rank":
                rank,

            "similarity":
                float(scores[idx]),

            "historical_support_tweet_id":
                case["support_tweet_id"],

            "historical_customer":
                case["customer_text"],

            "historical_support":
                case["support_text"]

        })


results = pd.DataFrame(rows)


results.to_csv(
    OUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ---------------------------------------------------------------------
# 9. Diagnostics
# ---------------------------------------------------------------------

top1 = results[
    results["rank"] == 1
]["similarity"]


top5_best = (
    results
    .groupby("golden_id")["similarity"]
    .max()
)


print()
print("=" * 75)
print("RETRIEVAL DIAGNOSTICS")
print("=" * 75)

print(
    f"Evaluation queries: "
    f"{len(golden)}"
)

print(
    f"Mean Top-1 similarity: "
    f"{top1.mean():.4f}"
)

print(
    f"Median Top-1 similarity: "
    f"{top1.median():.4f}"
)

print(
    f"Mean best-of-5 similarity: "
    f"{top5_best.mean():.4f}"
)

for threshold in [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60
]:

    coverage = (
        top5_best >= threshold
    ).mean()

    print(
        f"Top-5 similarity >= "
        f"{threshold:.2f}: "
        f"{coverage:.2%}"
    )


# ---------------------------------------------------------------------
# 10. Inspect important examples
# ---------------------------------------------------------------------

print()
print("=" * 75)
print("SAMPLE HISTORICAL SUPPORT CASE RETRIEVALS")
print("=" * 75)


for golden_id in [
    1,
    2,
    3,
    5,
    12
]:

    subset = results[
        results["golden_id"] == golden_id
    ]

    if subset.empty:
        continue

    first = subset.iloc[0]

    print()
    print("-" * 75)

    print(
        f"Golden ID : {golden_id}"
    )

    print(
        f"Intent    : "
        f"{first['golden_intent']}"
    )

    print(
        f"Query     : "
        f"{first['golden_query']}"
    )

    for _, row in subset.iterrows():

        print()

        print(
            f"Rank {row['rank']} | "
            f"similarity="
            f"{row['similarity']:.4f}"
        )

        print(
            "Historical customer: "
            f"{row['historical_customer'][:600]}"
        )

        print(
            "Historical support: "
            f"{row['historical_support'][:600]}"
        )


print()
print("=" * 75)
print("DONE")
print("=" * 75)

print(
    f"Saved: {OUT_PATH}"
)