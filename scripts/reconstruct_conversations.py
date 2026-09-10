import pandas as pd
from pathlib import Path
import json

# =========================================================
# CONFIG
# =========================================================

DATA_FILE = Path("data/twcs.csv")
SAMPLE_FILE = Path("outputs/candidate_conversations_sample.csv")
OUTPUT_DIR = Path("outputs")

CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD SAMPLED SUPPORT TWEETS
# =========================================================

print("=" * 70)
print("HIVER - CONVERSATION RECONSTRUCTION")
print("=" * 70)

sample = pd.read_csv(SAMPLE_FILE)

print("\nSample size:", len(sample))

# IDs we definitely need
target_ids = set()

for _, row in sample.iterrows():

    # Support tweet
    if pd.notna(row["support_tweet_id"]):
        target_ids.add(int(row["support_tweet_id"]))

    # Parent customer tweet
    if pd.notna(row["parent_customer_tweet_id"]):
        target_ids.add(int(row["parent_customer_tweet_id"]))

print(
    "Initial target tweet IDs:",
    len(target_ids)
)

# =========================================================
# FIRST PASS
# Retrieve the selected tweets
# =========================================================

print("\nScanning dataset for selected tweets...")

selected_rows = []

for chunk_no, df in enumerate(
    pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    ids = pd.to_numeric(
        df["tweet_id"],
        errors="coerce"
    )

    mask = ids.isin(target_ids)

    if mask.any():

        selected_rows.extend(
            df.loc[mask].to_dict("records")
        )

    print(f"Processed chunk {chunk_no}")

selected = pd.DataFrame(selected_rows)

print(
    "\nTweets retrieved:",
    len(selected)
)

# =========================================================
# EXPAND ONE LEVEL IN BOTH DIRECTIONS
# =========================================================

new_ids = set()

for _, row in selected.iterrows():

    # Parent
    parent = row["in_response_to_tweet_id"]

    if pd.notna(parent):

        try:
            new_ids.add(int(parent))
        except:
            pass

    # Children
    children = row["response_tweet_id"]

    if pd.notna(children):

        for child in str(children).split(","):

            child = child.strip()

            if child:

                try:
                    new_ids.add(int(child))
                except:
                    pass

target_ids.update(new_ids)

print(
    "Expanded target IDs:",
    len(target_ids)
)

# =========================================================
# SECOND PASS
# Retrieve expanded conversation
# =========================================================

print("\nSecond pass: retrieving conversation context...")

selected_rows = []

for chunk_no, df in enumerate(
    pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    ids = pd.to_numeric(
        df["tweet_id"],
        errors="coerce"
    )

    mask = ids.isin(target_ids)

    if mask.any():

        selected_rows.extend(
            df.loc[mask].to_dict("records")
        )

    print(f"Processed chunk {chunk_no}")

selected = pd.DataFrame(selected_rows)

# Remove duplicates
selected = selected.drop_duplicates(
    subset=["tweet_id"]
)

# =========================================================
# ADD SPEAKER LABEL
# =========================================================

selected["speaker"] = selected["inbound"].map(
    {
        True: "customer",
        False: "support"
    }
)

# =========================================================
# SAVE FLAT CONVERSATION DATA
# =========================================================

flat_file = (
    OUTPUT_DIR /
    "reconstructed_conversation_tweets.csv"
)

selected.to_csv(
    flat_file,
    index=False
)

print(
    "\nSaved:",
    flat_file
)

# =========================================================
# BUILD CONVERSATION THREADS
# =========================================================

tweet_map = {}

for _, row in selected.iterrows():

    tweet_id = int(row["tweet_id"])

    tweet_map[tweet_id] = {
        "tweet_id": tweet_id,
        "speaker": row["speaker"],
        "author_id": str(row["author_id"]),
        "created_at": row["created_at"],
        "text": str(row["text"]),
        "parent_id": (
            int(row["in_response_to_tweet_id"])
            if pd.notna(
                row["in_response_to_tweet_id"]
            )
            else None
        ),
    }

# =========================================================
# CREATE CONVERSATION RECORDS
# =========================================================

conversation_records = []

for _, sample_row in sample.iterrows():

    brand = sample_row["brand"]

    support_id = int(
        sample_row["support_tweet_id"]
    )

    if support_id not in tweet_map:
        continue

    # Walk backwards from support tweet
    chain = []

    current_id = support_id

    visited = set()

    while (
        current_id in tweet_map
        and current_id not in visited
    ):

        visited.add(current_id)

        tweet = tweet_map[current_id]

        chain.append(tweet)

        parent_id = tweet["parent_id"]

        if parent_id is None:
            break

        current_id = parent_id

    # Reverse so oldest comes first
    chain.reverse()

    conversation_records.append(
        {
            "brand": brand,
            "support_tweet_id": support_id,
            "num_messages": len(chain),
            "conversation": json.dumps(
                chain,
                ensure_ascii=False
            )
        }
    )

# =========================================================
# SAVE JSONL
# =========================================================

jsonl_file = (
    OUTPUT_DIR /
    "reconstructed_conversations.jsonl"
)

with open(
    jsonl_file,
    "w",
    encoding="utf-8"
) as f:

    for record in conversation_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

print(
    "Saved:",
    jsonl_file
)

# =========================================================
# SUMMARY
# =========================================================

result = pd.DataFrame(
    conversation_records
)

if not result.empty:

    summary = (
        result
        .groupby("brand")
        .agg(
            conversations=(
                "support_tweet_id",
                "count"
            ),
            avg_messages=(
                "num_messages",
                "mean"
            ),
            max_messages=(
                "num_messages",
                "max"
            )
        )
        .reset_index()
    )

    summary_file = (
        OUTPUT_DIR /
        "conversation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("CONVERSATION SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        summary_file
    )

print("\n" + "=" * 70)
print("RECONSTRUCTION COMPLETE")
print("=" * 70)