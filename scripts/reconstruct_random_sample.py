import pandas as pd
from pathlib import Path
import json

# =========================================================
# CONFIG
# =========================================================

DATA_FILE = Path("data/twcs.csv")
SAMPLE_FILE = Path("outputs/random_candidate_sample.csv")
OUTPUT_DIR = Path("outputs")

CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# LOAD RANDOM SAMPLE
# =========================================================

sample = pd.read_csv(SAMPLE_FILE)

print("=" * 70)
print("HIVER - RANDOM SAMPLE CONVERSATION RECONSTRUCTION")
print("=" * 70)

print("\nSample size:", len(sample))

# The support tweets we sampled
target_ids = set(
    pd.to_numeric(
        sample["tweet_id"],
        errors="coerce"
    ).dropna().astype(int)
)

print(
    "Initial support tweet IDs:",
    len(target_ids)
)

# =========================================================
# ITERATIVELY WALK BACK THROUGH PARENTS
# =========================================================

all_conversation_ids = set(target_ids)

previous_count = 0
iteration = 0

while len(all_conversation_ids) > previous_count:

    iteration += 1
    previous_count = len(all_conversation_ids)

    print("\n" + "-" * 70)
    print(f"PASS {iteration}")
    print("-" * 70)

    found_rows = []

    for chunk_no, df in enumerate(
        pd.read_csv(
            DATA_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        tweet_ids = pd.to_numeric(
            df["tweet_id"],
            errors="coerce"
        )

        mask = tweet_ids.isin(
            all_conversation_ids
        )

        if mask.any():

            found_rows.append(
                df.loc[mask].copy()
            )

        print(
            f"Processed chunk {chunk_no}"
        )

    if not found_rows:
        break

    found = pd.concat(
        found_rows,
        ignore_index=True
    )

    found = found.drop_duplicates(
        subset=["tweet_id"]
    )

    # Add parents
    parent_ids = pd.to_numeric(
        found["in_response_to_tweet_id"],
        errors="coerce"
    ).dropna().astype(int)

    new_parent_ids = set(parent_ids) - all_conversation_ids

    all_conversation_ids.update(
        new_parent_ids
    )

    print(
        "Tweets found:",
        len(found)
    )

    print(
        "New parent tweets:",
        len(new_parent_ids)
    )

    print(
        "Total conversation IDs:",
        len(all_conversation_ids)
    )

    if len(new_parent_ids) == 0:
        break

# =========================================================
# FINAL RETRIEVAL
# =========================================================

print("\n" + "=" * 70)
print("FINAL RETRIEVAL")
print("=" * 70)

all_rows = []

for chunk_no, df in enumerate(
    pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    tweet_ids = pd.to_numeric(
        df["tweet_id"],
        errors="coerce"
    )

    mask = tweet_ids.isin(
        all_conversation_ids
    )

    if mask.any():

        all_rows.append(
            df.loc[mask].copy()
        )

    print(
        f"Processed chunk {chunk_no}"
    )

tweets = pd.concat(
    all_rows,
    ignore_index=True
)

tweets = tweets.drop_duplicates(
    subset=["tweet_id"]
)

# =========================================================
# SPEAKER
# =========================================================

tweets["speaker"] = tweets["inbound"].map(
    {
        True: "CUSTOMER",
        False: "SUPPORT"
    }
)

# =========================================================
# MAP TWEETS
# =========================================================

tweet_map = {}

for _, row in tweets.iterrows():

    tweet_id = int(row["tweet_id"])

    parent_id = None

    if pd.notna(
        row["in_response_to_tweet_id"]
    ):

        try:
            parent_id = int(
                row["in_response_to_tweet_id"]
            )
        except:
            parent_id = None

    tweet_map[tweet_id] = {
        "tweet_id": tweet_id,
        "speaker": row["speaker"],
        "author_id": str(row["author_id"]),
        "created_at": str(row["created_at"]),
        "text": str(row["text"]),
        "parent_id": parent_id
    }

# =========================================================
# BUILD CONVERSATIONS
# =========================================================

conversation_records = []

for _, sample_row in sample.iterrows():

    brand = sample_row["brand"]

    support_id = int(
        sample_row["tweet_id"]
    )

    if support_id not in tweet_map:
        continue

    chain = []

    current_id = support_id

    visited = set()

    while (
        current_id in tweet_map
        and current_id not in visited
    ):

        visited.add(current_id)

        message = tweet_map[current_id]

        chain.append(message)

        parent_id = message["parent_id"]

        if parent_id is None:
            break

        current_id = parent_id

    chain.reverse()

    conversation_records.append(
        {
            "brand": brand,
            "support_tweet_id": support_id,
            "num_messages": len(chain),
            "conversation": chain
        }
    )

# =========================================================
# SAVE JSONL
# =========================================================

jsonl_file = (
    OUTPUT_DIR /
    "random_reconstructed_conversations.jsonl"
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

# =========================================================
# CREATE READABLE TEXT FILE
# =========================================================

readable_file = (
    OUTPUT_DIR /
    "random_conversations_readable.txt"
)

with open(
    readable_file,
    "w",
    encoding="utf-8"
) as f:

    for i, record in enumerate(
        conversation_records,
        start=1
    ):

        f.write("\n")
        f.write("=" * 80)
        f.write("\n")

        f.write(
            f"CONVERSATION {i}\n"
        )

        f.write(
            f"BRAND: {record['brand']}\n"
        )

        f.write(
            f"SUPPORT TWEET: "
            f"{record['support_tweet_id']}\n"
        )

        f.write(
            f"MESSAGES: "
            f"{record['num_messages']}\n"
        )

        f.write("=" * 80)
        f.write("\n\n")

        for message in record["conversation"]:

            f.write(
                f"[{message['speaker']}] "
                f"{message['text']}\n"
            )

            f.write(
                f"    tweet_id: "
                f"{message['tweet_id']}\n"
            )

            f.write("\n")

# =========================================================
# SUMMARY
# =========================================================

summary_rows = []

for record in conversation_records:

    summary_rows.append(
        {
            "brand": record["brand"],
            "support_tweet_id":
                record["support_tweet_id"],
            "num_messages":
                record["num_messages"]
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUTPUT_DIR /
    "random_conversation_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# =========================================================
# DISPLAY SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("RANDOM CONVERSATION SUMMARY")
print("=" * 70)

if not summary.empty:

    brand_summary = (
        summary
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
            median_messages=(
                "num_messages",
                "median"
            ),
            max_messages=(
                "num_messages",
                "max"
            )
        )
        .reset_index()
    )

    print(
        brand_summary.to_string(
            index=False
        )
    )

print("\nSaved:")
print(jsonl_file)
print(readable_file)
print(summary_file)

print("\n" + "=" * 70)
print("RECONSTRUCTION COMPLETE")
print("=" * 70)