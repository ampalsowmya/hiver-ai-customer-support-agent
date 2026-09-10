import pandas as pd
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_DIR = Path("outputs")
CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# GLOBAL STATISTICS
# ---------------------------------------------------------

total_rows = 0
inbound_rows = 0
outbound_rows = 0

missing_counts = Counter()

brand_stats = {}

unique_tweet_ids = set()

duplicate_tweet_ids = 0
duplicate_texts = Counter()

all_columns = None

# ---------------------------------------------------------
# READ DATA IN CHUNKS
# ---------------------------------------------------------

print("=" * 70)
print("HIVER SDE INTERN - DATASET AUDIT")
print("=" * 70)

print(f"\nInput file: {INPUT_FILE}")
print(f"Chunk size: {CHUNK_SIZE:,}\n")

for chunk_number, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    print(f"Processing chunk {chunk_number}...")

    total_rows += len(df)

    # Save column information
    if all_columns is None:
        all_columns = list(df.columns)

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    for column in df.columns:
        missing_counts[column] += df[column].isna().sum()

    # -----------------------------------------------------
    # INBOUND / OUTBOUND
    # -----------------------------------------------------

    inbound_rows += (df["inbound"] == True).sum()
    outbound_rows += (df["inbound"] == False).sum()

    # -----------------------------------------------------
    # DUPLICATE TWEET IDS
    # -----------------------------------------------------

    for tweet_id in df["tweet_id"]:
        if tweet_id in unique_tweet_ids:
            duplicate_tweet_ids += 1
        else:
            unique_tweet_ids.add(tweet_id)

    # -----------------------------------------------------
    # TEXT DUPLICATES
    # -----------------------------------------------------

    text_counts = df["text"].fillna("").value_counts()

    for text, count in text_counts.items():
        duplicate_texts[text] += count

    # -----------------------------------------------------
    # BRAND / AUTHOR STATISTICS
    # -----------------------------------------------------

    for author_id, group in df.groupby("author_id"):

        if author_id not in brand_stats:
            brand_stats[author_id] = {
                "tweets": 0,
                "inbound": 0,
                "outbound": 0,
                "responses": 0
            }

        brand_stats[author_id]["tweets"] += len(group)

        brand_stats[author_id]["inbound"] += (
            group["inbound"] == True
        ).sum()

        brand_stats[author_id]["outbound"] += (
            group["inbound"] == False
        ).sum()

        brand_stats[author_id]["responses"] += (
            group["response_tweet_id"].notna()
        ).sum()

# ---------------------------------------------------------
# BRAND DATAFRAME
# ---------------------------------------------------------

brand_rows = []

for author_id, stats in brand_stats.items():

    brand_rows.append({
        "author_id": author_id,
        "tweets": stats["tweets"],
        "inbound": stats["inbound"],
        "outbound": stats["outbound"],
        "tweets_with_responses": stats["responses"]
    })

brands_df = pd.DataFrame(brand_rows)

brands_df["response_rate"] = (
    brands_df["tweets_with_responses"]
    / brands_df["tweets"].replace(0, 1)
)

# Sort by total activity
brands_df = brands_df.sort_values(
    "tweets",
    ascending=False
)

# ---------------------------------------------------------
# SAVE BRAND STATISTICS
# ---------------------------------------------------------

brands_df.to_csv(
    OUTPUT_DIR / "brand_statistics.csv",
    index=False
)

# ---------------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "total_rows",
        "inbound_customer_tweets",
        "outbound_support_tweets",
        "unique_tweet_ids",
        "duplicate_tweet_ids"
    ],
    "value": [
        total_rows,
        inbound_rows,
        outbound_rows,
        len(unique_tweet_ids),
        duplicate_tweet_ids
    ]
})

summary.to_csv(
    OUTPUT_DIR / "dataset_summary.csv",
    index=False
)

# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------

missing_df = pd.DataFrame(
    [
        {
            "column": column,
            "missing_values": count,
            "missing_percentage": (
                count / total_rows * 100
                if total_rows > 0 else 0
            )
        }
        for column, count in missing_counts.items()
    ]
)

missing_df.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    index=False
)

# ---------------------------------------------------------
# TOP AUTHORS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP AUTHORS / ACCOUNTS")
print("=" * 70)

print(
    brands_df.head(30).to_string(index=False)
)

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(missing_df.to_string(index=False))

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nFiles created:")
print("  outputs/dataset_summary.csv")
print("  outputs/brand_statistics.csv")
print("  outputs/missing_values.csv")