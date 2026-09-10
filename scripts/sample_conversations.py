import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_DIR = Path("outputs")
CHUNK_SIZE = 100_000

TARGET_BRANDS = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "VirginTrains",
    "VerizonSupport",
}

SAMPLES_PER_BRAND = 100

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("HIVER - CONVERSATION SAMPLING")
print("=" * 70)

# ---------------------------------------------------------
# We first collect support tweets from our target brands
# ---------------------------------------------------------

support_samples = {
    brand: []
    for brand in TARGET_BRANDS
}

for chunk_no, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    df = df[
        (df["inbound"] == False)
        & (df["author_id"].isin(TARGET_BRANDS))
    ]

    for brand, group in df.groupby("author_id"):

        remaining = (
            SAMPLES_PER_BRAND
            - len(support_samples[brand])
        )

        if remaining <= 0:
            continue

        sample = group.head(remaining)

        support_samples[brand].extend(
            sample.to_dict("records")
        )

    print(f"Processed chunk {chunk_no}")

# ---------------------------------------------------------
# Create output
# ---------------------------------------------------------

rows = []

for brand, records in support_samples.items():

    for row in records:

        rows.append({
            "brand": brand,
            "support_tweet_id": row["tweet_id"],
            "support_text": row["text"],
            "parent_customer_tweet_id":
                row["in_response_to_tweet_id"],
            "response_tweet_id":
                row["response_tweet_id"],
            "created_at": row["created_at"]
        })

sample_df = pd.DataFrame(rows)

sample_df.to_csv(
    OUTPUT_DIR / "candidate_conversations_sample.csv",
    index=False
)

print("\n" + "=" * 70)
print("SAMPLE CREATED")
print("=" * 70)

print(
    sample_df.groupby("brand")
    .size()
    .sort_values(ascending=False)
)

print("\nSaved:")
print(
    "outputs/candidate_conversations_sample.csv"
)