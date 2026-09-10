import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_DIR = Path("outputs")

CHUNK_SIZE = 100_000
SAMPLES_PER_BRAND = 200
RANDOM_SEED = 42

TARGET_BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "VirginTrains",
    "VerizonSupport",
]

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("HIVER - RANDOM CONVERSATION SAMPLING")
print("=" * 70)

# ---------------------------------------------------------
# Collect support tweets for candidate brands
# ---------------------------------------------------------

brand_rows = {
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

    support = df[
        (df["inbound"] == False)
        & (df["author_id"].isin(TARGET_BRANDS))
    ]

    for brand, group in support.groupby("author_id"):

        brand_rows[brand].extend(
            group[
                [
                    "tweet_id",
                    "author_id",
                    "created_at",
                    "text",
                    "in_response_to_tweet_id",
                    "response_tweet_id"
                ]
            ].to_dict("records")
        )

    print(f"Processed chunk {chunk_no}")

# ---------------------------------------------------------
# Random sample
# ---------------------------------------------------------

sampled = []

for brand in TARGET_BRANDS:

    df = pd.DataFrame(
        brand_rows[brand]
    )

    sample_size = min(
        SAMPLES_PER_BRAND,
        len(df)
    )

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )

    sample["brand"] = brand

    sampled.append(sample)

    print(
        f"{brand}: "
        f"{sample_size} sampled"
    )

result = pd.concat(
    sampled,
    ignore_index=True
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "random_candidate_sample.csv"
)

result.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("SAMPLE CREATED")
print("=" * 70)

print(
    result.groupby("brand")
    .size()
)

print("\nSaved:")
print(output_file)