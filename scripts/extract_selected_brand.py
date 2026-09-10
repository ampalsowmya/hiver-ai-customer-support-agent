import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_FILE = Path("data/spotifycares.csv")

CHUNK_SIZE = 100_000
BRAND = "SpotifyCares"

print("=" * 70)
print("HIVER - EXTRACT SELECTED BRAND")
print("=" * 70)

first_write = True
total_rows = 0

for chunk_no, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    brand_df = df[
        df["author_id"].eq(BRAND)
        |
        (
            df["inbound"].eq(True)
            &
            df["in_response_to_tweet_id"].isin(
                df.loc[
                    df["author_id"].eq(BRAND),
                    "tweet_id"
                ]
            )
        )
    ]

    if len(brand_df) > 0:

        brand_df.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False
        total_rows += len(brand_df)

    print(
        f"Processed chunk {chunk_no} | "
        f"selected rows: {len(brand_df)}"
    )

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print("Spotify-related rows:", total_rows)
print("Saved:", OUTPUT_FILE)