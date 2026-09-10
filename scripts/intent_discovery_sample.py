import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/spotifycares_clean.csv")
OUTPUT_FILE = Path("outputs/spotify_intent_discovery_sample.csv")

SAMPLE_SIZE = 500
RANDOM_STATE = 42

print("=" * 70)
print("HIVER - SPOTIFY INTENT DISCOVERY SAMPLE")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load cleaned Spotify dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Total Spotify rows: {len(df):,}")

# ---------------------------------------------------------
# 2. Keep CUSTOMER messages only
# ---------------------------------------------------------

inbound = (
    df["inbound"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

customers = df[inbound].copy()

print(f"Customer messages: {len(customers):,}")

# ---------------------------------------------------------
# 3. Remove empty messages
# ---------------------------------------------------------

customers = customers[
    customers["clean_text"].fillna("").str.strip().ne("")
].copy()

# ---------------------------------------------------------
# 4. Remove exact duplicate text for discovery
# ---------------------------------------------------------

customers = customers.drop_duplicates(
    subset=["clean_text"]
).reset_index(drop=True)

print(f"Unique customer messages: {len(customers):,}")

# ---------------------------------------------------------
# 5. Random sample
# ---------------------------------------------------------

sample_size = min(SAMPLE_SIZE, len(customers))

sample = customers.sample(
    n=sample_size,
    random_state=RANDOM_STATE
).copy()

# ---------------------------------------------------------
# 6. Select useful columns
# ---------------------------------------------------------

columns = [
    "tweet_id",
    "created_at",
    "author_id",
    "clean_text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

columns = [c for c in columns if c in sample.columns]

sample = sample[columns]

# ---------------------------------------------------------
# 7. Sort for easier inspection
# ---------------------------------------------------------

sample = sample.sort_values(
    "created_at",
    kind="stable"
).reset_index(drop=True)

# ---------------------------------------------------------
# 8. Save
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

sample.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("INTENT DISCOVERY SAMPLE COMPLETE")
print("=" * 70)

print(f"Customer messages available: {len(customers):,}")
print(f"Sample generated:            {len(sample):,}")
print()
print(f"Saved:")
print(f"  {OUTPUT_FILE}")