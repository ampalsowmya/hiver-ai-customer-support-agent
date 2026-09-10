import pandas as pd
from pathlib import Path

INPUT_FILE = Path("outputs/spotify_intent_clusters.csv")
OUTPUT_FILE = Path("outputs/spotify_cluster_representatives.csv")

SAMPLES_PER_CLUSTER = 10
RANDOM_STATE = 42

print("=" * 70)
print("HIVER - SPOTIFY CLUSTER REPRESENTATIVE INSPECTION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["clean_text"] = (
    df["clean_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(f"Total clustered messages: {len(df):,}")
print(f"Number of clusters: {df['cluster'].nunique()}")

# ---------------------------------------------------------
# Select representative messages from every cluster
# ---------------------------------------------------------

representatives = []

for cluster_id in sorted(df["cluster"].unique()):

    cluster_df = df[df["cluster"] == cluster_id].copy()

    sample_size = min(
        SAMPLES_PER_CLUSTER,
        len(cluster_df)
    )

    sample = cluster_df.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    sample = sample[
        [
            "cluster",
            "tweet_id",
            "created_at",
            "clean_text"
        ]
    ]

    representatives.append(sample)

representatives = pd.concat(
    representatives,
    ignore_index=True
)

# Sort by cluster
representatives = representatives.sort_values(
    ["cluster", "created_at"],
    kind="stable"
).reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

representatives.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# Print to terminal
# ---------------------------------------------------------

print()
print("=" * 70)
print("REPRESENTATIVE MESSAGES")
print("=" * 70)

for cluster_id in sorted(
    representatives["cluster"].unique()
):

    cluster_messages = representatives[
        representatives["cluster"] == cluster_id
    ]

    total = (
        df["cluster"] == cluster_id
    ).sum()

    print()
    print(
        f"CLUSTER {cluster_id} "
        f"({total} total messages)"
    )
    print("-" * 70)

    for i, text in enumerate(
        cluster_messages["clean_text"],
        start=1
    ):

        print(f"{i:02d}. {text}")

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(f"Saved:")
print(f"  {OUTPUT_FILE}")