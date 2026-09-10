import pandas as pd
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_DIR = Path("outputs")

CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("HIVER - CONVERSATION AUDIT")
print("=" * 70)

# ---------------------------------------------------------
# PASS 1
# Collect all inbound/customer tweet IDs
# ---------------------------------------------------------

print("\nPASS 1: Collecting customer tweet IDs...")

customer_tweet_ids = set()

total_rows = 0
customer_rows = 0
support_rows = 0

for chunk_no, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    total_rows += len(df)

    inbound_mask = df["inbound"] == True

    customer_ids = df.loc[
        inbound_mask,
        "tweet_id"
    ]

    customer_tweet_ids.update(
        customer_ids.tolist()
    )

    customer_rows += inbound_mask.sum()
    support_rows += (~inbound_mask).sum()

    print(
        f"Chunk {chunk_no}: "
        f"{len(df):,} rows"
    )

print("\nCustomer tweets:", f"{customer_rows:,}")
print("Support tweets:", f"{support_rows:,}")
print(
    "Unique customer tweet IDs:",
    f"{len(customer_tweet_ids):,}"
)

# ---------------------------------------------------------
# PASS 2
# Count actual support replies to customer tweets
# ---------------------------------------------------------

print("\nPASS 2: Measuring actual customer -> support responses...")

brand_stats = defaultdict(
    lambda: {
        "support_tweets": 0,
        "customer_messages_answered": 0,
        "support_replies_to_customer": 0
    }
)

for chunk_no, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    support_mask = df["inbound"] == False

    support_df = df.loc[
        support_mask
    ].copy()

    for _, row in support_df.iterrows():

        brand = row["author_id"]

        brand_stats[brand]["support_tweets"] += 1

        parent_id = row["in_response_to_tweet_id"]

        if pd.notna(parent_id):

            try:
                parent_id = int(parent_id)
            except:
                pass

            if parent_id in customer_tweet_ids:

                brand_stats[brand][
                    "customer_messages_answered"
                ] += 1

                brand_stats[brand][
                    "support_replies_to_customer"
                ] += 1

    print(
        f"Chunk {chunk_no} processed"
    )

# ---------------------------------------------------------
# Create results
# ---------------------------------------------------------

rows = []

for brand, stats in brand_stats.items():

    support_tweets = stats["support_tweets"]

    answered = stats[
        "customer_messages_answered"
    ]

    rows.append({
        "brand": brand,
        "support_tweets": support_tweets,
        "customer_messages_answered": answered,
        "support_replies_to_customer":
            stats["support_replies_to_customer"]
    })

results = pd.DataFrame(rows)

results = results.sort_values(
    "customer_messages_answered",
    ascending=False
)

results.to_csv(
    OUTPUT_DIR / "brand_conversation_statistics.csv",
    index=False
)

# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP BRANDS BY ACTUAL CUSTOMER RESPONSES")
print("=" * 70)

print(
    results.head(40).to_string(index=False)
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nSaved:"
    "\noutputs/brand_conversation_statistics.csv"
)