import pandas as pd
from pathlib import Path
from collections import Counter
import re

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_DIR = Path("outputs")
CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# BRANDS WE WANT TO INVESTIGATE
# ---------------------------------------------------------

TARGET_BRANDS = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "VirginTrains",
    "VerizonSupport",
}

# ---------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------

stats = {
    brand: {
        "support_tweets": 0,
        "customer_replies": 0,
        "support_reply_texts": [],
        "support_text_lengths": [],
        "dm_requests": 0,
        "apology_count": 0,
        "resolution_signal_count": 0,
    }
    for brand in TARGET_BRANDS
}

# Words/phrases that indicate an actual action or resolution.
RESOLUTION_PATTERNS = [
    "refund",
    "refunded",
    "replace",
    "replacement",
    "reset",
    "cancel",
    "cancelled",
    "canceled",
    "activate",
    "activated",
    "update",
    "updated",
    "fixed",
    "fix",
    "resolved",
    "check",
    "checked",
    "tracking",
    "track",
    "delivery",
    "delivered",
    "invoice",
    "credit",
    "charged",
    "charge",
    "payment",
    "account",
    "password",
    "ticket",
    "case",
    "booking",
    "reservation",
]

print("=" * 70)
print("HIVER - BRAND QUALITY AUDIT")
print("=" * 70)

# ---------------------------------------------------------
# READ DATA
# ---------------------------------------------------------

for chunk_no, df in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    # Only target support accounts
    support_df = df[
        (df["inbound"] == False)
        & (df["author_id"].isin(TARGET_BRANDS))
    ].copy()

    if support_df.empty:
        continue

    for brand, group in support_df.groupby("author_id"):

        stats[brand]["support_tweets"] += len(group)

        texts = (
            group["text"]
            .fillna("")
            .astype(str)
        )

        for text in texts:

            text_lower = text.lower()

            stats[brand][
                "support_reply_texts"
            ].append(text)

            stats[brand][
                "support_text_lengths"
            ].append(len(text.split()))

            # DM / private-contact requests
            if (
                "dm" in text_lower
                or "direct message" in text_lower
            ):
                stats[brand]["dm_requests"] += 1

            # Apologies
            if (
                "sorry" in text_lower
                or "apolog" in text_lower
            ):
                stats[brand]["apology_count"] += 1

            # Resolution/action signals
            if any(
                phrase in text_lower
                for phrase in RESOLUTION_PATTERNS
            ):
                stats[brand][
                    "resolution_signal_count"
                ] += 1

    print(f"Processed chunk {chunk_no}")

# ---------------------------------------------------------
# CREATE RESULTS
# ---------------------------------------------------------

results = []

for brand, s in stats.items():

    texts = s["support_reply_texts"]

    total = len(texts)

    unique_texts = len(set(texts))

    avg_length = (
        sum(s["support_text_lengths"]) / total
        if total else 0
    )

    dm_rate = (
        s["dm_requests"] / total
        if total else 0
    )

    apology_rate = (
        s["apology_count"] / total
        if total else 0
    )

    resolution_rate = (
        s["resolution_signal_count"] / total
        if total else 0
    )

    response_diversity = (
        unique_texts / total
        if total else 0
    )

    results.append({
        "brand": brand,
        "support_tweets": total,
        "unique_support_responses": unique_texts,
        "response_diversity": response_diversity,
        "average_response_words": avg_length,
        "dm_request_rate": dm_rate,
        "apology_rate": apology_rate,
        "resolution_signal_rate": resolution_rate,
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

results_df = results_df.sort_values(
    "support_tweets",
    ascending=False
)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "brand_quality_audit.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("BRAND QUALITY RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)