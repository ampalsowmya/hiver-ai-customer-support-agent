import pandas as pd
import html
import re
from pathlib import Path

INPUT_FILE = Path("data/spotifycares.csv")
OUTPUT_FILE = Path("data/spotifycares_clean.csv")
QUALITY_FILE = Path("outputs/spotify_data_quality.csv")
REPORT_FILE = Path("outputs/spotify_data_quality_report.txt")

print("=" * 70)
print("HIVER - SPOTIFYCARES DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded rows: {len(df):,}")

# ---------------------------------------------------------
# 1. Preserve original text
# ---------------------------------------------------------

df["raw_text"] = df["text"].fillna("").astype(str)

# ---------------------------------------------------------
# 2. Clean text
# ---------------------------------------------------------

def clean_text(text):
    text = html.unescape(text)

    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " <URL> ",
        text
    )

    # Twitter mentions
    text = re.sub(
        r"@\w+",
        " <USER> ",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df["raw_text"].apply(clean_text)

# ---------------------------------------------------------
# 3. Parse timestamps
# ---------------------------------------------------------

df["created_at_parsed"] = pd.to_datetime(
    df["created_at"],
    errors="coerce"
)

# ---------------------------------------------------------
# 4. Basic quality checks
# ---------------------------------------------------------

duplicate_tweet_ids = df["tweet_id"].duplicated(keep=False).sum()

duplicate_text = df["raw_text"].duplicated(keep=False).sum()

empty_text = df["raw_text"].str.strip().eq("").sum()

invalid_dates = df["created_at_parsed"].isna().sum()

# ---------------------------------------------------------
# 5. Customer vs support messages
# ---------------------------------------------------------

inbound = (
    df["inbound"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

customer_rows = inbound.sum()
support_rows = (~inbound).sum()

# ---------------------------------------------------------
# 6. Parent-reference validation
# ---------------------------------------------------------

tweet_ids = set(df["tweet_id"].astype(str))

parent_ids = (
    df["in_response_to_tweet_id"]
    .dropna()
    .astype(str)
)

missing_parent_refs = (
    ~parent_ids.isin(tweet_ids)
).sum()

# ---------------------------------------------------------
# 7. Sort chronologically
# ---------------------------------------------------------

df = df.sort_values(
    ["created_at_parsed", "tweet_id"],
    kind="stable"
).reset_index(drop=True)

# ---------------------------------------------------------
# 8. Create clean analysis copy
# ---------------------------------------------------------

clean_df = (
    df
    .drop_duplicates(
        subset=["tweet_id"],
        keep="first"
    )
    .reset_index(drop=True)
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------
# 9. Quality report
# ---------------------------------------------------------

metrics = pd.DataFrame({
    "metric": [
        "raw_rows",
        "clean_rows",
        "duplicate_tweet_id_rows",
        "duplicate_text_rows",
        "empty_text_rows",
        "invalid_created_at_rows",
        "customer_inbound_rows",
        "support_outbound_rows",
        "missing_parent_references"
    ],
    "value": [
        len(df),
        len(clean_df),
        duplicate_tweet_ids,
        duplicate_text,
        empty_text,
        invalid_dates,
        customer_rows,
        support_rows,
        missing_parent_refs
    ]
})

metrics.to_csv(
    QUALITY_FILE,
    index=False
)

# ---------------------------------------------------------
# 10. Human-readable report
# ---------------------------------------------------------

report = f"""
HIVER - SPOTIFYCARES DATA QUALITY REPORT
=========================================

Raw rows:
{len(df):,}

Clean rows:
{len(clean_df):,}

Duplicate tweet IDs:
{duplicate_tweet_ids:,}

Duplicate tweet text:
{duplicate_text:,}

Empty text:
{empty_text:,}

Invalid timestamps:
{invalid_dates:,}

Customer / inbound rows:
{customer_rows:,}

Support / outbound rows:
{support_rows:,}

Parent references missing from selected dataset:
{missing_parent_refs:,}

OUTPUTS
=======

{OUTPUT_FILE}
{QUALITY_FILE}
"""

REPORT_FILE.write_text(
    report,
    encoding="utf-8"
)

print()
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Raw rows:        {len(df):,}")
print(f"Clean rows:      {len(clean_df):,}")
print(f"Customer rows:   {customer_rows:,}")
print(f"Support rows:    {support_rows:,}")
print(f"Duplicate IDs:   {duplicate_tweet_ids:,}")
print(f"Empty text:      {empty_text:,}")
print(f"Invalid dates:   {invalid_dates:,}")
print()
print("Saved:")
print(f"  {OUTPUT_FILE}")
print(f"  {QUALITY_FILE}")
print(f"  {REPORT_FILE}")