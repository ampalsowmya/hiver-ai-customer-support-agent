import pandas as pd
from pathlib import Path

GOLDEN_FILE = Path("outputs/spotify_golden_set_annotation.csv")
DATA_FILE = Path("data/spotifycares_clean.csv")
OUTPUT_FILE = Path("outputs/spotify_golden_annotation_context.csv")

golden = pd.read_csv(GOLDEN_FILE, low_memory=False)
data = pd.read_csv(DATA_FILE, low_memory=False)

# Normalize IDs to strings so joins work reliably
data["tweet_id"] = data["tweet_id"].astype(str)
data["response_tweet_id"] = data["response_tweet_id"].fillna("").astype(str)
data["in_response_to_tweet_id"] = (
    data["in_response_to_tweet_id"].fillna("").astype(str)
)

golden["tweet_id"] = golden["tweet_id"].astype(str)

# Fast lookup
tweet_map = data.set_index("tweet_id").to_dict("index")

def get_context(tweet_id, max_messages=10):
    """
    Build a small conversation context around the selected customer tweet.

    We walk backwards through in_response_to_tweet_id and then forwards
    through response_tweet_id where possible.
    """

    if tweet_id not in tweet_map:
        return ""

    current = tweet_map[tweet_id]

    chain = [current]

    # ---------------------------------------------------------
    # Walk backwards
    # ---------------------------------------------------------
    visited = {tweet_id}
    parent_id = str(current.get("in_response_to_tweet_id", ""))

    while parent_id and parent_id != "nan":
        if parent_id in visited:
            break

        if parent_id not in tweet_map:
            break

        parent = tweet_map[parent_id]
        chain.insert(0, parent)
        visited.add(parent_id)

        parent_id = str(parent.get("in_response_to_tweet_id", ""))

        if len(chain) >= max_messages:
            break

    # ---------------------------------------------------------
    # Walk forwards
    # ---------------------------------------------------------
    current_id = tweet_id

    while len(chain) < max_messages:
        current_row = tweet_map.get(current_id)

        if not current_row:
            break

        child_id = str(current_row.get("response_tweet_id", ""))

        if not child_id or child_id == "nan":
            break

        if child_id in visited:
            break

        if child_id not in tweet_map:
            break

        child = tweet_map[child_id]

        chain.append(child)
        visited.add(child_id)
        current_id = child_id

    # ---------------------------------------------------------
    # Format context
    # ---------------------------------------------------------
    lines = []

    for row in chain:
        inbound = bool(row.get("inbound", False))

        if inbound:
            speaker = "CUSTOMER"
        else:
            speaker = "SPOTIFY"

        text = str(row.get("clean_text", "")).strip()

        if text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


golden["conversation_context"] = golden["tweet_id"].apply(
    get_context
)

# Put context immediately after customer message
columns = list(golden.columns)

columns.remove("conversation_context")

insert_at = columns.index("clean_text") + 1

columns = (
    columns[:insert_at]
    + ["conversation_context"]
    + columns[insert_at:]
)

golden = golden[columns]

golden.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 70)
print("HIVER - GOLDEN SET CONVERSATION CONTEXT")
print("=" * 70)

print(f"Golden examples: {len(golden)}")
print(
    "Examples with context:",
    golden["conversation_context"].astype(str).str.len().gt(0).sum()
)

print()
print("Saved:")
print(f"  {OUTPUT_FILE}")

print()
print("Sample:")
print("-" * 70)

for _, row in golden.head(5).iterrows():
    print(f"\nAnnotation ID: {row['annotation_id']}")
    print(f"Customer tweet: {row['clean_text']}")
    print("Conversation:")
    print(row["conversation_context"])
    print("-" * 70)