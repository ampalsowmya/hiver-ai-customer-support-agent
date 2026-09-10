import pandas as pd
from pathlib import Path

INPUT_FILE = Path("outputs/spotify_intent_discovery_sample.csv")
OUTPUT_FILE = Path("outputs/spotify_golden_set.csv")

RANDOM_STATE = 42
GOLDEN_SIZE = 200

print("=" * 70)
print("HIVER - SPOTIFY GOLDEN SET CREATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load exploratory sample
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE, low_memory=False)

df["clean_text"] = (
    df["clean_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[df["clean_text"].ne("")].copy()

print(f"Exploratory messages: {len(df):,}")

# ---------------------------------------------------------
# 2. Remove exact duplicate customer messages
# ---------------------------------------------------------

df = df.drop_duplicates(
    subset=["clean_text"]
).reset_index(drop=True)

print(f"Unique messages: {len(df):,}")

# ---------------------------------------------------------
# 3. Randomly select candidate GOLDEN SET
# ---------------------------------------------------------

sample_size = min(GOLDEN_SIZE, len(df))

golden = df.sample(
    n=sample_size,
    random_state=RANDOM_STATE
).copy()

# ---------------------------------------------------------
# 4. Add annotation columns
# ---------------------------------------------------------

golden["intent"] = ""
golden["resolution_status"] = ""
golden["requires_escalation"] = ""
golden["annotation_notes"] = ""

# ---------------------------------------------------------
# 5. Annotation guidance
# ---------------------------------------------------------

golden["intent_options"] = (
    "ACCOUNT_LOGIN | "
    "SUBSCRIPTION_PREMIUM | "
    "BILLING_PAYMENT | "
    "PLAYBACK | "
    "APP_TECHNICAL | "
    "PLAYLIST_LIBRARY | "
    "FAMILY_DUO | "
    "CONTENT_AVAILABILITY | "
    "FEATURE_REQUEST | "
    "ACCOUNT_SECURITY | "
    "DEVICE_PLATFORM | "
    "OTHER"
)

golden["resolution_options"] = (
    "RESOLVED | "
    "UNRESOLVED | "
    "PARTIALLY_RESOLVED | "
    "INFORMATION_ONLY | "
    "UNCLEAR"
)

golden["escalation_options"] = (
    "YES | NO | UNCLEAR"
)

# ---------------------------------------------------------
# 6. Save
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

golden.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("GOLDEN SET CREATED")
print("=" * 70)

print(f"Golden-set size: {len(golden):,}")

print()
print("Annotation columns:")
print("  intent")
print("  resolution_status")
print("  requires_escalation")
print("  annotation_notes")

print()
print("Saved:")
print(f"  {OUTPUT_FILE}")

print()
print("IMPORTANT:")
print("Do NOT train or evaluate the AI model on this set yet.")
print("This set becomes the human-labelled evaluation ground truth.")