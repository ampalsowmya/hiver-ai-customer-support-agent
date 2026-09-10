import pandas as pd

INPUT = "outputs/spotify_golden_annotation_context.csv"

VALID_INTENTS = {
    "ACCOUNT_LOGIN",
    "ACCOUNT_SECURITY",
    "SUBSCRIPTION_PREMIUM",
    "BILLING_PAYMENT",
    "FAMILY_DUO",
    "PLAYBACK",
    "APP_TECHNICAL",
    "DEVICE_PLATFORM",
    "PLAYLIST_LIBRARY",
    "CONTENT_AVAILABILITY",
    "FEATURE_REQUEST",
    "OTHER",
}

VALID_RESOLUTION = {
    "RESOLVED",
    "UNRESOLVED",
    "PARTIALLY_RESOLVED",
    "INFORMATION_ONLY",
    "UNCLEAR",
}

VALID_ESCALATION = {
    "YES",
    "NO",
    "UNCLEAR",
}

df = pd.read_csv(INPUT)

print("=" * 75)
print("HIVER GOLDEN SET VALIDATION")
print("=" * 75)

print(f"\nTotal rows: {len(df)}")

# ---------------------------------------------------------
# 1. ID validation
# ---------------------------------------------------------

expected_ids = set(range(1, 201))
actual_ids = set(df["annotation_id"].dropna().astype(int))

missing_ids = sorted(expected_ids - actual_ids)
extra_ids = sorted(actual_ids - expected_ids)
duplicate_ids = df[
    df["annotation_id"].duplicated(keep=False)
]["annotation_id"].unique().tolist()

print("\n--- ID CHECK ---")
print(f"Expected IDs: 1-200")
print(f"Missing IDs: {missing_ids if missing_ids else 'NONE'}")
print(f"Extra IDs: {extra_ids if extra_ids else 'NONE'}")
print(f"Duplicate IDs: {duplicate_ids if duplicate_ids else 'NONE'}")

# ---------------------------------------------------------
# 2. Blank validation
# ---------------------------------------------------------

label_columns = [
    "intent",
    "resolution_status",
    "requires_escalation",
    "annotation_notes",
]

print("\n--- BLANK LABEL CHECK ---")

for column in label_columns:
    blanks = df[column].isna().sum()
    empty_strings = (
        df[column]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        f"{column}: "
        f"{blanks} NaN, "
        f"{empty_strings} empty"
    )

# ---------------------------------------------------------
# 3. Valid intent check
# ---------------------------------------------------------

print("\n--- INVALID INTENT CHECK ---")

invalid_intents = sorted(
    set(df["intent"].dropna()) - VALID_INTENTS
)

print(
    f"Invalid intents: "
    f"{invalid_intents if invalid_intents else 'NONE'}"
)

# ---------------------------------------------------------
# 4. Valid resolution check
# ---------------------------------------------------------

print("\n--- INVALID RESOLUTION CHECK ---")

invalid_resolution = sorted(
    set(df["resolution_status"].dropna())
    - VALID_RESOLUTION
)

print(
    f"Invalid resolution values: "
    f"{invalid_resolution if invalid_resolution else 'NONE'}"
)

# ---------------------------------------------------------
# 5. Valid escalation check
# ---------------------------------------------------------

print("\n--- INVALID ESCALATION CHECK ---")

invalid_escalation = sorted(
    set(df["requires_escalation"].dropna())
    - VALID_ESCALATION
)

print(
    f"Invalid escalation values: "
    f"{invalid_escalation if invalid_escalation else 'NONE'}"
)

# ---------------------------------------------------------
# 6. Intent distribution
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("INTENT DISTRIBUTION")
print("=" * 75)

print(
    df["intent"]
    .value_counts(dropna=False)
    .to_string()
)

# ---------------------------------------------------------
# 7. Resolution distribution
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("RESOLUTION DISTRIBUTION")
print("=" * 75)

print(
    df["resolution_status"]
    .value_counts(dropna=False)
    .to_string()
)

# ---------------------------------------------------------
# 8. Escalation distribution
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("ESCALATION DISTRIBUTION")
print("=" * 75)

print(
    df["requires_escalation"]
    .value_counts(dropna=False)
    .to_string()
)

# ---------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------

total_missing_labels = sum(
    df[column].isna().sum()
    for column in [
        "intent",
        "resolution_status",
        "requires_escalation",
    ]
)

print("\n" + "=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

if (
    len(df) == 200
    and not missing_ids
    and not extra_ids
    and not duplicate_ids
    and total_missing_labels == 0
    and not invalid_intents
    and not invalid_resolution
    and not invalid_escalation
):
    print("STATUS: PASS")
    print("All 200 rows have valid required labels.")
else:
    print("STATUS: REVIEW REQUIRED")

    if missing_ids:
        print(f"- Missing IDs: {missing_ids}")

    if extra_ids:
        print(f"- Extra IDs: {extra_ids}")

    if duplicate_ids:
        print(f"- Duplicate IDs: {duplicate_ids}")

    if total_missing_labels:
        print(
            f"- Missing required labels: "
            f"{total_missing_labels}"
        )

    if invalid_intents:
        print(f"- Invalid intents: {invalid_intents}")

    if invalid_resolution:
        print(
            f"- Invalid resolution values: "
            f"{invalid_resolution}"
        )

    if invalid_escalation:
        print(
            f"- Invalid escalation values: "
            f"{invalid_escalation}"
        )

print("=" * 75)