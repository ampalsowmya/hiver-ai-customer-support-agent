from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "outputs" / "reconstructed_conversations.jsonl"
OUTPUT = ROOT / "outputs" / "spotify_reconstructed_conversations.jsonl"


spotify = []

with open(INPUT, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        if str(record.get("brand", "")).lower() == "spotifycares":
            spotify.append(record)


with open(OUTPUT, "w", encoding="utf-8") as f:

    for record in spotify:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


print("=" * 70)
print("SPOTIFY CONVERSATION EXTRACTION")
print("=" * 70)

print(f"SpotifyCares conversations: {len(spotify)}")
print(f"Saved: {OUTPUT}")

if spotify:

    lengths = [
        int(x.get("num_messages", 0))
        for x in spotify
    ]

    print(
        f"Average messages: "
        f"{sum(lengths) / len(lengths):.2f}"
    )

    print(
        f"Maximum messages: "
        f"{max(lengths)}"
    )

print("=" * 70)