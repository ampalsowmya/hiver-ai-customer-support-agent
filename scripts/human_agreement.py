import pandas as pd
from pathlib import Path


INPUT_FILE = Path("outputs/evaluation/llm_judge_evaluation.csv")
OUTPUT_FILE = Path("outputs/evaluation/human_agreement.csv")


def get_score():
    while True:
        value = input("\nYour HUMAN overall score (1-5): ").strip()

        if value in {"1", "2", "3", "4", "5"}:
            return int(value)

        print("Please enter only 1, 2, 3, 4, or 5.")


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("HUMAN vs LLM AGREEMENT ANNOTATION")
    print("=" * 70)

    print("\nScoring guide:")
    print("1 = Poor")
    print("2 = Weak")
    print("3 = Acceptable")
    print("4 = Good")
    print("5 = Excellent")

    print("\nIMPORTANT:")
    print("- Judge the AI response independently.")
    print("- Do NOT use the LLM overall score when deciding.")
    print("- We will calculate agreement automatically afterward.")
    print()

    human_scores = []

    for i, row in df.iterrows():

        print("\n" + "=" * 70)
        print(f"CASE {i + 1} / {len(df)}")
        print("=" * 70)

        print("\nCUSTOMER MESSAGE:")
        print(str(row["query"]))

        print("\nAI RESPONSE:")
        print(str(row["response"]))

        print("\nLLM JUDGE REASON:")
        print(str(row["reason"]))

        # Deliberately do NOT display the LLM overall score.
        score = get_score()

        human_scores.append(score)

        print(f"Saved human score: {score}")

    df["human_overall"] = human_scores

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Human scores recorded: {len(human_scores)}")


if __name__ == "__main__":
    main()