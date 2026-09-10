from pathlib import Path
import pandas as pd

from retriever import retrieve


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "outputs"
    / "spotify_golden_annotation_context.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "retrieval_evaluation.csv"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_retrieval():

    df = pd.read_csv(INPUT_PATH)

    df = df.dropna(
        subset=[
            "clean_text",
            "intent"
        ]
    )

    results = []

    print("=" * 70)
    print("HIVER RETRIEVAL EVALUATION")
    print("=" * 70)

    for _, row in df.iterrows():

        query = str(row["clean_text"])
        intent = str(row["intent"])

        retrieved = retrieve(
            query=query,
            intent=intent,
            top_k=3
        )

        for rank, result in enumerate(
            retrieved,
            start=1
        ):

            results.append(
                {
                    "query": query,
                    "gold_intent": intent,
                    "rank": rank,
                    "historical_customer": result[
                        "customer_case"
                    ],
                    "historical_support": result[
                        "support_response"
                    ],
                    "similarity": result[
                        "similarity"
                    ],
                    "intent_compatibility": result[
                        "intent_compatibility"
                    ],
                    "combined_score": result[
                        "combined_score"
                    ],
                }
            )

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        f"Golden queries evaluated: {len(df)}"
    )

    print(
        f"Retrieved records: {len(output)}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("RETRIEVAL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_retrieval()