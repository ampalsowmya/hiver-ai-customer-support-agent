from pathlib import Path

import pandas as pd

from pipeline import run_agent


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
    / "agent_evaluation.csv"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_agent():

    df = pd.read_csv(
        INPUT_PATH
    )

    df = df.dropna(
        subset=[
            "clean_text",
            "intent"
        ]
    )

    results = []

    print("=" * 70)
    print("HIVER END-TO-END AGENT EVALUATION")
    print("=" * 70)

    for index, row in df.iterrows():

        query = str(
            row["clean_text"]
        )

        gold_intent = str(
            row["intent"]
        )

        gold_resolution = str(
            row.get(
                "resolution_status",
                ""
            )
        )

        gold_escalation = str(
            row.get(
                "escalation",
                ""
            )
        )

        agent_result = run_agent(
            query=query,
            top_k=3
        )

        results.append(
            {
                "example_id": index + 1,

                "query": query,

                "gold_intent": gold_intent,

                "predicted_intent": agent_result[
                    "intent"
                ],

                "classifier_confidence": agent_result[
                    "classifier_confidence"
                ],

                "retrieval_similarity": agent_result[
                    "retrieval_similarity"
                ],

                "gold_resolution": gold_resolution,

                "gold_escalation": gold_escalation,

                "agent_action": agent_result[
                    "action"
                ],

                "decision_reason": agent_result[
                    "decision_reason"
                ],

                "response": agent_result[
                    "response"
                ],

                "grounded": agent_result[
                    "grounded"
                ],

                "needs_escalation": agent_result[
                    "needs_escalation"
                ],
            }
        )

        if (index + 1) % 25 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(df)}"
            )

    output = pd.DataFrame(
        results
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Basic diagnostics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Examples evaluated: "
        f"{len(output)}"
    )

    print()

    print("Agent decisions:")

    print(
        output[
            "agent_action"
        ].value_counts()
    )

    print()

    print(
        "Grounded responses:"
    )

    print(
        output[
            "grounded"
        ].value_counts()
    )

    print()

    print(
        "Predicted intents:"
    )

    print(
        output[
            "predicted_intent"
        ].value_counts()
    )

    print()

    print("Saved:")

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("END-TO-END AGENT EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    evaluate_agent()