import json
import re
import subprocess
from pathlib import Path

import pandas as pd


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path("outputs/evaluation/agent_evaluation.csv")
RETRIEVAL_FILE = Path("outputs/evaluation/retrieval_evaluation.csv")
OUTPUT_FILE = Path("outputs/evaluation/llm_judge_evaluation.csv")

MODEL = "llama3.2:latest"

# Human-vs-LLM agreement will use this deterministic subset.
SAMPLE_SIZE = 30


# ============================================================
# LLM RUBRIC
# ============================================================

RUBRIC = """
Evaluate the customer-support response on five dimensions.

Give each dimension an integer score from 1 to 5.

1. Groundedness:
   Is the response supported by the available historical evidence?

   1 = unsupported or contradicts the evidence
   2 = mostly unsupported
   3 = partially supported
   4 = well supported
   5 = clearly supported by the historical evidence

2. Relevance:
   Does the response address the customer's actual issue?

   1 = irrelevant
   2 = mostly irrelevant
   3 = partially addresses the issue
   4 = addresses the main issue
   5 = directly addresses the customer's issue

3. Actionability:
   Does the response give the customer a useful next step?

   1 = no useful next step
   2 = very limited guidance
   3 = somewhat useful
   4 = useful next step
   5 = clear, specific and useful next step

4. Tone:
   Is the response appropriate, concise, professional and helpful?

   1 = poor or inappropriate
   2 = weak
   3 = acceptable
   4 = good
   5 = excellent

5. Overall:
   Overall quality of the customer-support response.

   1 = poor response
   2 = weak response
   3 = acceptable response
   4 = good response
   5 = excellent response

IMPORTANT:
- Evaluate the actual customer issue, not just keywords.
- Use the historical cases as evidence.
- Do not assume facts that are not present.
- Do not give a high groundedness score merely because the response sounds plausible.
- A request for clarification can be appropriate when the customer message genuinely lacks enough information.
- If the customer already provided enough information, unnecessary clarification should receive a low relevance/actionability score.
- Historical support responses are evidence of how the brand historically responded, not automatically proof that the response is correct.

Return ONLY valid JSON.

Use exactly this structure:

{
  "groundedness": 1,
  "relevance": 1,
  "actionability": 1,
  "tone": 1,
  "overall": 1,
  "reason": "short explanation"
}
"""


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout.strip()


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text: str):
    # Remove markdown code fences if Ollama adds them.
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    # Find the JSON object.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found: {text}")

    candidate = match.group(0)

    # Remove literal control characters that can break JSON.
    candidate = re.sub(r"[\x00-\x1f\x7f]", " ", candidate)

    return json.loads(candidate)


# ============================================================
# RETRIEVAL EVIDENCE
# ============================================================

def build_evidence_lookup(retrieval_df):
    """
    Build:

        query -> top retrieved historical cases

    retrieval_evaluation.csv does not contain example_id,
    so the query is used as the join key.
    """

    evidence_lookup = {}

    for query, group in retrieval_df.groupby("query"):
        cases = []

        # Keep the retrieval ranking order.
        group = group.sort_values("rank")

        for _, row in group.iterrows():
            cases.append(
                {
                    "rank": int(row["rank"]),
                    "historical_customer": str(
                        row["historical_customer"]
                    ),
                    "historical_support": str(
                        row["historical_support"]
                    ),
                    "similarity": float(row["similarity"]),
                    "intent_compatibility": float(
                        row["intent_compatibility"]
                    ),
                    "combined_score": float(
                        row["combined_score"]
                    ),
                }
            )

        evidence_lookup[query] = cases

    return evidence_lookup


def format_evidence(cases):
    """
    Convert retrieved cases into readable evidence for the LLM.
    """

    if not cases:
        return "No historical evidence was retrieved."

    sections = []

    for case in cases:
        sections.append(
            f"""
--- Historical Case {case["rank"]} ---

Historical Customer:
{case["historical_customer"]}

Historical Support Response:
{case["historical_support"]}

Retrieval similarity:
{case["similarity"]:.3f}

Intent compatibility:
{case["intent_compatibility"]:.3f}

Combined retrieval score:
{case["combined_score"]:.3f}
"""
        )

    return "\n".join(sections)


# ============================================================
# PROMPT
# ============================================================

def build_prompt(row, evidence_lookup):

    customer = str(row["query"])
    response = str(row["response"])

    # Match retrieval evidence using the exact customer query.
    cases = evidence_lookup.get(customer, [])

    historical_evidence = format_evidence(cases)

    return f"""
You are evaluating an AI customer-support response.

CUSTOMER MESSAGE:
{customer}

AI RESPONSE:
{response}

AVAILABLE HISTORICAL EVIDENCE:
{historical_evidence}

{RUBRIC}

Be conservative and evidence-based.

Do not assume facts that are not present in:
1. the customer message,
2. the AI response, or
3. the historical evidence.

The goal is to determine whether this response would be trustworthy
as an AI customer-support response.
"""


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    if not RETRIEVAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing retrieval file: {RETRIEVAL_FILE}"
        )

    # --------------------------------------------------------
    # Load evaluation data
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    retrieval_df = pd.read_csv(RETRIEVAL_FILE)

    print(f"Agent evaluation rows: {len(df)}")
    print(f"Retrieval evaluation rows: {len(retrieval_df)}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_agent_columns = {
        "example_id",
        "query",
        "response",
    }

    required_retrieval_columns = {
        "query",
        "rank",
        "historical_customer",
        "historical_support",
        "similarity",
        "intent_compatibility",
        "combined_score",
    }

    missing_agent = required_agent_columns - set(df.columns)

    missing_retrieval = (
        required_retrieval_columns
        - set(retrieval_df.columns)
    )

    if missing_agent:
        raise ValueError(
            f"Missing columns in agent evaluation: {missing_agent}"
        )

    if missing_retrieval:
        raise ValueError(
            f"Missing columns in retrieval evaluation: "
            f"{missing_retrieval}"
        )

    # --------------------------------------------------------
    # Build retrieval evidence lookup
    # --------------------------------------------------------

    evidence_lookup = build_evidence_lookup(
        retrieval_df
    )

    matched = 0

    for query in df["query"]:
        if query in evidence_lookup:
            matched += 1

    print(f"Agent rows with retrieval evidence: {matched}/{len(df)}")

    # --------------------------------------------------------
    # Deterministic sample
    # --------------------------------------------------------

    sample = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=42,
    ).copy()

    print(
        f"LLM judge sample size: {len(sample)}"
    )

    # --------------------------------------------------------
    # Run LLM judge
    # --------------------------------------------------------

    results = []

    for i, (_, row) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        print(
            f"Judging {i}/{len(sample)}..."
        )

        try:

            prompt = build_prompt(
                row,
                evidence_lookup,
            )

            raw = call_ollama(prompt)

            judgment = parse_json(raw)

            results.append(
                {
                    "example_id": row["example_id"],
                    "query": row["query"],
                    "response": row["response"],
                    "groundedness": judgment.get(
                        "groundedness"
                    ),
                    "relevance": judgment.get(
                        "relevance"
                    ),
                    "actionability": judgment.get(
                        "actionability"
                    ),
                    "tone": judgment.get(
                        "tone"
                    ),
                    "overall": judgment.get(
                        "overall"
                    ),
                    "reason": judgment.get(
                        "reason",
                        "",
                    ),
                }
            )

        except Exception as exc:

            print(
                f"  Judge error: {exc}"
            )

            results.append(
                {
                    "example_id": row["example_id"],
                    "query": row["query"],
                    "response": row["response"],
                    "groundedness": None,
                    "relevance": None,
                    "actionability": None,
                    "tone": None,
                    "overall": None,
                    "reason": f"Judge error: {exc}",
                }
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LLM JUDGE COMPLETE")
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Examples evaluated: {len(output)}"
    )

    valid = output["overall"].notna().sum()

    print(
        f"Valid LLM judgments: {valid}"
    )

    if valid:

        print()
        print("Mean scores:")

        for column in [
            "groundedness",
            "relevance",
            "actionability",
            "tone",
            "overall",
        ]:

            print(
                f"{column}: "
                f"{output[column].mean():.2f}"
            )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()