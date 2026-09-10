from agent.classifier import classify
from agent.retriever import retrieve
from agent.responder import generate_grounded_response
from agent.escalation import escalation_decision


def run_agent(query: str, top_k: int = 3):
    """
    End-to-end support pipeline:
    1. Classify intent
    2. Retrieve historical cases
    3. Evaluate evidence quality
    4. Decide ANSWER / CLARIFY / ESCALATE
    5. Generate grounded response only when evidence is strong
    """

    # ---------------------------------------------------------
    # 1. Intent classification
    # ---------------------------------------------------------
    classification = classify(query)

    intent = classification["intent"]
    confidence = classification["confidence"]
    source = classification.get("source", "ml")

    # ---------------------------------------------------------
    # 2. Historical retrieval
    # ---------------------------------------------------------
    retrieved_cases = retrieve(
        query,
        intent=intent,
        top_k=top_k
    )

    # ---------------------------------------------------------
    # 3. Evidence assessment
    # ---------------------------------------------------------
    if retrieved_cases:
        best_case = retrieved_cases[0]

        retrieval_similarity = best_case.get(
            "similarity", 0.0
        )

        combined_score = best_case.get(
            "combined_score",
            retrieval_similarity
        )

        intent_compatibility = best_case.get(
            "intent_compatibility", 0.0
        )
    else:
        retrieval_similarity = 0.0
        combined_score = 0.0
        intent_compatibility = 0.0

    # ---------------------------------------------------------
    # 4. Safety-critical escalation
    # ---------------------------------------------------------
    if intent == "ACCOUNT_SECURITY":
        action = "ESCALATE"
        decision_reason = (
            "Account security issues require additional "
            "verification or human handling."
        )

    elif intent == "BILLING_PAYMENT":
        action = "ESCALATE"
        decision_reason = (
            "Billing/payment issues may require "
            "account-specific investigation."
        )

    # ---------------------------------------------------------
    # 5. Evidence-aware routing
    # ---------------------------------------------------------
    else:

        # Very weak evidence
        if not retrieved_cases or retrieval_similarity < 0.20:
            action = "CLARIFY"
            decision_reason = (
                "No sufficiently similar historical case "
                "was found."
            )

        # Similar text but poor intent compatibility
        elif (
            retrieval_similarity >= 0.35
            and intent_compatibility < 0.10
        ):
            action = "CLARIFY"
            decision_reason = (
                "A similar historical case was found, but "
                "its intent is not sufficiently compatible "
                "with the predicted issue."
            )

        # Moderate evidence
        elif combined_score < 0.30:
            action = "CLARIFY"
            decision_reason = (
                "Historical evidence is only moderately "
                "relevant, so clarification is safer than "
                "giving a potentially incorrect answer."
            )

        # Strong evidence
        else:
            action = "ANSWER"
            decision_reason = (
                "A sufficiently relevant and intent-compatible "
                "historical case was found."
            )

    # ---------------------------------------------------------
    # 6. Generate response
    # ---------------------------------------------------------
    if action == "ANSWER":
        response_data = generate_grounded_response(
            query,
            intent,
            retrieved_cases
        )

        response = response_data["response"]
        grounded = response_data["grounded"]
        needs_escalation = response_data["needs_escalation"]

    elif action == "CLARIFY":
        response = (
            "Could you provide a little more detail about "
            "the issue, such as the device/platform you are "
            "using and what you expected to happen?"
        )

        grounded = False
        needs_escalation = False

    else:
        response = (
            "This issue may require additional investigation. "
            "Please provide the relevant account, device, or "
            "issue details so the support team can look into it."
        )

        grounded = False
        needs_escalation = True

    # ---------------------------------------------------------
    # 7. Structured result
    # ---------------------------------------------------------
    return {
        "query": query,
        "intent": intent,
        "classifier_confidence": confidence,
        "classifier_source": source,
        "retrieval_similarity": retrieval_similarity,
        "intent_compatibility": intent_compatibility,
        "combined_score": combined_score,
        "action": action,
        "decision_reason": decision_reason,
        "response": response,
        "grounded": grounded,
        "needs_escalation": needs_escalation,
        "retrieved_cases": retrieved_cases,
    }


if __name__ == "__main__":

    test_queries = [
        "My Spotify app keeps crashing on Windows",
        "My playlist disappeared from my library",
        "Spotify should add a feature to show lyrics automatically",
        "I was charged twice for Spotify Premium",
        "Someone hacked my Spotify account",
    ]

    print("=" * 70)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 70)

    for query in test_queries:

        print("\nQuery:", query)

        result = run_agent(query)

        print("Intent:", result["intent"])
        print(
            "Confidence:",
            round(result["classifier_confidence"], 3)
        )
        print(
            "Retrieval similarity:",
            round(result["retrieval_similarity"], 3)
        )
        print(
            "Intent compatibility:",
            round(result["intent_compatibility"], 3)
        )
        print(
            "Combined score:",
            round(result["combined_score"], 3)
        )
        print("Action:", result["action"])
        print("Reason:", result["decision_reason"])
        print("Response:", result["response"])