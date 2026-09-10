def escalation_decision(
    intent,
    classifier_confidence,
    retrieval_similarity,
    resolution_status=None,
    escalation_label=None,
):
    """
    Decide whether the agent should ANSWER, CLARIFY, or ESCALATE.

    The decision uses both classifier confidence and
    historical retrieval similarity.
    """

    # --------------------------------------------------------
    # 1. Explicit historical escalation
    # --------------------------------------------------------

    if escalation_label == "YES":
        return {
            "action": "ESCALATE",
            "reason": (
                "Historical case indicates human/team "
                "intervention was required."
            ),
        }

    # --------------------------------------------------------
    # 2. Security and billing require additional caution
    # --------------------------------------------------------

    if intent == "ACCOUNT_SECURITY":
        return {
            "action": "ESCALATE",
            "reason": (
                "Account security issues require "
                "additional verification or human handling."
            ),
        }

    if intent == "BILLING_PAYMENT":
        return {
            "action": "ESCALATE",
            "reason": (
                "Billing/payment issues may require "
                "account-specific investigation."
            ),
        }

    # --------------------------------------------------------
    # 3. Extremely weak evidence
    # --------------------------------------------------------

    if (
        classifier_confidence < 0.12
        and retrieval_similarity < 0.20
    ):
        return {
            "action": "CLARIFY",
            "reason": (
                "Both intent confidence and historical "
                "retrieval similarity are very low."
            ),
        }

    # --------------------------------------------------------
    # 4. Strong historical evidence can compensate for
    #    moderate classifier confidence.
    # --------------------------------------------------------

    if retrieval_similarity >= 0.35:
        return {
            "action": "ANSWER",
            "reason": (
                "A sufficiently similar historical case "
                "was found."
            ),
        }

    # --------------------------------------------------------
    # 5. Moderate evidence -> clarify
    # --------------------------------------------------------

    if retrieval_similarity >= 0.20:
        return {
            "action": "CLARIFY",
            "reason": (
                "Historical similarity is moderate; "
                "additional information is needed."
            ),
        }

    # --------------------------------------------------------
    # 6. Weak retrieval
    # --------------------------------------------------------

    return {
        "action": "ESCALATE",
        "reason": (
            "No sufficiently similar historical case "
            "was found."
        ),
    }


if __name__ == "__main__":

    test_cases = [
        {
            "name": "Strong historical case",
            "intent": "APP_TECHNICAL",
            "classifier_confidence": 0.15,
            "retrieval_similarity": 0.40,
        },
        {
            "name": "Moderate historical case",
            "intent": "APP_TECHNICAL",
            "classifier_confidence": 0.15,
            "retrieval_similarity": 0.25,
        },
        {
            "name": "Very weak evidence",
            "intent": "OTHER",
            "classifier_confidence": 0.10,
            "retrieval_similarity": 0.10,
        },
        {
            "name": "Security issue",
            "intent": "ACCOUNT_SECURITY",
            "classifier_confidence": 0.80,
            "retrieval_similarity": 0.60,
        },
        {
            "name": "Billing issue",
            "intent": "BILLING_PAYMENT",
            "classifier_confidence": 0.80,
            "retrieval_similarity": 0.60,
        },
    ]

    print("=" * 70)
    print("HIVER ESCALATION GATE")
    print("=" * 70)

    for case in test_cases:

        result = escalation_decision(
            intent=case["intent"],
            classifier_confidence=case["classifier_confidence"],
            retrieval_similarity=case["retrieval_similarity"],
        )

        print()
        print(case["name"])
        print("Action:", result["action"])
        print("Reason:", result["reason"])