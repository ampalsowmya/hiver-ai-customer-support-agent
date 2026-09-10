import re


def clean_response(text):
    """
    Clean historical support text before using it as context.
    """
    if not text:
        return ""

    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\d+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def generate_grounded_response(query, intent, retrieved_cases):
    """
    Generate a safe response using historical support cases.

    This prototype intentionally avoids inventing facts.
    It either:
    1. Uses a relevant historical support response, or
    2. Asks for clarification when retrieval confidence is weak.
    """

    if not retrieved_cases:
        return {
            "response": (
                "I’m sorry you’re having trouble with this. "
                "Could you share a few more details so we can better understand the issue?"
            ),
            "grounded": False,
            "needs_escalation": True,
        }

    best_case = retrieved_cases[0]
    similarity = float(best_case.get("similarity", 0))

    support_response = clean_response(
        best_case.get("support_response", "")
    )

    # Very low-confidence retrieval:
    # Do NOT pretend we know the solution.
    if similarity < 0.20:
        return {
            "response": (
                "Sorry you’re experiencing this. "
                "Could you provide a little more detail about the issue, "
                "including the device or platform you’re using?"
            ),
            "grounded": False,
            "needs_escalation": True,
        }

    # Moderate-confidence retrieval:
    # Use historical support guidance but avoid pretending it is exact.
    if similarity < 0.35:
        if support_response:
            return {
                "response": (
                    "Sorry you’re running into this. "
                    "A similar issue was previously handled by checking "
                    "the account or app details first. "
                    "Could you share the relevant details so we can narrow this down?"
                ),
                "grounded": True,
                "needs_escalation": False,
            }

        return {
            "response": (
                "Sorry you’re experiencing this. "
                "Could you provide more details so we can investigate?"
            ),
            "grounded": False,
            "needs_escalation": True,
        }

    # Higher-confidence retrieval.
    if support_response:
        return {
            "response": support_response,
            "grounded": True,
            "needs_escalation": False,
        }

    return {
        "response": (
            "Sorry you’re experiencing this. "
            "Could you provide more details so we can investigate?"
        ),
        "grounded": False,
        "needs_escalation": True,
    }


if __name__ == "__main__":

    sample_cases = [
        {
            "customer_case": "My Spotify app keeps crashing",
            "support_response": (
                "Could you let us know what device, operating system, "
                "and Spotify version you're using?"
            ),
            "similarity": 0.31,
        }
    ]

    result = generate_grounded_response(
        query="Spotify keeps crashing when I open the app",
        intent="APP_TECHNICAL",
        retrieved_cases=sample_cases,
    )

    print("=" * 70)
    print("HIVER GROUNDED RESPONSE ENGINE")
    print("=" * 70)
    print()
    print("Response:")
    print(result["response"])
    print()
    print("Grounded:", result["grounded"])
    print("Needs escalation:", result["needs_escalation"])