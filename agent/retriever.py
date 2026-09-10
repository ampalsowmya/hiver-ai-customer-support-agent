from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

VECTORIZER_PATH = (
    ROOT
    / "outputs"
    / "model"
    / "tfidf_vectorizer.pkl"
)

RETRIEVAL_DATA_PATH = (
    ROOT
    / "outputs"
    / "spotify_retrieval_v2_predictions.csv"
)


# ============================================================
# INTENT KEYWORDS
# ============================================================

INTENT_KEYWORDS = {

    "ACCOUNT_LOGIN": [
        "login",
        "log in",
        "sign in",
        "password",
        "cannot access",
        "can't access",
        "account",
    ],

    "ACCOUNT_SECURITY": [
        "hacked",
        "hack",
        "security",
        "stolen",
        "unauthorized",
        "someone accessed",
        "suspicious",
    ],

    "BILLING_PAYMENT": [
        "charged twice",
        "charged",
        "charge",
        "payment",
        "refund",
        "billing",
        "money",
        "paid",
    ],

    "SUBSCRIPTION_PREMIUM": [
        "premium",
        "subscription",
        "plan",
        "upgrade",
        "downgrade",
        "free trial",
    ],

    "APP_TECHNICAL": [
        "crash",
        "crashing",
        "bug",
        "error",
        "freeze",
        "freezing",
        "not working",
        "app",
    ],

    "DEVICE_PLATFORM": [
        "iphone",
        "ipad",
        "android",
        "windows",
        "mac",
        "computer",
        "phone",
        "device",
    ],

    "PLAYBACK": [
        "play",
        "playing",
        "playback",
        "pause",
        "stopped",
        "skip",
        "audio",
        "sound",
    ],

    "PLAYLIST_LIBRARY": [
        "playlist",
        "library",
        "saved",
        "album",
        "songs",
        "tracks",
    ],

    "CONTENT_AVAILABILITY": [
        "unavailable",
        "missing",
        "removed",
        "greyed",
        "not available",
        "song unavailable",
        "track unavailable",
    ],

    "FEATURE_REQUEST": [
        "feature",
        "suggest",
        "request",
        "add",
        "option",
        "would be nice",
        "could you add",
    ],

    "FAMILY_DUO": [
        "family",
        "duo",
        "kids",
        "family plan",
    ],

    "OTHER": [],
}


def keyword_score(text, intent):
    """
    Calculate a lightweight intent compatibility score.
    """

    text = str(text).lower()

    keywords = INTENT_KEYWORDS.get(
        intent,
        []
    )

    if not keywords:
        return 0.0

    matches = sum(
        1
        for keyword in keywords
        if keyword in text
    )

    return matches / len(keywords)


# ============================================================
# HISTORICAL RETRIEVER
# ============================================================

class HistoricalRetriever:

    def __init__(self):

        self.vectorizer = None
        self.cases = None
        self.case_matrix = None
        self.is_loaded = False

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load(self):

        self.vectorizer = joblib.load(
            VECTORIZER_PATH
        )

        df = pd.read_csv(
            RETRIEVAL_DATA_PATH
        )

        df = df.dropna(
            subset=[
                "historical_customer",
                "historical_support"
            ]
        )

        df = df.drop_duplicates(
            subset=[
                "historical_customer"
            ]
        )

        self.cases = df[
            [
                "historical_customer",
                "historical_support"
            ]
        ].reset_index(drop=True)

        self.case_matrix = self.vectorizer.transform(
            self.cases[
                "historical_customer"
            ].astype(str)
        )

        self.is_loaded = True

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    def retrieve(
        self,
        query,
        intent=None,
        top_k=3
    ):

        if not self.is_loaded:
            self.load()

        query = str(query)

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.case_matrix
        )[0]

        results = []

        for index, similarity in enumerate(
            similarities
        ):

            customer_case = str(
                self.cases.iloc[index][
                    "historical_customer"
                ]
            )

            support_response = str(
                self.cases.iloc[index][
                    "historical_support"
                ]
            )

            # ------------------------------------------------
            # Intent compatibility
            # ------------------------------------------------

            compatibility = 0.0

            if intent:
                compatibility = keyword_score(
                    customer_case,
                    intent
                )

            # ------------------------------------------------
            # Combined score
            # ------------------------------------------------

            combined_score = (
                0.75 * float(similarity)
                +
                0.25 * compatibility
            )

            results.append(
                {
                    "customer_case": customer_case,
                    "support_response": support_response,
                    "similarity": float(similarity),
                    "intent_compatibility": float(
                        compatibility
                    ),
                    "combined_score": float(
                        combined_score
                    ),
                }
            )

        # Sort using combined score
        results.sort(
            key=lambda x: x["combined_score"],
            reverse=True
        )

        return results[:top_k]


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def retrieve(
    query,
    intent=None,
    top_k=3
):

    retriever = HistoricalRetriever()

    return retriever.retrieve(
        query=query,
        intent=intent,
        top_k=top_k
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retriever = HistoricalRetriever()

    retriever.load()

    test_queries = [
        (
            "Spotify keeps crashing when I open the app",
            "APP_TECHNICAL"
        ),

        (
            "I cannot log into my Spotify account",
            "ACCOUNT_LOGIN"
        ),

        (
            "Why is this song unavailable?",
            "CONTENT_AVAILABILITY"
        ),

        (
            "I was charged twice for Premium",
            "BILLING_PAYMENT"
        ),
    ]

    print("=" * 70)
    print("HIVER INTENT-AWARE HISTORICAL RETRIEVER")
    print("=" * 70)

    for query, intent in test_queries:

        print()
        print("-" * 70)
        print("QUERY:", query)
        print("INTENT:", intent)
        print("-" * 70)

        results = retriever.retrieve(
            query=query,
            intent=intent,
            top_k=3
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"RESULT {i} | "
                f"similarity="
                f"{result['similarity']:.4f} | "
                f"intent_match="
                f"{result['intent_compatibility']:.4f} | "
                f"combined="
                f"{result['combined_score']:.4f}"
            )

            print(
                "Historical customer:"
            )

            print(
                result["customer_case"]
            )

            print(
                "Historical support:"
            )

            print(
                result["support_response"]
            )

    print()
    print("=" * 70)
    print("INTENT-AWARE RETRIEVER TEST COMPLETE")
    print("=" * 70)