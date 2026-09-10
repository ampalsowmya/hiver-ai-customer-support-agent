from pathlib import Path
import re

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "outputs"
    / "spotify_golden_annotation_context.csv"
)

MODEL_DIR = ROOT / "outputs" / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "intent_classifier.pkl"


# ============================================================
# HIGH-PRECISION INTENT ROUTING
# ============================================================

INTENT_RULES = {

    "ACCOUNT_SECURITY": [
        r"\bhacked\b",
        r"\bhack(ed|ing)?\b",
        r"\bunauthorized\b",
        r"\bcompromised\b",
        r"\bstolen account\b",
        r"\bsomeone (accessed|got into|logged into)\b",
        r"\baccount was accessed\b",
    ],

    "BILLING_PAYMENT": [
        r"\bcharged twice\b",
        r"\bdouble charge\b",
        r"\bcharged again\b",
        r"\bcharged twice\b",
        r"\bwrong charge\b",
        r"\bincorrect charge\b",
        r"\bpayment\b",
        r"\bpaid\b",
        r"\bbilled\b",
        r"\bbilling\b",
        r"\brefund\b",
        r"\btransaction\b",
    ],

    "ACCOUNT_LOGIN": [
        r"\bcan't log in\b",
        r"\bcannot log in\b",
        r"\bcan not log in\b",
        r"\bcan't login\b",
        r"\bcannot login\b",
        r"\bcan't sign in\b",
        r"\bcannot sign in\b",
        r"\blog ?in\b",
        r"\blogin\b",
        r"\bsign ?in\b",
        r"\bpassword\b",
    ],

    "CONTENT_AVAILABILITY": [
        r"\bsong unavailable\b",
        r"\btrack unavailable\b",
        r"\bsong is unavailable\b",
        r"\btrack is unavailable\b",
        r"\bsong missing\b",
        r"\btrack missing\b",
        r"\bsong disappeared\b",
        r"\btrack disappeared\b",
        r"\bcan't find (the )?(song|track)\b",
        r"\bcannot find (the )?(song|track)\b",
        r"\bremoved song\b",
        r"\bremoved track\b",
        r"\bnot available\b",
    ],

    "PLAYLIST_LIBRARY": [
        r"\bplaylist\b",
        r"\bplaylists\b",
        r"\blibrary\b",
        r"\bsaved songs?\b",
        r"\bsaved music\b",
        r"\bmy music disappeared\b",
    ],

    "FEATURE_REQUEST": [
        r"\badd a feature\b",
        r"\bnew feature\b",
        r"\bfeature request\b",
        r"\brequest a feature\b",
        r"\bwould like (a )?feature\b",
        r"\bwould like spotify to\b",
        r"\bspotify should\b",
        r"\bplease add\b",
        r"\bcan you add\b",
        r"\bwould be nice if\b",
        r"\bi wish spotify\b",
    ],

    "SUBSCRIPTION_PREMIUM": [
        r"\bpremium subscription\b",
        r"\bspotify premium\b",
        r"\bupgrade to premium\b",
        r"\bcancel premium\b",
        r"\bpremium plan\b",
        r"\bfree trial\b",
    ],

    "FAMILY_DUO": [
        r"\bfamily plan\b",
        r"\bspotify family\b",
        r"\bduo plan\b",
        r"\bspotify duo\b",
    ],

    "APP_TECHNICAL": [
        r"\bapp keeps crashing\b",
        r"\bapp is crashing\b",
        r"\bapp crashes\b",
        r"\bspotify keeps crashing\b",
        r"\bspotify is crashing\b",
        r"\bcrash(es|ing)?\b",
        r"\bcrashing\b",
        r"\bapp won't open\b",
        r"\bapp will not open\b",
        r"\bapp not opening\b",
        r"\btechnical issue\b",
        r"\bapp error\b",
        r"\berror message\b",
        r"\bbug\b",
    ],

    "PLAYBACK": [
        r"\bsong won't play\b",
        r"\bsong will not play\b",
        r"\btrack won't play\b",
        r"\btrack will not play\b",
        r"\bmusic won't play\b",
        r"\bmusic will not play\b",
        r"\bcan't play\b",
        r"\bcannot play\b",
        r"\bplayback\b",
        r"\bkeeps pausing\b",
        r"\bkeeps skipping\b",
    ],

    "DEVICE_PLATFORM": [
        r"\bandroid\b",
        r"\biphone\b",
        r"\bios\b",
        r"\bwindows\b",
        r"\bmac\b",
        r"\bmacos\b",
        r"\bxbox\b",
        r"\bplaystation\b",
        r"\bps4\b",
        r"\bps5\b",
        r"\bcar\b",
        r"\bdesktop\b",
        r"\bmobile\b",
    ],
}


def rule_based_intent(text):
    """
    Return a high-confidence intent when the message contains
    strong, unambiguous intent signals.

    Returns None when no high-confidence rule matches.
    """

    normalized = str(text).lower().strip()

    # Security should take priority over all other intents.
    for pattern in INTENT_RULES["ACCOUNT_SECURITY"]:
        if re.search(pattern, normalized):
            return {
                "intent": "ACCOUNT_SECURITY",
                "confidence": 0.99,
                "source": "rule"
            }

    # Billing should take priority over subscription classification.
    for pattern in INTENT_RULES["BILLING_PAYMENT"]:
        if re.search(pattern, normalized):
            return {
                "intent": "BILLING_PAYMENT",
                "confidence": 0.99,
                "source": "rule"
            }

    # Login.
    for pattern in INTENT_RULES["ACCOUNT_LOGIN"]:
        if re.search(pattern, normalized):
            return {
                "intent": "ACCOUNT_LOGIN",
                "confidence": 0.99,
                "source": "rule"
            }

    # Content availability.
    for pattern in INTENT_RULES["CONTENT_AVAILABILITY"]:
        if re.search(pattern, normalized):
            return {
                "intent": "CONTENT_AVAILABILITY",
                "confidence": 0.99,
                "source": "rule"
            }

    # Playlist/library.
    for pattern in INTENT_RULES["PLAYLIST_LIBRARY"]:
        if re.search(pattern, normalized):
            return {
                "intent": "PLAYLIST_LIBRARY",
                "confidence": 0.99,
                "source": "rule"
            }

    # Feature requests.
    for pattern in INTENT_RULES["FEATURE_REQUEST"]:
        if re.search(pattern, normalized):
            return {
                "intent": "FEATURE_REQUEST",
                "confidence": 0.99,
                "source": "rule"
            }

    # Family/Duo.
    for pattern in INTENT_RULES["FAMILY_DUO"]:
        if re.search(pattern, normalized):
            return {
                "intent": "FAMILY_DUO",
                "confidence": 0.99,
                "source": "rule"
            }

    # Technical app issues.
    for pattern in INTENT_RULES["APP_TECHNICAL"]:
        if re.search(pattern, normalized):
            return {
                "intent": "APP_TECHNICAL",
                "confidence": 0.99,
                "source": "rule"
            }

    # Playback.
    for pattern in INTENT_RULES["PLAYBACK"]:
        if re.search(pattern, normalized):
            return {
                "intent": "PLAYBACK",
                "confidence": 0.99,
                "source": "rule"
            }

    # Device/platform is intentionally checked last because
    # device words often appear alongside the real issue.
    for pattern in INTENT_RULES["DEVICE_PLATFORM"]:
        if re.search(pattern, normalized):
            return {
                "intent": "DEVICE_PLATFORM",
                "confidence": 0.99,
                "source": "rule"
            }

    return None


# ============================================================
# INTENT CLASSIFIER
# ============================================================

class IntentClassifier:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=50000
        )

        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )

        self.is_fitted = False

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    def train(self):

        df = pd.read_csv(DATA_PATH)

        df = df.dropna(
            subset=["clean_text", "intent"]
        )

        X = df["clean_text"].astype(str)
        y = df["intent"].astype(str)

        # FAMILY_DUO has only one example.
        # Logistic Regression needs at least two examples
        # for a class.
        counts = y.value_counts()

        valid_classes = counts[
            counts >= 2
        ].index

        mask = y.isin(valid_classes)

        X = X[mask]
        y = y[mask]

        # Convert text into TF-IDF features.
        X_tfidf = self.vectorizer.fit_transform(X)

        # Train classifier.
        self.model.fit(
            X_tfidf,
            y
        )

        self.is_fitted = True

        # Save trained artifacts.
        joblib.dump(
            self.vectorizer,
            VECTORIZER_PATH
        )

        joblib.dump(
            self.model,
            MODEL_PATH
        )

        return {
            "training_examples": len(X),
            "classes": sorted(
                y.unique().tolist()
            )
        }

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def load(self):

        self.vectorizer = joblib.load(
            VECTORIZER_PATH
        )

        self.model = joblib.load(
            MODEL_PATH
        )

        self.is_fitted = True

    # --------------------------------------------------------
    # PREDICT — PURE ML
    # --------------------------------------------------------

    def predict(self, text):

        if not self.is_fitted:

            raise RuntimeError(
                "Classifier is not trained or loaded."
            )

        text = str(text)

        X = self.vectorizer.transform(
            [text]
        )

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        best_index = probabilities.argmax()

        intent = self.model.classes_[
            best_index
        ]

        confidence = float(
            probabilities[best_index]
        )

        return {
            "intent": intent,
            "confidence": confidence,
            "source": "ml"
        }


# ============================================================
# PUBLIC FUNCTION USED BY pipeline.py
# ============================================================

def classify(text):

    """
    Hybrid intent classification.

    High-confidence deterministic rules handle obvious
    support intents. The TF-IDF Logistic Regression model
    handles ambiguous cases.
    """

    # --------------------------------------------------------
    # STEP 1 — HIGH-PRECISION RULES
    # --------------------------------------------------------

    rule_result = rule_based_intent(text)

    if rule_result is not None:
        return rule_result

    # --------------------------------------------------------
    # STEP 2 — ML FALLBACK
    # --------------------------------------------------------

    classifier = IntentClassifier()

    classifier.load()

    return classifier.predict(text)


# ============================================================
# TRAINING / MANUAL TEST
# ============================================================

if __name__ == "__main__":

    classifier = IntentClassifier()

    result = classifier.train()

    print("=" * 70)
    print("HIVER HYBRID INTENT CLASSIFIER")
    print("=" * 70)

    print(
        f"Training examples: "
        f"{result['training_examples']}"
    )

    print(
        f"ML classes: "
        f"{len(result['classes'])}"
    )

    print()

    print("ML Classes:")

    for intent in result["classes"]:
        print(f"  - {intent}")

    print()

    print("Model saved:")

    print(
        VECTORIZER_PATH
    )

    print(
        MODEL_PATH
    )

    print()

    # Test examples.
    examples = [

        "Spotify keeps crashing when I open the app",

        "I can't log into my Spotify account",

        "Why is this song unavailable?",

        "I was charged twice for Premium",

        "My playlist disappeared from my library",

        "Spotify should add a feature to show lyrics automatically",

        "Someone hacked my Spotify account",

        "My music won't play on my phone",

    ]

    print("=" * 70)
    print("TEST PREDICTIONS")
    print("=" * 70)

    for text in examples:

        prediction = classify(text)

        print()

        print(
            f"Query: {text}"
        )

        print(
            f"Intent: "
            f"{prediction['intent']}"
        )

        print(
            f"Confidence: "
            f"{prediction['confidence']:.3f}"
        )

        print(
            f"Source: "
            f"{prediction['source']}"
        )

    print()
    print("=" * 70)
    print("CLASSIFIER TEST COMPLETE")
    print("=" * 70)