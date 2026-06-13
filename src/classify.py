"""
Two complaint classifiers:
  KeywordClassifier  – rule-based, zero training needed
  TfidfLRClassifier  – TF-IDF vectorisation + Logistic Regression
"""
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Keyword lexicon ──────────────────────────────────────────────────────────
PRODUCT_KEYWORDS: Dict[str, List[str]] = {
    "Credit reporting": [
        "credit report", "credit score", "bureau", "experian", "equifax",
        "transunion", "dispute", "inaccurate", "error", "hard inquiry",
        "soft inquiry", "remove", "record", "history",
    ],
    "Debt collection": [
        "debt", "collector", "collection agency", "owe", "harass", "contact",
        "garnish", "wage", "lawsuit", "cease", "desist", "validation",
        "verify debt", "third party",
    ],
    "Mortgage": [
        "mortgage", "foreclosure", "escrow", "servicer", "refinance",
        "modification", "lien", "deed", "interest rate", "forbearance",
        "loan modification", "monthly payment", "property",
    ],
    "Credit card": [
        "credit card", "card", "balance", "statement", "charge", "interest",
        "annual fee", "rewards", "credit limit", "chargeback", "billing",
        "minimum payment",
    ],
    "Checking or savings account": [
        "checking", "savings", "bank account", "overdraft", "deposit",
        "withdrawal", "transfer", "atm", "direct deposit", "branch",
        "routing number", "wire",
    ],
    "Student loan": [
        "student loan", "student debt", "tuition", "servicer", "repayment",
        "forgiveness", "deferment", "forbearance", "income driven",
        "pell grant", "fafsa", "navient", "sallie mae",
    ],
    "Personal loan": [
        "personal loan", "installment loan", "auto loan", "vehicle loan",
        "payday loan", "title loan", "interest rate", "principal",
        "lender", "apr",
    ],
}


def simplify_product(product: str) -> str:
    """Map verbose CFPB product names to the 7 categories above."""
    p = product.lower()
    if "credit reporting" in p or "credit repair" in p:
        return "Credit reporting"
    if "debt" in p:
        return "Debt collection"
    if "mortgage" in p:
        return "Mortgage"
    if "credit card" in p or "prepaid card" in p:
        return "Credit card"
    if "checking" in p or "savings" in p:
        return "Checking or savings account"
    if "student loan" in p:
        return "Student loan"
    if "personal loan" in p or "vehicle loan" in p or "payday" in p:
        return "Personal loan"
    return "Other"


# ── Keyword classifier ────────────────────────────────────────────────────────
class KeywordClassifier:
    """Score each complaint against per-category keyword lists."""

    def __init__(self, keyword_dict: Dict[str, List[str]] = PRODUCT_KEYWORDS):
        self.keyword_dict = keyword_dict

    def _score(self, text: str) -> str:
        text_l = text.lower()
        scores = {cat: sum(kw in text_l for kw in kws)
                  for cat, kws in self.keyword_dict.items()}
        best, best_score = max(scores.items(), key=lambda x: x[1])
        return best if best_score > 0 else "Other"

    def predict(self, texts: List[str]) -> List[str]:
        return [self._score(t) for t in texts]

    def evaluate(self, texts: List[str], y_true: List[str]) -> dict:
        preds = self.predict(texts)
        acc = accuracy_score(y_true, preds)
        print(f"Keyword Classifier  accuracy: {acc:.4f}\n")
        print(classification_report(y_true, preds, zero_division=0))
        return {
            "accuracy": acc,
            "report": classification_report(y_true, preds, output_dict=True, zero_division=0),
            "predictions": preds,
        }


# ── TF-IDF + Logistic Regression ─────────────────────────────────────────────
class TfidfLRClassifier:
    """Unigram + bigram TF-IDF features fed into multinomial Logistic Regression."""

    def __init__(self, max_features: int = 15_000, C: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(C=C, max_iter=1_000, random_state=42)
        self.label_enc = LabelEncoder()
        self._fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfLRClassifier":
        X = self.vectorizer.fit_transform(texts)
        y = self.label_enc.fit_transform(labels)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, texts: List[str]) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = self.vectorizer.transform(texts)
        return self.label_enc.inverse_transform(self.model.predict(X)).tolist()

    def evaluate(self, texts: List[str], y_true: List[str]) -> dict:
        preds = self.predict(texts)
        acc = accuracy_score(y_true, preds)
        print(f"TF-IDF + LR         accuracy: {acc:.4f}\n")
        print(classification_report(y_true, preds, zero_division=0))
        return {
            "accuracy": acc,
            "report": classification_report(y_true, preds, output_dict=True, zero_division=0),
            "predictions": preds,
        }

    def top_features(self, category: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return the *n* highest-weight TF-IDF features for *category*."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        idx = list(self.label_enc.classes_).index(category)
        coefs = self.model.coef_[idx]
        names = self.vectorizer.get_feature_names_out()
        top = np.argsort(coefs)[-n:][::-1]
        return [(names[i], float(coefs[i])) for i in top]
