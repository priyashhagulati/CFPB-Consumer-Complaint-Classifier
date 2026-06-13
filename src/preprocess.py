"""Text preprocessing pipeline: clean → tokenize → stop-word removal → lemmatise."""
import re
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK assets once at import time
for _res in ("punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_res, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# CFPB redacts personal info with repeated X characters — strip them
_REDACTED = re.compile(r"\bx{2,}\b", re.IGNORECASE)
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs, redacted tokens, punctuation, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = _REDACTED.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def lemmatize(tokens: List[str]) -> List[str]:
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_text(text: str) -> str:
    """Full pipeline → returns a single cleaned, lemmatised string."""
    return " ".join(lemmatize(remove_stopwords(tokenize(clean_text(text)))))


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "Consumer complaint narrative",
) -> pd.DataFrame:
    """Apply the full pipeline to *text_col* and store result in 'processed_text'."""
    df = df.copy()
    print(f"  Preprocessing {len(df):,} documents…")
    df["processed_text"] = df[text_col].fillna("").apply(preprocess_text)
    return df
