#!/usr/bin/env python3
"""
CFPB Consumer Complaint Classifier
====================================
Runs the full pipeline:
  1. Download / load data (300 K rows by default)
  2. EDA visualisations on the full dataset
  3. Filter to records that include a consumer narrative
  4. Text preprocessing (clean → tokenise → stopwords → lemmatise)
  5. Keyword-based classification (zero-shot)
  6. TF-IDF + Logistic Regression classification
  7. Model comparison and confusion matrices → outputs/figures/

Usage
-----
  python main.py                             # stream 300 K rows (≈30 MB download)
  python main.py --sample 50000             # cap rows for faster iteration
  python main.py --bulk                     # download full ~200 MB dataset
  python main.py --data-path data/complaints_300k.csv
"""
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.classify import KeywordClassifier, TfidfLRClassifier, simplify_product
from src.download_data import download_bulk, load_data, stream_bulk_nrows
from src.preprocess import preprocess_dataframe
from src.visualize import (
    plot_complaint_trends,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_product_distribution,
)

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

TEXT_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"


def run_pipeline(data_path: str | None, sample: int | None, bulk: bool) -> None:
    # ── 1. Download / load ─────────────────────────────────────────────────
    if data_path is None:
        if bulk:
            data_path = download_bulk()
        else:
            data_path = stream_bulk_nrows(n_rows=300_000)
    df_all = load_data(data_path, sample=sample)
    print(f"\nTotal complaints loaded: {len(df_all):,}")

    # ── 2. Simplify product labels ─────────────────────────────────────────
    df_all["product_simplified"] = df_all[PRODUCT_COL].apply(simplify_product)
    top7 = df_all["product_simplified"].value_counts().head(7).index.tolist()
    df_top = df_all[df_all["product_simplified"].isin(top7)].copy()

    # ── 3. EDA visualisations (full dataset, no narrative filter) ──────────
    print("\nGenerating EDA visualisations on full dataset…")
    plot_product_distribution(df_top)
    plot_complaint_trends(df_top)

    # ── 4. Filter to narrated records ──────────────────────────────────────
    df = df_top.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    df = df[df[TEXT_COL].str.strip() != ""].reset_index(drop=True)
    print(f"\nRecords with consumer narrative: {len(df):,}")
    print(df["product_simplified"].value_counts().to_string(), "\n")

    if len(df) < 50:
        raise RuntimeError(
            "Too few narrated records to build a classifier.\n"
            "Try --bulk to download the full dataset (30 % narrative rate)."
        )

    # ── 5. Text preprocessing ──────────────────────────────────────────────
    print("Preprocessing text…")
    df = preprocess_dataframe(df, text_col=TEXT_COL)

    # ── 6. Train / test split ──────────────────────────────────────────────
    X_proc = df["processed_text"].tolist()
    X_raw = df[TEXT_COL].tolist()
    y = df["product_simplified"].tolist()
    labels = sorted(set(y))

    X_tr, X_te, Xr_tr, Xr_te, y_tr, y_te = train_test_split(
        X_proc, X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}\n")

    # ── 7. Keyword classifier ──────────────────────────────────────────────
    print("=" * 55)
    print("Keyword-Based Classifier (zero-shot, no training)")
    print("=" * 55)
    kw_clf = KeywordClassifier()
    kw_res = kw_clf.evaluate(Xr_te, y_te)

    # ── 8. TF-IDF + LR ────────────────────────────────────────────────────
    print("=" * 55)
    print("TF-IDF + Logistic Regression")
    print("=" * 55)
    tfidf_clf = TfidfLRClassifier()
    tfidf_clf.fit(X_tr, y_tr)
    tfidf_res = tfidf_clf.evaluate(X_te, y_te)

    print("\nTop discriminative TF-IDF features per category:")
    for cat in labels:
        try:
            feats = tfidf_clf.top_features(cat, n=5)
            print(f"  {cat:35s}: {', '.join(f for f, _ in feats)}")
        except ValueError:
            pass

    # ── 9. Result visualisations ───────────────────────────────────────────
    print("\nGenerating result plots…")
    plot_model_comparison(kw_res["accuracy"], tfidf_res["accuracy"])
    plot_confusion_matrix(
        y_te, tfidf_res["predictions"], labels,
        title="TF-IDF + LR Confusion Matrix",
        filename="confusion_matrix_tfidf.png",
    )
    plot_confusion_matrix(
        y_te, kw_res["predictions"], labels,
        title="Keyword Classifier Confusion Matrix",
        filename="confusion_matrix_keyword.png",
    )

    print(f"\nDone!  All figures saved to outputs/figures/")
    print(
        f"\nSummary\n"
        f"  Total records loaded  : {len(df_all):,}\n"
        f"  Records with narrative: {len(df):,}\n"
        f"  Keyword accuracy      : {kw_res['accuracy']:.1%}\n"
        f"  TF-IDF + LR accuracy  : {tfidf_res['accuracy']:.1%}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFPB Complaint Classifier")
    parser.add_argument("--data-path", default=None, help="Path to existing CSV")
    parser.add_argument("--sample", type=int, default=None, help="Row cap")
    parser.add_argument("--bulk", action="store_true", help="Download full dataset")
    args = parser.parse_args()
    run_pipeline(data_path=args.data_path, sample=args.sample, bulk=args.bulk)
