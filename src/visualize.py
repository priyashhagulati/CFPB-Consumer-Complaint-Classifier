"""Visualisation helpers — all figures saved to outputs/figures/."""
from pathlib import Path
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DIR = Path("outputs/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 7
PALETTE = sns.color_palette("tab10", TOP_N)


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_product_distribution(df: pd.DataFrame, product_col: str = "product_simplified") -> None:
    counts = df[product_col].value_counts().head(TOP_N)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=PALETTE)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of Complaints")
    ax.set_title("CFPB Consumer Complaints — Distribution by Product Category",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, counts.max() * 1.12)
    fig.tight_layout()
    _save(fig, "product_distribution.png")


def plot_complaint_trends(
    df: pd.DataFrame,
    product_col: str = "product_simplified",
    date_col: str = "Date received",
) -> None:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, product_col])
    df[date_col] = df[date_col].dt.tz_localize(None) if df[date_col].dt.tz is not None else df[date_col]
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    top_products = df[product_col].value_counts().head(TOP_N).index.tolist()
    monthly = (
        df[df[product_col].isin(top_products)]
        .groupby(["month", product_col])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, product in enumerate(top_products):
        sub = monthly[monthly[product_col] == product].sort_values("month")
        ax.plot(sub["month"], sub["count"], label=product, color=PALETTE[i], linewidth=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Complaint Count")
    ax.set_title("Monthly CFPB Complaint Trends by Product Category",
                 fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, "complaint_trends.png")


def plot_model_comparison(keyword_acc: float, tfidf_acc: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Keyword-Based\nClassifier", "TF-IDF +\nLogistic Regression"]
    accs = [keyword_acc * 100, tfidf_acc * 100]
    colors = ["#5599cc", "#ee7722"]
    bars = ax.bar(labels, accs, color=colors, width=0.45, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Classifier Accuracy Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    title: str = "Confusion Matrix",
    filename: str = "confusion_matrix.png",
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
