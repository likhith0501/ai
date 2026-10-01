"""Model evaluation helpers: metrics, confusion matrices, ROC curves and plots."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CYBER_BG = "#0b1120"
PANEL_BG = "#111a2e"
ACCENT = "#22d3ee"
MALICIOUS = "#f43f5e"
NORMAL = "#22c55e"
GRID = "#1e293b"
TEXT = "#cbd5e1"


def _safe_auc(y_true: Sequence[int], y_proba: Optional[np.ndarray], binary: bool) -> float:
    if y_proba is None or len(np.unique(np.asarray(y_true))) < 2:
        return float("nan")
    try:
        if binary and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        scores = []
        for index in range(y_proba.shape[1]):
            positive = (np.asarray(y_true) == index).astype(int)
            if len(np.unique(positive)) < 2:
                continue
            scores.append(float(roc_auc_score(positive, y_proba[:, index])))
        return float(np.mean(scores)) if scores else float("nan")
    except ValueError:
        return float("nan")


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
    binary: bool = True,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute the full metric bundle required by the dashboard."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = list(range(y_proba.shape[1])) if (y_proba is not None and y_proba.ndim == 2) else None

    average = "binary" if binary else "weighted"
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0)),
        "roc_auc": _safe_auc(y_true_arr, y_proba, binary),
        "average_precision": float("nan"),
        "support": int(y_true_arr.size),
    }

    if binary and y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        try:
            metrics["average_precision"] = float(average_precision_score(y_true_arr, y_proba[:, 1]))
        except ValueError:
            metrics["average_precision"] = float("nan")

    labels_for_cm = labels if labels else sorted(np.unique(y_true_arr).tolist())
    matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_for_cm)
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["labels"] = labels_for_cm
    metrics["classification_report"] = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels_for_cm,
        target_names=list(class_names) if class_names and len(class_names) == len(labels_for_cm) else None,
        zero_division=0,
        output_dict=True,
    )
    metrics["roc_curve"] = roc_curve_payload(y_true_arr, y_proba, binary)
    if not binary and y_proba is not None:
        metrics["roc_curves"] = multiclass_roc_payload(y_true_arr, y_proba, class_names)
    return metrics


def roc_curve_payload(y_true: Sequence[int], y_proba: Optional[np.ndarray], binary: bool) -> Optional[Dict[str, List[float]]]:
    """Return a JSON-serialisable binary ROC curve for Chart.js."""
    if y_proba is None or y_proba.ndim != 2 or not binary or y_proba.shape[1] != 2:
        return None
    y_true_arr = np.asarray(y_true)
    try:
        fpr, tpr, _ = roc_curve(y_true_arr, y_proba[:, 1])
    except ValueError:
        return None
    return {
        "fpr": [float(round(v, 6)) for v in fpr],
        "tpr": [float(round(v, 6)) for v in tpr],
    }


def multiclass_roc_payload(y_true: Sequence[int], y_proba: np.ndarray,
                           class_names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """One-vs-rest ROC curves for a multiclass target."""
    curves: List[Dict[str, Any]] = []
    y_true_arr = np.asarray(y_true)
    for index in range(y_proba.shape[1]):
        positive = (y_true_arr == index).astype(int)
        if len(np.unique(positive)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(positive, y_proba[:, index])
            auc = float(roc_auc_score(positive, y_proba[:, index]))
        except ValueError:
            continue
        name = str(class_names[index]) if class_names and index < len(class_names) else str(index)
        curves.append(
            {
                "class": name,
                "auc": round(auc, 6),
                "fpr": [float(round(v, 6)) for v in fpr],
                "tpr": [float(round(v, 6)) for v in tpr],
            }
        )
    return curves


def _finish(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=CYBER_BG, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(path)


def plot_confusion_matrix(matrix: Sequence[Sequence[int]], labels: Sequence[Any], model_name: str, out_dir: str) -> str:
    """Render a confusion matrix heatmap for the model performance page."""
    os.makedirs(out_dir, exist_ok=True)
    matrix_arr = np.asarray(matrix, dtype=float)
    row_totals = matrix_arr.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    normalised = matrix_arr / row_totals

    names = [str(label) for label in labels]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    fig.patch.set_facecolor(CYBER_BG)
    ax.set_facecolor(PANEL_BG)
    sns.heatmap(
        normalised,
        annot=matrix_arr.astype(int),
        fmt="d",
        cmap=sns.dark_palette(ACCENT, as_cmap=True),
        linewidths=0.5,
        linecolor=GRID,
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Normalised"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label", color=TEXT)
    ax.set_ylabel("True label", color=TEXT)
    ax.set_title(f"{model_name} — Confusion Matrix", color=TEXT, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT, labelsize=9)
    for text in ax.texts:
        text.set_color("#0b1120")
    return _finish(fig, os.path.join(out_dir, f"confusion_{model_name.lower().replace(' ', '_')}.png"))


def plot_roc(curve: Optional[Dict[str, List[float]]], auc_value: float, model_name: str, out_dir: str) -> Optional[str]:
    """Render the ROC curve for a binary model."""
    if not curve:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    fig.patch.set_facecolor(CYBER_BG)
    ax.set_facecolor(PANEL_BG)
    ax.plot(curve["fpr"], curve["tpr"], color=ACCENT, linewidth=2, label=f"{model_name} (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", color=TEXT)
    ax.set_ylabel("True Positive Rate", color=TEXT)
    ax.set_title("ROC Curve", color=TEXT, fontsize=12, pad=12)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    legend = ax.legend(facecolor=PANEL_BG, edgecolor=GRID, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT)
    return _finish(fig, os.path.join(out_dir, f"roc_{model_name.lower().replace(' ', '_')}.png"))


def plot_model_comparison(records: List[Dict[str, Any]], out_dir: str) -> str:
    """Grouped bar chart comparing the trained models."""
    os.makedirs(out_dir, exist_ok=True)
    frame = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(CYBER_BG)

    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
               ("f1_score", "F1-score"), ("roc_auc", "ROC-AUC")]
    melted = frame.melt(id_vars=["model_name"], value_vars=[m for m, _ in metrics],
                       var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].map(dict(metrics))
    sns.barplot(data=melted, x="model_name", y="score", hue="metric", ax=axes[0],
                palette=sns.color_palette("husl", len(metrics)))
    axes[0].set_title("Model comparison", color=TEXT)
    axes[0].set_ylabel("Score", color=TEXT)
    axes[0].set_ylim(0, 1.05)

    comparison = frame.melt(id_vars=["model_name"], value_vars=["accuracy", "f1_score", "roc_auc"],
                            var_name="metric", value_name="score")
    sns.barplot(data=comparison, x="score", y="model_name", hue="metric", ax=axes[1],
                palette=[NORMAL, ACCENT, "#a78bfa"])
    axes[1].set_title("Headline metrics", color=TEXT)
    axes[1].set_xlabel("Score", color=TEXT)
    axes[1].set_xlim(0, 1.05)

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
        ax.tick_params(colors=TEXT, labelsize=9)
        if ax.get_legend():
            legend = ax.get_legend()
            legend.get_frame().set_facecolor(PANEL_BG)
            legend.get_frame().set_edgecolor(GRID)
            for text in legend.get_texts():
                text.set_color(TEXT)
    return _finish(fig, os.path.join(out_dir, "model_comparison.png"))


def plot_probability_histogram(probabilities: Sequence[float], out_dir: str) -> Optional[str]:
    """Histogram of threat probabilities from the latest analysis."""
    values = [float(p) for p in probabilities if p is not None and not np.isnan(p)]
    if not values:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(CYBER_BG)
    ax.set_facecolor(PANEL_BG)
    bins = np.linspace(0, 1, 21)
    ax.hist(values, bins=bins, color=MALICIOUS, alpha=0.85, edgecolor=CYBER_BG)
    ax.axvline(0.5, color=ACCENT, linestyle="--", linewidth=1.5, label="Decision threshold")
    ax.set_xlabel("Threat probability", color=TEXT)
    ax.set_ylabel("Records", color=TEXT)
    ax.set_title("Threat probability distribution", color=TEXT, fontsize=12, pad=10)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    legend = ax.legend(facecolor=PANEL_BG, edgecolor=GRID, fontsize=8)
    legend.get_frame().set_facecolor(PANEL_BG)
    legend.get_frame().set_edgecolor(GRID)
    for text in legend.get_texts():
        text.set_color(TEXT)
    return _finish(fig, os.path.join(out_dir, "probability_distribution.png"))


def format_metric(value: Optional[float], digits: int = 4) -> Optional[float]:
    """Round a metric for JSON storage, mapping NaN to ``None``."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return round(numeric, digits)


def sanitise_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of a metric dictionary."""
    safe: Dict[str, Any] = {}
    passthrough = {"confusion_matrix", "classification_report", "roc_curve", "roc_curves", "labels"}
    for key, value in metrics.items():
        if key in passthrough or isinstance(value, str):
            safe[key] = value
        else:
            safe[key] = format_metric(value)
    return safe