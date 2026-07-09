"""
smartml.core.visualizer
=========================
Matplotlib-based plotting. Runs headless (Agg backend) and writes PNGs to
an output directory, returning the paths so the report builder / Assistant
can reference them.

The Agg backend is activated lazily inside each function rather than at
import time, so this module is safe to import in interactive environments
(Jupyter, Qt GUIs) without hijacking the user's chosen backend.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import learning_curve


def _ensure_dir(path: str) -> None:
    """Create output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _use_agg() -> None:
    """Switch to the non-interactive Agg backend if no display is available.

    Called at the top of every plot function rather than at module import
    time so callers in interactive environments keep their own backend.
    """
    try:
        # If a display is available (e.g. Jupyter), keep whatever backend is
        # already active.  If not (CI, headless server), force Agg.
        import matplotlib
        if matplotlib.get_backend().lower() == "agg":
            return
        # Only force Agg when DISPLAY / MPLBACKEND suggest headless.
        if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
            matplotlib.use("Agg")
    except Exception:
        matplotlib.use("Agg")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> str:
    """Generate a correlation heatmap for all numeric columns.

    Args:
        df: Input DataFrame (any columns; non-numeric are ignored).
        out_dir: Directory where the PNG file is written.

    Returns:
        Absolute path of the saved PNG.
    """
    _use_agg()
    _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        # Nothing meaningful to correlate — write a placeholder
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "Not enough numeric columns.", ha="center", va="center")
        ax.axis("off")
        path = os.path.join(out_dir, "correlation_heatmap.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    corr = numeric_df.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n), max(5, 0.5 * n)))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    path = os.path.join(out_dir, "correlation_heatmap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_missing_heatmap(df: pd.DataFrame, out_dir: str) -> str:
    """Generate a heatmap of missing values (white = present, black = missing).

    Args:
        df: Input DataFrame.
        out_dir: Directory where the PNG file is written.

    Returns:
        Absolute path of the saved PNG.
    """
    _use_agg()
    _ensure_dir(out_dir)

    missing_mask = df.isna()
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(df.columns)), max(4, 0.02 * len(df))))
    ax.imshow(missing_mask.values.T, aspect="auto", cmap="binary", interpolation="none")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns, fontsize=7)
    ax.set_xlabel("Row index")
    ax.set_title(f"Missing Values Heatmap  ({missing_mask.sum().sum()} total missing)")
    fig.tight_layout()

    path = os.path.join(out_dir, "missing_heatmap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_importance(importances: pd.Series, out_dir: str, top_n: int = 15) -> str:
    """Generate a horizontal bar chart of feature importances.

    Args:
        importances: Series of importance values indexed by feature name,
            sorted descending.
        out_dir: Directory where the PNG file is written.
        top_n: Maximum number of features to display.

    Returns:
        Absolute path of the saved PNG.
    """
    _use_agg()
    _ensure_dir(out_dir)
    top = importances.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
    ax.barh(top.index, top.values, color="#3b6ea5")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Top {})".format(len(top)))
    fig.tight_layout()

    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    y_true,
    y_pred,
    out_dir: str,
    labels: Optional[list] = None,
) -> str:
    """Generate a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        out_dir: Directory where the PNG file is written.
        labels: Optional ordered list of class labels.

    Returns:
        Absolute path of the saved PNG.
    """
    _use_agg()
    _ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * cm.shape[0]), max(5, 0.6 * cm.shape[0])))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    tick_labels = [str(lbl) for lbl in (labels if labels is not None else range(cm.shape[0]))]
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_residuals(y_true, y_pred, out_dir: str) -> str:
    """Generate a residual plot for regression models.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        out_dir: Directory where the PNG file is written.

    Returns:
        Absolute path of the saved PNG.
    """
    _use_agg()
    _ensure_dir(out_dir)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    residuals = y_true_arr - y_pred_arr

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred_arr, residuals, alpha=0.6, edgecolor="k", linewidth=0.3, color="#3b6ea5")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")

    # Residual distribution
    axes[1].hist(residuals, bins=30, color="#3b6ea5", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    path = os.path.join(out_dir, "residual_plot.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(leaderboard, out_dir: str) -> str:
    """Generate a horizontal bar chart comparing model CV scores.

    Args:
        leaderboard: List of ``LeaderboardEntry`` objects from the trainer.
        out_dir: Directory where the PNG file is written.

    Returns:
        Absolute path of the saved PNG.
    """
    _use_agg()
    _ensure_dir(out_dir)
    names = [e.name for e in leaderboard]
    scores = [e.cv_score for e in leaderboard]
    stds = [e.cv_std for e in leaderboard]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(names))))
    colors = ["#2e7d32" if i == 0 else "#90a4ae" for i in range(len(names))]
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], xerr=stds[::-1], capsize=4)
    ax.set_xlabel("Cross-Validated Score")
    ax.set_title("Model Comparison (CV score ± std)")
    ax.set_xlim(left=max(0, min(scores) - 0.1))
    fig.tight_layout()

    path = os.path.join(out_dir, "model_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_curve(
    y_true,
    y_score,
    out_dir: str,
    pos_label=None,
) -> str:
    """Generate a ROC curve for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Probability scores for the positive class.
        out_dir: Directory where the PNG file is written.
        pos_label: The positive class label. Inferred if None.

    Returns:
        Absolute path of the saved PNG, or empty string if plotting failed
        (e.g. multi-class without one-vs-rest scores).
    """
    _use_agg()
    _ensure_dir(out_dir)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#3b6ea5", linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.set_xlim(left=0.0, right=1.0)
        ax.set_ylim(bottom=0.0, top=1.05)
        fig.tight_layout()

        path = os.path.join(out_dir, "roc_curve.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception:
        return ""


def plot_pr_curve(
    y_true,
    y_score,
    out_dir: str,
    pos_label=None,
) -> str:
    """Generate a Precision-Recall curve for binary classification.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Probability scores for the positive class.
        out_dir: Directory where the PNG file is written.
        pos_label: The positive class label. Inferred if None.

    Returns:
        Absolute path of the saved PNG, or empty string if plotting failed.
    """
    _use_agg()
    _ensure_dir(out_dir)
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
        ap = average_precision_score(y_true, y_score, pos_label=pos_label)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.step(recall, precision, where="post", color="#3b6ea5", linewidth=2,
                label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        ax.set_xlim(left=0.0, right=1.0)
        ax.set_ylim(bottom=0.0, top=1.05)
        fig.tight_layout()

        path = os.path.join(out_dir, "pr_curve.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception:
        return ""


def plot_learning_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str,
    task: str = "classification",
    cv: int = 5,
) -> str:
    """Generate a learning curve (train vs. validation score vs. dataset size).

    Args:
        model: Fitted (or unfitted) scikit-learn compatible estimator.
        X: Feature matrix.
        y: Target vector.
        out_dir: Directory where the PNG file is written.
        task: ``"classification"`` or ``"regression"`` (determines scoring).
        cv: Number of cross-validation folds.

    Returns:
        Absolute path of the saved PNG, or empty string if plotting failed.
    """
    _use_agg()
    _ensure_dir(out_dir)
    try:
        scoring = "f1_weighted" if task == "classification" else "r2"
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1,
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#2e7d32",
                label="Training score")
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1),
                        alpha=0.15, color="#2e7d32")
        ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#3b6ea5",
                label="Validation score")
        ax.fill_between(train_sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1),
                        alpha=0.15, color="#3b6ea5")
        ax.set_xlabel("Training set size")
        ax.set_ylabel(scoring)
        ax.set_title("Learning Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()

        path = os.path.join(out_dir, "learning_curve.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception:
        return ""


def plot_distribution(df: pd.DataFrame, out_dir: str, max_cols: int = 12) -> str:
    """Generate a grid of distribution plots for numeric columns.

    Args:
        df: Input DataFrame.
        out_dir: Directory where the PNG file is written.
        max_cols: Maximum number of columns to plot.

    Returns:
        Absolute path of the saved PNG.
    """
    _use_agg()
    _ensure_dir(out_dir)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No numeric columns to plot.", ha="center", va="center")
        ax.axis("off")
        path = os.path.join(out_dir, "distribution.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    n = len(numeric_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="#3b6ea5", edgecolor="white",
                     linewidth=0.5, density=True)
        axes[i].set_title(col, fontsize=8)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=11, y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, "distribution.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
