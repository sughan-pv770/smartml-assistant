"""
smartml.assistant
===================
The conversational layer on top of the SmartML engine.

    import smartml as sml

    assistant = sml.Assistant(df, target="Price")
    assistant.ask("Clean my dataset")
    assistant.ask("Train the best regression model")
    assistant.ask("Why did you choose this algorithm?")
    assistant.ask("Show feature importance")
    assistant.ask("Generate a report")
    assistant.ask("Predict on this CSV file", file_path="new_data.csv")

Design goals
------------
* Every reply is generated from *actual computed results*, never a canned
  string pretending something happened — if you haven't trained a model
  yet and ask "why did you choose X", the assistant tells you that.
* Intent recognition is rule-based (regex over normalised text) rather
  than a black-box classifier, which keeps it fast, dependency-free, and
  itself explainable — you can read exactly which patterns route to which
  handler below.
* The assistant is a thin dispatcher. All real work happens in
  ``smartml.core``, so the exact same functionality is available
  programmatically without going through natural language.
* The assistant uses ``logging.getLogger("smartml")`` instead of
  ``print()`` so it behaves like a proper library.
"""

from __future__ import annotations

import logging
import os
import re
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .core import (
    analyzer,
    preprocessor,
    trainer as trainer_mod,
    explainer,
    visualizer,
    reporter,
    persistence,
    feature_engineer,
    feature_selector,
    clustering,
    anomaly,
    loaders,
)

_log = logging.getLogger("smartml")


# --------------------------------------------------------------------------- #
# Session: holds all state for one Assistant's lifetime
# --------------------------------------------------------------------------- #

@dataclass
class Session:
    """All mutable state for one Assistant conversation.

    Attributes:
        raw_df: The original DataFrame as loaded (never mutated).
        target: Target column name (can be set or changed during the conversation).
        output_dir: Directory where plots, reports, and models are written.
        dataset_report: Latest result from :func:`~smartml.core.analyzer.analyze`.
        cleaning_result: Result from :class:`~smartml.core.preprocessor.Preprocessor`.
        feature_engineering_result: Result from
            :class:`~smartml.core.feature_engineer.FeatureEngineer`.
        feature_selection_result: Result from
            :class:`~smartml.core.feature_selector.FeatureSelector`.
        training_result: Result from :class:`~smartml.core.trainer.Trainer`.
        clustering_result: Result from
            :class:`~smartml.core.clustering.ClusteringTrainer`.
        anomaly_result: Result from :class:`~smartml.core.anomaly.AnomalyDetector`.
        importance_result: Result from
            :func:`~smartml.core.explainer.explain_feature_importance`.
        visualizations: Dict mapping label → file path for generated plots.
        report_paths: Dict mapping format → file path for generated reports.
        loaded_bundle: A :class:`~smartml.core.persistence.SavedBundle` loaded
            from disk.
        history: Chronological list of conversation turns.
    """

    raw_df: pd.DataFrame
    target: Optional[str] = None
    output_dir: str = "smartml_output"

    dataset_report: Optional[analyzer.DatasetReport] = None
    cleaning_result: Optional[preprocessor.CleaningResult] = None
    feature_engineering_result: Optional[feature_engineer.FeatureEngineeringResult] = None
    feature_selection_result: Optional[feature_selector.FeatureSelectionResult] = None
    training_result: Optional[trainer_mod.TrainingResult] = None
    clustering_result: Optional[clustering.ClusteringResult] = None
    anomaly_result: Optional[anomaly.AnomalyResult] = None
    importance_result: Optional[explainer.ImportanceResult] = None
    visualizations: dict = field(default_factory=dict)
    report_paths: dict = field(default_factory=dict)
    loaded_bundle: Optional[persistence.SavedBundle] = None

    history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.target:
            self.dataset_report = analyzer.analyze(self.raw_df, target=self.target)


# --------------------------------------------------------------------------- #
# Intent patterns → handler method names.  Order matters (first match wins).
# --------------------------------------------------------------------------- #

_INTENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bhelp\b|what can you do|list commands",                    "_handle_help"),
    (r"\bset target\b|\btarget is\b|\btarget to\b|^target[=:]",   "_handle_set_target"),
    (r"\bclean\b|\bpreprocess\b|\bfix.*(missing|nulls?)\b",        "_handle_clean"),
    (r"\bengineer\b.*\bfeature|\bfeature.*\bengineer|\bcreate features\b",
                                                                   "_handle_feature_engineer"),
    (r"\bselect\b.*\bfeature|\bfeature.*\bselect|\breduce features\b",
                                                                   "_handle_feature_select"),
    (r"\banaly[sz]e\b|\bsummar[yi]ze?\b|\bdescribe (the )?(data|dataset)\b|\bprofile\b",
                                                                   "_handle_analyze"),
    (r"\bwhy\b.*\b(choose|chose|pick|picked|select|selected)\b"
     r"|\bexplain.*(choice|model|decision)\b|\bjustify\b",         "_handle_explain_choice"),
    (r"\bfeature importan",                                        "_handle_feature_importance"),
    (r"\bcluster",                                                  "_handle_clustering"),
    (r"\banomaly|\bdetect outliers\b|\boutlier detect",             "_handle_anomaly"),
    (r"\b(predict|inference|score)\b",                             "_handle_predict"),
    (r"\bsave\b.*\bmodel\b|\bexport\b.*\bmodel\b",                 "_handle_save_model"),
    (r"\bload\b.*\bmodel\b",                                       "_handle_load_model"),
    (r"\b(visuali[sz]e|plot|chart|show)\b.*"
     r"\b(heatmap|correlation|confusion|residual|comparison|importance|roc|pr curve)\b",
                                                                   "_handle_visualize"),
    (r"\bvisuali[sz]e\b|\bplot\b|\bchart\b",                       "_handle_visualize"),
    (r"\b(generate|build|create|make|export)\b.*\breport\b|^report$",
                                                                   "_handle_report"),
    (r"\btrain\b|\bfind\b.*\b(best )?model\b|\bbuild\b.*\bmodel\b"
     r"|\bbest model\b|\bfit\b.*\bmodel\b",                        "_handle_train"),
]

_FORMAT_PATTERNS = {
    "pdf":      r"\bpdf\b",
    "markdown": r"\bmarkdown\b|\bmd\b",
    "html":     r"\bhtml\b",
    "docx":     r"\bdocx\b|\bword\b",
}

_TASK_OVERRIDE_PATTERNS = {
    "classification": r"\bclassif",
    "regression":     r"\bregress",
}

_PATH_RE = re.compile(r"[\w\-./\\]+\.(csv|xlsx?|json|parquet|feather)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Assistant
# --------------------------------------------------------------------------- #

class Assistant:
    """Conversational front-end over the SmartML engine.

    Args:
        df: The working dataset as a pandas DataFrame.
        target: Target column name. Can also be set later via
            ``assistant.ask("set target to <column>")``.
        output_dir: Directory where generated plots, reports, and saved
            models are written.

    Raises:
        TypeError: If *df* is not a ``pandas.DataFrame``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        output_dir: str = "smartml_output",
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Assistant expects a pandas DataFrame, got {type(df).__name__}."
            )
        self.session = Session(raw_df=df, target=target, output_dir=output_dir)
        self._greet()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def ask(self, text: str, **context: Any) -> str:
        """Send a natural-language command to the assistant.

        *context* lets you pass structured data a sentence cannot easily
        carry, for example::

            assistant.ask("Predict on this CSV", file_path="x.csv")
            assistant.ask("Predict on this data", data=new_df)
            assistant.ask("Load the model", path="model.joblib")
            assistant.ask("Set target", target="Species")

        Args:
            text: Natural-language command.
            **context: Optional structured arguments (see examples above).

        Returns:
            The assistant's reply as a string.
        """
        self.session.history.append({
            "role": "user",
            "text": text,
            "time": dt.datetime.now().isoformat(),
        })
        normalised = text.strip().lower()

        handler_name: Optional[str] = None
        for pattern, name in _INTENT_PATTERNS:
            if re.search(pattern, normalised):
                handler_name = name
                break

        if handler_name is None:
            reply = self._handle_unknown(text)
        else:
            handler = getattr(self, handler_name)
            try:
                reply = handler(text, normalised, **context)
            except Exception as exc:
                _log.exception("Handler %s raised: %s", handler_name, exc)
                reply = (
                    f"That command hit an unexpected error: {exc}\n"
                    "Please check the logs for details."
                )

        self.session.history.append({
            "role": "assistant",
            "text": reply,
            "time": dt.datetime.now().isoformat(),
        })
        _log.info("[assistant] %s", reply[:200])
        return reply

    @property
    def history(self) -> list:
        """Chronological list of all conversation turns."""
        return self.session.history

    # ------------------------------------------------------------------ #
    # Greeting
    # ------------------------------------------------------------------ #

    def _greet(self) -> None:
        n_rows, n_cols = self.session.raw_df.shape
        msg = (
            f"SmartML Assistant ready. Loaded a dataset with {n_rows} rows and "
            f"{n_cols} columns"
        )
        msg += (
            f", target set to '{self.session.target}'."
            if self.session.target
            else (
                " (no target set yet — say e.g. \"set target to Price\", "
                "or pass target= to predict/train)."
            )
        )
        self.session.history.append({
            "role": "assistant",
            "text": msg,
            "time": dt.datetime.now().isoformat(),
        })

    # ------------------------------------------------------------------ #
    # Intent handlers
    # ------------------------------------------------------------------ #

    def _handle_help(self, text: str, norm: str, **ctx: Any) -> str:
        return (
            "I can help you with:\n"
            "  • analyze / summarize   — profile your dataset\n"
            "  • clean / preprocess    — remove duplicates, impute, encode, scale\n"
            "  • engineer features     — create polynomial, date, frequency features\n"
            "  • select features       — remove low-value features\n"
            "  • train (best model)    — compare algorithms and pick the winner\n"
            "  • cluster the data      — unsupervised KMeans / DBSCAN / GMM\n"
            "  • detect anomalies      — flag outlier rows\n"
            "  • why did you choose X  — explain model selection\n"
            "  • feature importance    — top predictors\n"
            "  • visualize / plot      — correlation, confusion, residuals, ROC, …\n"
            "  • generate a report     — HTML or Markdown\n"
            "  • save / load model     — persist and reload\n"
            "  • predict on CSV        — score new data (pass file_path=...)\n"
            "  • set target to <col>   — change the target column\n\n"
            "Tip: Pass file_path='data.csv' or data=df to ask() for predictions."
        )

    def _handle_set_target(self, text: str, norm: str, **ctx: Any) -> str:
        col = ctx.get("target")
        if col is None:
            m = re.search(
                r"target(?:\s+(?:is|to))?[:\s]+([A-Za-z0-9_ ]+)$",
                text.strip(), re.IGNORECASE,
            )
            col = m.group(1).strip() if m else None
        if not col or col not in self.session.raw_df.columns:
            available = ", ".join(self.session.raw_df.columns[:15])
            suffix = "..." if len(self.session.raw_df.columns) > 15 else ""
            return (
                f"I couldn't find that column. "
                f"Available columns include: {available}{suffix}."
            )
        self.session.target = col
        self.session.dataset_report = analyzer.analyze(self.session.raw_df, target=col)
        return f"Target set to '{col}'. {self.session.dataset_report.task_reason}"

    def _handle_analyze(self, text: str, norm: str, **ctx: Any) -> str:
        report = analyzer.analyze(self.session.raw_df, target=self.session.target)
        self.session.dataset_report = report
        lines = [
            f"{report.n_rows} rows × {report.n_cols} columns ({report.memory_mb} MB).",
            f"Numeric: {len(report.numeric_cols)}, "
            f"Categorical: {len(report.categorical_cols)}, "
            f"Datetime: {len(report.datetime_cols)}, "
            f"Boolean: {len(report.boolean_cols)}, "
            f"Text: {len(report.text_cols)}.",
            (
                f"Missing values: {report.total_missing} total"
                + (
                    f" across {sum(1 for v in report.missing_counts.values() if v > 0)} "
                    f"column(s) ({report.missing_pct_total:.1f}% of all cells)."
                    if report.has_missing
                    else " — none."
                )
            ),
            f"Duplicate rows: {report.duplicate_rows}.",
            f"Data quality score: {report.data_quality_score}/100 — {report.data_quality_grade}.",
        ]
        if report.constant_cols:
            lines.append(f"Constant (zero-information) columns: {report.constant_cols}.")
        if report.near_constant_cols:
            lines.append(f"Near-constant columns (≥95% one value): {report.near_constant_cols}.")
        if report.high_cardinality_cols:
            lines.append(f"High-cardinality columns: {report.high_cardinality_cols}.")
        if report.outlier_counts:
            total_outliers = sum(report.outlier_counts.values())
            if total_outliers:
                top = sorted(report.outlier_counts.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"'{c}': {n}" for c, n in top)
                lines.append(
                    f"Outliers (IQR rule): {total_outliers} total. "
                    f"Top columns — {top_str}."
                )
        if report.task:
            lines.append(report.task_reason)
        return "\n".join(lines)

    def _handle_clean(self, text: str, norm: str, **ctx: Any) -> str:
        pre = preprocessor.Preprocessor()
        result = pre.clean(self.session.raw_df, target=self.session.target)
        self.session.cleaning_result = result
        if not result.steps:
            return (
                "Dataset was already clean — no missing values, duplicates, "
                "or constant columns found."
            )
        return "Cleaning complete:\n- " + "\n- ".join(result.steps)

    def _handle_feature_engineer(self, text: str, norm: str, **ctx: Any) -> str:
        if self.session.cleaning_result is None:
            self.ask("Clean my dataset")
        assert self.session.cleaning_result is not None
        report = self.session.dataset_report or analyzer.analyze(
            self.session.raw_df, target=self.session.target
        )
        eng = feature_engineer.FeatureEngineer()
        result = eng.fit_transform(
            self.session.cleaning_result.df,
            target=self.session.target,
            datetime_cols=report.datetime_cols,
            categorical_cols=report.categorical_cols,
        )
        self.session.feature_engineering_result = result
        return (
            f"Feature engineering complete — added {len(result.new_features)} "
            f"new feature(s):\n- " + "\n- ".join(result.steps)
        )

    def _handle_feature_select(self, text: str, norm: str, **ctx: Any) -> str:
        if self.session.cleaning_result is None:
            self.ask("Clean my dataset")
        assert self.session.cleaning_result is not None
        df_work = (
            self.session.feature_engineering_result.df
            if self.session.feature_engineering_result is not None
            else self.session.cleaning_result.df
        )
        X = df_work.drop(columns=[self.session.target], errors="ignore")
        y = (
            self.session.raw_df.loc[df_work.index, self.session.target]
            if self.session.target
            else None
        )
        report = self.session.dataset_report or analyzer.analyze(
            self.session.raw_df, target=self.session.target
        )
        task = (report.task or "classification") if self.session.target else "classification"
        sel = feature_selector.FeatureSelector()
        result = sel.select(X, y=y, task=task)
        self.session.feature_selection_result = result
        return (
            f"Feature selection complete: kept {result.n_selected} of "
            f"{result.n_original} feature(s) "
            f"({result.reduction_pct:.1f}% reduction):\n- "
            + "\n- ".join(result.steps)
        )

    def _handle_train(self, text: str, norm: str, **ctx: Any) -> str:
        if not self.session.target:
            return (
                "I need a target column first — "
                "try \"set target to <column name>\"."
            )
        if self.session.cleaning_result is None:
            self.ask("Clean my dataset")
        assert self.session.cleaning_result is not None

        # Determine task
        task: Optional[str] = None
        for t, pattern in _TASK_OVERRIDE_PATTERNS.items():
            if re.search(pattern, norm):
                task = t
                break
        if task is None:
            if self.session.dataset_report is None or self.session.dataset_report.task is None:
                self.session.dataset_report = analyzer.analyze(
                    self.session.raw_df, target=self.session.target
                )
            task = self.session.dataset_report.task or "classification"

        # Use feature-selected or engineered data if available, else cleaned
        if self.session.feature_selection_result is not None:
            df_work = (
                self.session.feature_engineering_result.df
                if self.session.feature_engineering_result
                else self.session.cleaning_result.df
            )
            sel_features = self.session.feature_selection_result.selected_features
            X = df_work[[c for c in sel_features if c in df_work.columns]]
        elif self.session.feature_engineering_result is not None:
            df_work = self.session.feature_engineering_result.df
            X = df_work.drop(columns=[self.session.target], errors="ignore")
        else:
            X = self.session.cleaning_result.df.drop(
                columns=[self.session.target], errors="ignore"
            )

        y = self.session.raw_df.loc[X.index, self.session.target]

        tr = trainer_mod.Trainer(task=task)
        result = tr.run(X, y)
        self.session.training_result = result

        lines = [
            f"Trained and compared {len(result.leaderboard)} models for {task}:"
        ]
        for e in result.leaderboard:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in e.test_metrics.items())
            marker = " ← best" if e.name == result.best_name else ""
            lines.append(
                f"  {e.name}: CV={e.cv_score:.4f}±{e.cv_std:.4f} | {metric_str}"
                f" ({e.fit_time_s}s){marker}"
            )
        lines.append(result.best_reason)
        return "\n".join(lines)

    def _handle_clustering(self, text: str, norm: str, **ctx: Any) -> str:
        # Extract optional k from text (e.g. "cluster into 4 groups")
        k: Optional[int] = None
        m = re.search(r"\b(\d+)\b", norm)
        if m:
            k = int(m.group(1))

        ct = clustering.ClusteringTrainer(n_clusters=k)
        result = ct.run(self.session.raw_df)
        self.session.clustering_result = result
        return (
            f"Clustering complete ({result.algorithm}):\n"
            f"  Clusters found: {result.n_clusters}\n"
            f"  Silhouette score: {result.silhouette:.3f}\n"
            f"  Davies-Bouldin index: {result.davies_bouldin:.3f}\n"
            + result.narrative
            + "\n- ".join([""] + result.steps)
        )

    def _handle_anomaly(self, text: str, norm: str, **ctx: Any) -> str:
        det = anomaly.AnomalyDetector()
        result = det.detect(self.session.raw_df)
        self.session.anomaly_result = result
        return (
            f"Anomaly detection complete ({result.algorithm}):\n"
            f"  Anomalies found: {result.n_anomalies} "
            f"({result.anomaly_pct}% of rows)\n"
            + result.narrative
        )

    def _handle_explain_choice(self, text: str, norm: str, **ctx: Any) -> str:
        if self.session.training_result is None:
            return (
                "I haven't trained any models yet, so there's no choice to explain. "
                "Try \"train the best model\" first."
            )
        return self.session.training_result.best_reason

    def _handle_feature_importance(self, text: str, norm: str, **ctx: Any) -> str:
        if self.session.training_result is None:
            return (
                "Train a model first (e.g. \"train the best model\") "
                "so there's something to explain."
            )
        tr = self.session.training_result
        result = explainer.explain_feature_importance(
            model=tr.best_model,
            X_test=tr.X_test,
            y_test=tr.y_test,
            feature_names=tr.feature_names,
            task=tr.task,
        )
        self.session.importance_result = result
        top = result.importances.head(10)
        listing = "\n".join(
            f"  {i + 1}. {name}: {val:.4f}"
            for i, (name, val) in enumerate(top.items())
        )
        return (
            f"Feature importance (via {result.method}):\n"
            f"{listing}\n"
            f"{result.narrative}"
        )

    def _handle_visualize(self, text: str, norm: str, **ctx: Any) -> str:
        produced: list[tuple[str, str]] = []
        out_dir = os.path.join(self.session.output_dir, "plots")

        wants_corr = "correlation" in norm or "heatmap" in norm
        wants_importance = "importance" in norm
        wants_confusion = "confusion" in norm
        wants_residual = "residual" in norm
        wants_comparison = "comparison" in norm
        wants_roc = "roc" in norm
        wants_pr = "pr curve" in norm or "precision" in norm
        wants_missing = "missing" in norm
        wants_dist = "distribut" in norm
        wants_any_specific = any([
            wants_corr, wants_importance, wants_confusion,
            wants_residual, wants_comparison, wants_roc,
            wants_pr, wants_missing, wants_dist,
        ])

        if wants_corr or not wants_any_specific:
            path = visualizer.plot_correlation_heatmap(self.session.raw_df, out_dir)
            self.session.visualizations["Correlation Heatmap"] = path
            produced.append(("Correlation Heatmap", path))

        if wants_missing:
            path = visualizer.plot_missing_heatmap(self.session.raw_df, out_dir)
            self.session.visualizations["Missing Value Heatmap"] = path
            produced.append(("Missing Value Heatmap", path))

        if wants_dist:
            path = visualizer.plot_distribution(self.session.raw_df, out_dir)
            self.session.visualizations["Feature Distributions"] = path
            produced.append(("Feature Distributions", path))

        if self.session.training_result is not None:
            tr = self.session.training_result

            if wants_comparison or not wants_any_specific:
                path = visualizer.plot_model_comparison(tr.leaderboard, out_dir)
                self.session.visualizations["Model Comparison"] = path
                produced.append(("Model Comparison", path))

            if wants_importance:
                if self.session.importance_result is None:
                    self._handle_feature_importance(text, norm)
                if self.session.importance_result:
                    path = visualizer.plot_feature_importance(
                        self.session.importance_result.importances, out_dir
                    )
                    self.session.visualizations["Feature Importance"] = path
                    produced.append(("Feature Importance", path))

            preds = tr.best_model.predict(tr.X_test)

            if tr.task == "classification" and (wants_confusion or not wants_any_specific):
                path = visualizer.plot_confusion_matrix(tr.y_test, preds, out_dir)
                self.session.visualizations["Confusion Matrix"] = path
                produced.append(("Confusion Matrix", path))

            if tr.task == "regression" and (wants_residual or not wants_any_specific):
                path = visualizer.plot_residuals(tr.y_test, preds, out_dir)
                self.session.visualizations["Residual Plot"] = path
                produced.append(("Residual Plot", path))

            # ROC curve (binary classification with predict_proba)
            if (wants_roc or not wants_any_specific) and tr.task == "classification":
                try:
                    if hasattr(tr.best_model, "predict_proba"):
                        prob = tr.best_model.predict_proba(tr.X_test)
                        if prob.shape[1] == 2:
                            path = visualizer.plot_roc_curve(
                                tr.y_test, prob[:, 1], out_dir
                            )
                            if path:
                                self.session.visualizations["ROC Curve"] = path
                                produced.append(("ROC Curve", path))
                except Exception:
                    pass

        if not produced:
            return (
                "Nothing to visualize yet — train a model first for "
                "confusion/residual/importance/comparison plots."
            )
        listing = "\n".join(f"  - {name}: {path}" for name, path in produced)
        return f"Generated {len(produced)} visualization(s):\n{listing}"

    def _handle_report(self, text: str, norm: str, **ctx: Any) -> str:
        if self.session.dataset_report is None:
            self.session.dataset_report = analyzer.analyze(
                self.session.raw_df, target=self.session.target
            )
        if (
            self.session.training_result is not None
            and "Model Comparison" not in self.session.visualizations
        ):
            self._handle_visualize("show all plots", "show all plots")

        fmt = ctx.get("format")
        if fmt is None:
            fmt = next(
                (name for name, pat in _FORMAT_PATTERNS.items()
                 if re.search(pat, norm)),
                "html",
            )

        if fmt in ("pdf", "docx"):
            html_path = reporter.generate_html_report(
                self.session,
                os.path.join(self.session.output_dir, "report.html"),
            )
            self.session.report_paths["html"] = html_path
            return (
                f"{fmt.upper()} export requires an optional dependency "
                f"(weasyprint for PDF, python-docx for DOCX). "
                f"An HTML report was generated at {html_path} instead — "
                f"any browser can print that to PDF."
            )

        if fmt == "markdown":
            path = reporter.generate_markdown_report(
                self.session,
                os.path.join(self.session.output_dir, "report.md"),
            )
            self.session.report_paths["markdown"] = path
            return f"Markdown report generated: {path}"

        path = reporter.generate_html_report(
            self.session,
            os.path.join(self.session.output_dir, "report.html"),
        )
        self.session.report_paths["html"] = path
        return f"HTML report generated: {path}"

    def _handle_save_model(self, text: str, norm: str, **ctx: Any) -> str:
        if self.session.training_result is None:
            return (
                "There's no trained model to save yet — "
                "try \"train the best model\" first."
            )
        path = ctx.get("path", os.path.join(self.session.output_dir, "model"))
        tr = self.session.training_result
        saved_path = persistence.save_model(
            path=path,
            model=tr.best_model,
            feature_names=tr.feature_names,
            target=self.session.target or "",
            task=tr.task,
            encoders=(self.session.cleaning_result.encoders
                      if self.session.cleaning_result else {}),
            scaler=(self.session.cleaning_result.scaler
                    if self.session.cleaning_result else None),
            model_name=tr.best_name,
            impute_values=(self.session.cleaning_result.impute_values
                           if self.session.cleaning_result else {}),
        )
        return f"Saved '{tr.best_name}' to {saved_path}."

    def _handle_load_model(self, text: str, norm: str, **ctx: Any) -> str:
        path = ctx.get("path")
        if not path:
            m = _PATH_RE.search(text)
            path = (
                m.group(0) if m
                else os.path.join(self.session.output_dir, "model.joblib")
            )
        bundle = persistence.load_model(path)
        self.session.loaded_bundle = bundle
        return (
            f"Loaded '{bundle.model_name}' ({bundle.task}) trained for target "
            f"'{bundle.target}', saved at {bundle.saved_at}."
        )

    def _handle_predict(self, text: str, norm: str, **ctx: Any) -> str:
        # Resolve the model to use: freshly trained, or explicitly loaded
        model = feature_names = encoders = scaler = task = numeric_cols = None
        impute_values: dict = {}

        if self.session.training_result is not None and self.session.loaded_bundle is None:
            tr = self.session.training_result
            model, feature_names, task = tr.best_model, tr.feature_names, tr.task
            encoders = (self.session.cleaning_result.encoders
                        if self.session.cleaning_result else {})
            scaler = (self.session.cleaning_result.scaler
                      if self.session.cleaning_result else None)
            numeric_cols = (self.session.cleaning_result.numeric_cols
                            if self.session.cleaning_result else [])
            impute_values = (self.session.cleaning_result.impute_values
                             if self.session.cleaning_result else {})
        elif self.session.loaded_bundle is not None:
            b = self.session.loaded_bundle
            model, feature_names, task = b.model, b.feature_names, b.task
            encoders, scaler = b.encoders, b.scaler
            numeric_cols = (
                list(scaler.feature_names_in_)
                if scaler is not None and hasattr(scaler, "feature_names_in_")
                else []
            )
            impute_values = b.impute_values or {}
        else:
            return (
                "There's no trained or loaded model yet — "
                "train one first or load a saved model."
            )

        # Resolve input data
        new_df: Optional[pd.DataFrame] = ctx.get("data")
        if new_df is None:
            file_path = ctx.get("file_path")
            if not file_path:
                m = _PATH_RE.search(text)
                file_path = m.group(0) if m else None
            if not file_path:
                return (
                    "I need data to predict on — pass file_path=\"yourfile.csv\" "
                    "or data=your_dataframe to ask(), "
                    "or mention a filename ending in .csv/.xlsx/.json/.parquet."
                )
            if not os.path.exists(file_path):
                return f"I can't find a file at '{file_path}'."
            try:
                new_df = loaders.load(file_path)
            except Exception as exc:
                return f"Failed to load '{file_path}': {exc}"

        cleaning_proxy = preprocessor.CleaningResult(
            df=new_df,
            steps=[],
            feature_names=feature_names or [],
            encoders=encoders or {},
            scaler=scaler,
            numeric_cols=numeric_cols or [],
            impute_values=impute_values,
        )
        X_new = preprocessor.Preprocessor.transform_new(new_df, cleaning_proxy)
        preds = model.predict(X_new)

        out_path = os.path.join(self.session.output_dir, "predictions.csv")
        result_df = new_df.copy()
        result_df["prediction"] = preds
        result_df.to_csv(out_path, index=False)

        preview = ", ".join(str(p) for p in list(preds[:5]))
        more = "..." if len(preds) > 5 else ""
        return (
            f"Predicted {len(preds)} row(s) with the {task} model. "
            f"First values: [{preview}{more}]. "
            f"Full results saved to {out_path}."
        )

    def _handle_unknown(self, text: str) -> str:
        return (
            "I'm not sure how to do that yet. Try things like:\n"
            "  \"clean my dataset\", \"engineer features\",\n"
            "  \"select features\", \"train the best model\",\n"
            "  \"cluster the data\", \"detect anomalies\",\n"
            "  \"why did you choose this algorithm?\",\n"
            "  \"show feature importance\", \"generate a report\",\n"
            "  \"save the model\", \"predict on this CSV file\".\n"
            "Say \"help\" for the full list."
        )
