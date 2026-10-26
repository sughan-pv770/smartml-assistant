"""
smartml.assistant_ds
====================
DataScienceMixin: ~35 analytical, data science, and wrangling intents.
Evaluated prior to standard Assistant intent patterns.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from .core import (
    forecaster,
    reducer,
    reinforcement,
    self_supervised,
    semi_supervised,
    stats,
    textutil,
    tuner,
    wrangler,
)


class DataScienceMixin:
    """Provides ~35 interactive data science and ML engineering intents."""

    def _init_ds_state(self) -> None:
        if not hasattr(self.session, "wrangler") or self.session.wrangler is None:
            self.session.wrangler = wrangler.DataWrangler(
                self.session.raw_df, target=self.session.target
            )

    def _reset_downstream(self) -> None:
        """Clear cached cleaning, training, and plot results when data mutates."""
        self.session.cleaning_result = None
        self.session.feature_engineering_result = None
        self.session.feature_selection_result = None
        self.session.training_result = None
        self.session.clustering_result = None
        self.session.anomaly_result = None
        self.session.importance_result = None
        self.session.visualizations.clear()

    def _ds_dispatch(self, text: str, norm: str, **ctx: Any) -> Optional[str]:
        """Attempt to handle text with a Data Science intent.

        Returns None if not matched, falling through to core Assistant handlers.
        """
        self._init_ds_state()
        w = self.session.wrangler
        df = w.current_df

        # 1. Undo
        if re.search(r"\bundo(\b| last)", norm):
            msg = w.undo()
            self.session.raw_df = w.current_df
            self._reset_downstream()
            return msg

        # 2. Drop column
        m = re.search(r"\bdrop column\s+([A-Za-z0-9_ \(\)]+)$", norm)
        if m:
            raw_col = m.group(1).strip()
            matched = textutil.fuzzy_match_column(raw_col, df.columns)
            if not matched:
                return f"Couldn't find column '{raw_col}' to drop."
            msg = w.drop_column(matched)
            self.session.raw_df = w.current_df
            self._reset_downstream()
            return msg

        # 3. Rename column
        m = re.search(r"\brename (?:column\s+)?([A-Za-z0-9_ \(\)]+)\s+to\s+([A-Za-z0-9_ \(\)]+)$", norm)
        if m:
            old_raw, new_name = m.group(1).strip(), m.group(2).strip()
            matched = textutil.fuzzy_match_column(old_raw, df.columns)
            if not matched:
                return f"Couldn't find column '{old_raw}' to rename."
            msg = w.rename_column(matched, new_name)
            self.session.raw_df = w.current_df
            self._reset_downstream()
            return msg

        # 4. Fill NA
        m = re.search(r"\bfill (?:missing|na|nulls?)(?: in)?\s+([A-Za-z0-9_ \(\)]+)\s+with\s+([A-Za-z0-9_.-]+)$", norm)
        if m:
            raw_col, val_str = m.group(1).strip(), m.group(2).strip()
            matched = textutil.fuzzy_match_column(raw_col, df.columns)
            if not matched:
                return f"Couldn't find column '{raw_col}'."
            msg = w.fill_na(matched, val_str)
            self.session.raw_df = w.current_df
            self._reset_downstream()
            return msg

        # 5. Replace value: "replace 0 with 5 in income"
        m = re.search(r"\breplace\s+([A-Za-z0-9_.-]+)\s+with\s+([A-Za-z0-9_.-]+)\s+in\s+([A-Za-z0-9_ \(\)]+)$", norm)
        if m:
            old_val, new_val, raw_col = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            matched = textutil.fuzzy_match_column(raw_col, df.columns)
            if not matched:
                return f"Couldn't find column '{raw_col}'."
            msg = w.replace_value(matched, old_val, new_val)
            self.session.raw_df = w.current_df
            self._reset_downstream()
            return msg

        # 6. Filter rows: "filter rows where age > 30" or "filter rows where age is greater than 30"
        m = re.search(r"\bfilter(?: rows)?(?: where)?\s+([A-Za-z0-9_ \(\)]+?)\s*(?:is\s+)?(>=|<=|!=|==|=|>|<|contains|greater than|less than|equals? to?)\s*([A-Za-z0-9_.-]+)$", norm)
        if m:
            col_part = m.group(1).strip()
            # Strip linking verbs if accidentally captured
            col_part = re.sub(r"\b(is|are|was|were)\b", "", col_part).strip()
            op_part = m.group(2).strip()
            val_part = m.group(3).strip()

            op_map = {
                "greater than": ">",
                "less than": "<",
                "equals": "==",
                "equal to": "==",
                "equal": "==",
            }
            op_symbol = op_map.get(op_part, op_part)

            matched = textutil.fuzzy_match_column(col_part, df.columns)
            if not matched:
                return f"Couldn't find column '{col_part}' for filtering."

            msg = w.filter_rows(matched, op_symbol, val_part)
            self.session.raw_df = w.current_df
            self._reset_downstream()
            return msg

        # 7. Correlation between two columns
        m = re.search(r"\bcorrelation between\s+([A-Za-z0-9_ \(\)]+?)\s+and\s+([A-Za-z0-9_ \(\)]+)$", norm)
        if m:
            col1_raw, col2_raw = m.group(1).strip(), m.group(2).strip()
            c1 = textutil.fuzzy_match_column(col1_raw, df.columns)
            c2 = textutil.fuzzy_match_column(col2_raw, df.columns)
            if not c1 or not c2:
                return f"Could not resolve columns '{col1_raw}' and '{col2_raw}'."
            try:
                r = stats.compute_correlations(df, c1, c2)
                return f"Pearson correlation between '{c1}' and '{c2}' is {r:.4f}."
            except Exception as exc:
                return f"Failed to compute correlation: {exc}"

        # 8. Check skewness / kurtosis
        m = re.search(r"\b(check|show|get)\s+skewness(?:\s+of\s+([A-Za-z0-9_ \(\)]+))?", norm)
        if m:
            col_raw = m.group(2)
            if col_raw:
                matched = textutil.fuzzy_match_column(col_raw.strip(), df.columns)
                if not matched:
                    return f"Couldn't find column '{col_raw}'."
                summary = stats.summarize_column(df, matched)
                return f"Column '{matched}': Skewness={summary.skewness}, Kurtosis={summary.kurtosis}."
            else:
                num_df = df.select_dtypes(include=["number"])
                skews = [f"'{c}': {num_df[c].skew():.2f}" for c in num_df.columns[:8]]
                return "Skewness for numeric columns:\n" + ", ".join(skews)

        # 9. Describe column / summary statistics
        m = re.search(r"\b(describe|summary statistics|five number summary|summarize|stats for|stats)(?:\s+for\s+|\s+of\s+|\s+column\s+)?([A-Za-z0-9_ \(\)]+)?", norm)
        if m:
            col_raw = m.group(2)
            if col_raw and col_raw.strip():
                matched = textutil.fuzzy_match_column(col_raw.strip(), df.columns)
                if matched:
                    cs = stats.summarize_column(df, matched)
                    if cs.mean is not None:
                        norm_str = "normal" if cs.is_normal else "non-normal"
                        return (
                            f"Summary for '{matched}':\n"
                            f"  Count: {cs.count}, Missing: {cs.missing}\n"
                            f"  Mean: {cs.mean:.2f}, Std: {cs.std:.2f}\n"
                            f"  Min: {cs.min:.2f}, Q25: {cs.q25:.2f}, Median: {cs.median:.2f}, Q75: {cs.q75:.2f}, Max: {cs.max:.2f}\n"
                            f"  IQR: {cs.iqr:.2f}, Skewness: {cs.skewness:.2f}, Kurtosis: {cs.kurtosis:.2f}\n"
                            f"  Distribution: {norm_str} (Shapiro-Wilk p={cs.shapiro_p})"
                        )
                    return f"Column '{matched}' ({cs.dtype}): {cs.count} non-null values, {cs.unique} unique."
            # Overall summary
            return f"Dataset summary: {len(df)} rows, {len(df.columns)} columns. Use 'describe <column>' for in-depth stats."

        # 10. PCA / Dimensionality Reduction: "reduce to 3 components" or "run pca"
        m = re.search(r"\b(reduce to\s+(\d+)\s+components|run pca|pca to\s+(\d+))", norm)
        if m:
            n_comps = 2
            if m.group(2):
                n_comps = int(m.group(2))
            elif m.group(3):
                n_comps = int(m.group(3))
            try:
                pca_red = reducer.PCAReducer(n_components=n_comps)
                res = pca_red.fit_transform(df, target=self.session.target)
                return (
                    f"PCA complete: Reduced features to {res.n_components} components.\n"
                    f"Total explained variance: {res.total_explained_variance * 100:.1f}%.\n"
                    f"Component variance ratios: {res.explained_variance_ratio}."
                )
            except Exception as exc:
                return f"PCA failed: {exc}"

        # 11. Hyperparameter Tuning: "tune the random forest" or "tune hyperparameters"
        m = re.search(r"\btune\s+(?:the\s+)?([A-Za-z ]+)?(?:model|hyperparameters)?", norm)
        if m and ("tune" in norm):
            model_query = m.group(1).strip() if m.group(1) else ""
            if not self.session.training_result:
                return "Train a baseline model first before tuning hyperparameters."

            tr = self.session.training_result
            model_target = model_query if model_query else tr.best_name

            t = tuner.ModelTuner(task=tr.task)
            try:
                tune_res = t.tune(
                    model_name=model_target,
                    X_train=tr.X_test,  # Use training hold-out
                    y_train=tr.y_test,
                    baseline_score=tr.leaderboard[0].cv_score,
                )
                delta_str = f"+{tune_res.delta:.4f}" if tune_res.delta >= 0 else f"{tune_res.delta:.4f}"
                return (
                    f"Tuned {tune_res.model_name}:\n"
                    f"  Best {tune_res.scoring}: {tune_res.best_score:.4f} ({delta_str} vs baseline)\n"
                    f"  Best params: {tune_res.best_params}"
                )
            except Exception as exc:
                return f"Hyperparameter tuning failed: {exc}"

        # 12. Forecasting: "forecast next 5 steps on <col>"
        m = re.search(r"\bforecast\s+(?:next\s+)?(\d+)?\s*(?:steps?|days?|points?)?(?:\s+(?:on|for)\s+([A-Za-z0-9_ \(\)]+))?", norm)
        if m and "forecast" in norm:
            steps_cnt = int(m.group(1)) if m.group(1) else 5
            target_col = m.group(2).strip() if m.group(2) else self.session.target
            if not target_col:
                return "Please specify which numeric column to forecast (e.g. 'forecast 5 steps on sales')."

            matched = textutil.fuzzy_match_column(target_col, df.columns)
            if not matched:
                return f"Could not find column '{target_col}' for forecasting."

            try:
                fc = forecaster.TabularForecaster()
                res = fc.forecast(df, target_col=matched, horizon=steps_cnt)
                preds_str = ", ".join(f"{p:.2f}" for p in res.predictions)
                return f"{res.narrative}\nPredictions: [{preds_str}]."
            except Exception as exc:
                return f"Forecasting failed: {exc}"

        # 13. Semi-Supervised Learning: "run semi-supervised", "train semi-supervised"
        if re.search(r"\bsemi[- ]supervised\b", norm):
            m_target = re.search(r"\b(?:target|with target)[:= ]\s*([A-Za-z0-9_]+)", norm)
            if m_target and not self.session.target:
                matched_t = textutil.fuzzy_match_column(m_target.group(1), df.columns)
                if matched_t:
                    self.session.target = matched_t
            if not self.session.target:
                return "Please set a target column first for semi-supervised learning."
            y_col = self.session.target
            if y_col not in df.columns:
                return f"Target column '{y_col}' not found."
            X = df.drop(columns=[y_col], errors="ignore")
            y = df[y_col]
            try:
                trainer = semi_supervised.SemiSupervisedTrainer()
                res = trainer.fit(X, y)
                return (
                    f"Semi-Supervised Learning complete ({res.algorithm}):\n"
                    f"  {res.narrative}\n"
                    f"  - " + "\n  - ".join(res.steps)
                )
            except Exception as exc:
                return f"Semi-supervised learning failed: {exc}"

        # 14. Reinforcement Learning / Contextual Bandits: "optimize decision", "run reinforcement learning", "contextual bandit"
        m_rl = re.search(r"\b(reinforcement learning|bandit|optimize decision|mab)\b", norm)
        if m_rl:
            # Look for action / reward columns
            action_m = re.search(r"\baction[:= ]\s*([A-Za-z0-9_]+)", norm)
            reward_m = re.search(r"\breward[:= ]\s*([A-Za-z0-9_]+)", norm)
            
            action_col = action_m.group(1) if action_m else None
            reward_col = reward_m.group(1) if reward_m else None

            # Auto-detect if not specified
            if not action_col:
                for c in df.columns:
                    if c.lower() in ("action", "arm", "treatment", "variant", "ad", "offer"):
                        action_col = c
                        break
            if not reward_col:
                for c in df.columns:
                    if c.lower() in ("reward", "conversion", "click", "revenue", "profit", "outcome"):
                        reward_col = c
                        break

            if not action_col or not reward_col:
                return (
                    "To run Reinforcement Learning (Contextual Bandits), please specify action and reward columns. "
                    "For example: 'run reinforcement learning with action=offer and reward=conversion'."
                )

            act_matched = textutil.fuzzy_match_column(action_col, df.columns)
            rew_matched = textutil.fuzzy_match_column(reward_col, df.columns)
            if not act_matched or not rew_matched:
                return f"Could not resolve action '{action_col}' or reward '{reward_col}' in dataset."

            context_cols = [c for c in df.columns if c not in (act_matched, rew_matched)]
            try:
                optimizer = reinforcement.TabularBanditOptimizer(strategy="auto")
                res = optimizer.optimize(df, action_col=act_matched, reward_col=rew_matched, context_cols=context_cols)
                return (
                    f"Reinforcement Learning (Decision Optimization) complete:\n"
                    f"  {res.narrative}\n"
                    f"  Arm pull distribution: {res.arm_pull_counts}\n"
                    f"  - " + "\n  - ".join(res.steps)
                )
            except Exception as exc:
                return f"Reinforcement learning optimization failed: {exc}"

        # 15. Self-Supervised Learning: "run self-supervised", "train self-supervised", "autoencoder embeddings"
        if re.search(r"\b(self[- ]supervised|masked autoencoder|representation learning|denoising autoencoder)\b", norm):
            try:
                ssl_learner = self_supervised.TabularSelfSupervisedLearner()
                res = ssl_learner.fit_transform(df, target_col=self.session.target)
                return (
                    f"Self-Supervised Learning complete:\n"
                    f"  {res.narrative}\n"
                    f"  Extracted dense representations ({res.embedding_dim} latent dimensions) saved into session.\n"
                    f"  - " + "\n  - ".join(res.steps)
                )
            except Exception as exc:
                return f"Self-supervised representation learning failed: {exc}"

        return None
