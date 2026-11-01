"""
smartml.mial.tools
==================
Part B Tool Executor and OpenAI Function Schemas for Mial.

Implements the exact tools specified in Part B2 of MIAL_FULL_SYSTEM_PROMPTS.md:
  - load_dataset(name)
  - set_target(column)
  - ask_smartml(command, confirmed=False)
  - run_full_pipeline(confirmed=False)
  - get_status()
  - open_report()
  - undo()
  - stop()

Hard safety guarantees (Part A5 & Part B3/B5):
  - Zero dynamic code execution anywhere in the tool layer (A5.5).
  - Dataset text/columns wrapped in <untrusted_data> tags so data is never treated as instructions (B3.3).
  - Pre-mutation confirmation gate before changing data or overwriting saved models, noting 'undo' (B3.5).
  - Ambiguous column disambiguation ('Did you mean sale_price?') instead of silent guessing (B3.6).
  - Honest verification warnings for low/baseline scores, small datasets (<500 rows), leakage (>=0.995),
    forecasts not beating naive persistence, and non-rebuildable engineered features on predict (B5.1-B5.5).
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..agent.nlp import ALLOWED_DATA_EXTENSIONS, find_dataset_candidates, is_within_allowed_dirs
from ..agent.voice_agent import VoiceAgent
from ..core import analyzer, stats as sml_stats, textutil, visualizer
from .runner import is_heavy_command


_INJECTION_PHRASE_RE = re.compile(
    r"("
    r"ignore\s+(?:all|your|previous|prior|above|system|earlier)\b|"
    r"\bsystem\s+prompt\b|"
    r"\bapi[_\-\s]?key\b|"
    r"\breveal\s+(?:your|the)\b|"
    r"\bdisregard\s+(?:all|previous|prior)\b|"
    r"\bjailbreak\b|"
    r"\bcall\s+(?:run_full_pipeline|ask_smartml|load_dataset|set_target|open_report|undo|stop)\b|"
    r"\brun_full_pipeline\b|"
    r"\bdelete\s+(?:everything|all|files)\b|"
    r"\bexecute\s+command\b|"
    r"\byou\s+are\s+now\b"
    r")",
    re.IGNORECASE,
)

_GENERIC_WINNER_TAIL_RE = re.compile(
    r"\s+(?:As a tree-based ensemble,.*|Its strong performance suggests.*|This suggests the target is well predicted.*)$",
    re.DOTALL,
)

_RAW_PREVIEW_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Arm pull distribution:\s*\{[^}]*\}", re.IGNORECASE), "Arm pull distribution: [cell values omitted]"),
    (re.compile(r"First values:\s*\[[^\]]*\]", re.IGNORECASE), "First values: [cell values omitted]"),
    (re.compile(r"Recommended arm\s+'[^']*'", re.IGNORECASE), "Recommended arm [cell value omitted]"),
    (re.compile(r"Best arm:\s*'[^']*'", re.IGNORECASE), "Best arm: [cell value omitted]"),
    (re.compile(r"Top value:\s*'[^']*'", re.IGNORECASE), "Top value: [cell value omitted]"),
    (re.compile(r"Row\s+\d+:\s*[^\r\n]+", re.IGNORECASE), "[row values omitted]"),
]


def is_suspicious_identifier(value: Any) -> bool:
    """Return True if a column, file, or cell string looks like an instruction/prompt-injection payload."""
    cleaned = re.sub(r"[\r\n\t<>]+", " ", str(value or "")).strip()
    if not cleaned:
        return False
    if _INJECTION_PHRASE_RE.search(cleaned):
        return True
    if len(cleaned.split()) >= 8 and any(
        w in cleaned.lower().split() for w in ("ignore", "system", "prompt", "instruction", "instructions", "rules", "execute", "delete")
    ):
        return True
    return False


def redact_identifier(value: Any, kind: str = "column") -> str:
    """Return a placeholder if *value* is instruction-like, otherwise its cleaned text."""
    cleaned = re.sub(r"[\r\n\t<>]+", " ", str(value or "")).strip()
    if is_suspicious_identifier(cleaned):
        return f"[suspicious {kind} name omitted]"
    return cleaned


def wrap_untrusted(value: Any, max_len: int = 80) -> str:
    """Wrap dataset-originated text (column names, file names, values) as untrusted data (B3.3)."""
    cleaned = re.sub(r"[\r\n\t<>]+", " ", str(value)).strip()
    if is_suspicious_identifier(cleaned):
        return "<untrusted_data>[suspicious column name omitted]</untrusted_data>"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "..."
    return f"<untrusted_data>{cleaned}</untrusted_data>"


def _measured_why_winner(reason: Optional[str]) -> Optional[str]:
    """Return only the measured cross-validation sentence from Trainer._explain_choice (B3.1)."""
    if not reason:
        return None
    return _GENERIC_WINNER_TAIL_RE.sub("", str(reason).strip()).strip()


def _scrub_text_for_llm(text: str, df: Optional[pd.DataFrame] = None) -> str:
    """Strip raw row/cell previews, prompt-injection text, and raw string cell values before cloud transmission."""
    out = str(text or "")
    if not out:
        return out

    for pat, repl in _RAW_PREVIEW_PATTERNS:
        out = pat.sub(repl, out)

    # Redact any remaining line/segment containing prompt-injection instructions
    safe_lines: list[str] = []
    for line in out.splitlines():
        if _INJECTION_PHRASE_RE.search(line):
            safe_lines.append("[suspicious dataset text omitted]")
        else:
            safe_lines.append(line)
    out = "\n".join(safe_lines)

    # Scrub distinct string/categorical cell values from the active DataFrame (summaries only, never raw cells)
    if df is not None and not df.empty:
        col_names_lower = {str(c).strip().lower() for c in df.columns}
        protected_terms = col_names_lower | {
            "classification", "regression", "normal", "non-normal", "none", "true", "false",
            "mean", "median", "mode", "count", "missing", "summary", "dataset", "model",
            "accuracy", "precision", "recall", "silhouette", "kmeans",
        }
        cell_literals: set[str] = set()
        str_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        for col in str_cols:
            try:
                uniques = df[col].dropna().astype(str).unique()[:500]
            except Exception:
                continue
            for val in uniques:
                v = val.strip()
                if len(v) >= 4 and v.lower() not in protected_terms:
                    cell_literals.add(v)
        for lit in sorted(cell_literals, key=len, reverse=True):
            if lit in out:
                out = out.replace(lit, "[cell_value]")

    # Strip local filesystem directories from paths while keeping file basenames
    out = re.sub(r"[A-Za-z]:[\\/](?:[^\\/:\r\n]+[\\/])+([^\\/:\r\n]+)", r"\1", out)
    return out


def summarize_tool_output_for_llm(
    tool_name: str,
    output: dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Build a privacy-safe, aggregate-only summary dict to send to cloud LLM providers.

    Enforces two guarantees:
    1. Only aggregate metadata, column schemas, and numeric metrics are kept — never raw DataFrame rows or cells.
    2. Any instruction-like text in column names, cell values, or tool strings is redacted before leaving the machine.
    """
    if not isinstance(output, dict):
        return {"status": "error", "error": "Invalid tool output."}

    safe: dict[str, Any] = {}
    scalar_keys = (
        "status", "loaded", "rows", "cols", "cleaned_rows", "cleaned_cols",
        "numeric_cols", "categorical_cols", "datetime_cols", "total_missing",
        "duplicate_rows", "quality_score", "inferred_task", "task", "cleaned",
        "engineered_features_active", "best_model", "models_compared",
        "cv_metric", "cv_score", "cv_std", "last_cv_score", "test_metrics",
        "leaderboard", "undo_available", "unchanged", "new_features_count",
        "method", "stopped", "cancelled_job",
    )
    for k in scalar_keys:
        if k in output and output[k] is not None:
            safe[k] = output[k]

    if "file" in output and output["file"]:
        raw_f = re.sub(r"</?untrusted_data>", "", str(output["file"]))
        safe["file"] = wrap_untrusted(os.path.basename(raw_f))

    if "target" in output and output["target"] is not None:
        safe["target"] = redact_identifier(output["target"], kind="column")

    for list_key in ("columns", "matches", "discoverable_files", "suggestions", "new_features"):
        if list_key in output and isinstance(output[list_key], list):
            cleaned_items: list[str] = []
            for item in output[list_key][:20]:
                raw_item = re.sub(r"</?untrusted_data>", "", str(item))
                if list_key == "suggestions":
                    cleaned_items.append(redact_identifier(raw_item, kind="column"))
                else:
                    cleaned_items.append(wrap_untrusted(raw_item))
            safe[list_key] = cleaned_items

    if "top_features" in output:
        tf = output["top_features"]
        if isinstance(tf, list):
            safe["top_features"] = [redact_identifier(x, kind="feature") for x in tf[:8]]
        elif isinstance(tf, dict):
            safe["top_features"] = {
                redact_identifier(k, kind="feature"): v for k, v in list(tf.items())[:8]
            }

    if "missing_by_column" in output and isinstance(output["missing_by_column"], dict):
        safe["missing_by_column"] = {
            redact_identifier(k, kind="column"): int(v)
            for k, v in list(output["missing_by_column"].items())[:20]
        }

    if "why_winner" in output and output["why_winner"]:
        safe["why_winner"] = _measured_why_winner(str(output["why_winner"]))

    if "warnings" in output and isinstance(output["warnings"], list):
        safe["warnings"] = [_scrub_text_for_llm(str(w), df=df) for w in output["warnings"]]

    for path_map_key in ("reports", "report_paths", "plots"):
        if path_map_key in output and isinstance(output[path_map_key], dict):
            safe[path_map_key] = {
                str(k): os.path.basename(str(v)) for k, v in output[path_map_key].items() if v
            }

    for text_key in ("summary", "message", "error", "question", "command"):
        if text_key in output and output[text_key]:
            safe[text_key] = _scrub_text_for_llm(str(output[text_key]), df=df)[:900]

    return safe


_MUTATING_CMD_RE = re.compile(
    r"\b("
    r"drop\s+(?:column|columns|col|cols|duplicates)|"
    r"delete\s+(?:column|columns|rows)|"
    r"remove\s+(?:outliers?|duplicates?|rows?|columns?)|"
    r"filter\s+rows?|keep\s+rows?|"
    r"replace\b|fill\s+missing|impute\b|"
    r"rename\s+column|convert\s+|cast\s+"
    r")",
    re.IGNORECASE,
)


MIAL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "load_dataset",
            "description": "Find and load a data file by full or partial spoken name from the allowed folders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Spoken or partial dataset name, such as 'student' or 'sales.csv'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_target",
            "description": "Choose the target column to predict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to set as prediction target.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set true only after the user confirmed a suggested column name.",
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_smartml",
            "description": (
                "Run one SmartML command in plain English: summary of the data, clean my dataset, "
                "engineer features, select features, train the best model, tune, why did that model win, "
                "show feature importance, cluster the data into 3 segments, detect anomalies, run pca, "
                "forecast 5 steps on <col>, show all plots, generate html report, save model, "
                "remove outliers, drop column <col>, fill missing in <col> with median, predict on <file>."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Single plain-English SmartML command to execute.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": (
                            "Set true ONLY if the user explicitly asked for this exact data-changing "
                            "or model-overwriting operation, or confirmed when asked."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_full_pipeline",
            "description": "Analyse, clean, train models, compute feature importance, generate plots, build report, and save the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set true if overwriting an existing saved model was confirmed by the user.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Return current session status: loaded dataset, row/column counts, columns, target, trained model, and discoverable files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_report",
            "description": "Generate (if needed) and open the HTML analysis and model report in the browser.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": "Reverse the last data-wrangling change on the dataset.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Cancel any running background job and stop immediately.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class SmartMLToolExecutor:
    """Executes Part B SmartML tools against the live ``VoiceAgent`` session without eval/exec."""

    def __init__(self, agent: VoiceAgent) -> None:
        self.agent = agent
        self.pending_confirmation: Optional[dict[str, Any]] = None

    def execute_calls(
        self,
        tool_calls: list[dict[str, Any]],
        runner: Optional[Any] = None,
    ) -> list[dict[str, Any]]:
        """Execute an ordered list of ``{"tool": "<name>", "args": {...}}`` calls."""
        results: list[dict[str, Any]] = []
        for call in tool_calls:
            tool_name = str(call.get("tool", "")).strip()
            args = call.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            out = self.execute_one(tool_name, args, runner=runner)
            results.append({"tool": tool_name, "args": args, "output": out})
            if isinstance(out, dict) and out.get("status") in (
                "error",
                "needs_confirmation",
                "disambiguate",
                "disambiguate_column",
                "cancelled",
            ):
                break
        return results

    def execute_one(
        self,
        tool_name: str,
        args: dict[str, Any],
        runner: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Dispatch a single tool call and run Part B5 honesty verification checks."""
        short = tool_name.replace("smartml.", "").strip().lower()
        dispatch = {
            # Canonical Part B2 tools
            "load_dataset": self._tool_load_dataset,
            "set_target": self._tool_set_target,
            "ask_smartml": lambda a: self._tool_ask_smartml(a, runner=runner),
            "run_full_pipeline": lambda a: self._tool_run_full_pipeline(a, runner=runner),
            "get_status": self._tool_get_status,
            "open_report": self._tool_open_report,
            "undo": self._tool_undo,
            "stop": lambda a: self._tool_stop(a, runner=runner),
            # Safe compatibility aliases (no run_code / no exec)
            "load_data": self._tool_load_dataset,
            "profile_data": self._tool_profile_data,
            "clean_data": self._tool_clean_data,
            "eda": self._tool_eda,
            "engineer_features": self._tool_engineer_features,
            "train_automl": lambda a: self._tool_train_automl(a, runner=runner),
            "compare_models": self._tool_compare_models,
            "tune": lambda a: self._tool_tune(a, runner=runner),
            "evaluate": self._tool_evaluate,
            "explain": self._tool_explain,
            "predict": self._tool_predict,
            "plot": self._tool_plot,
            "generate_report": self._tool_generate_report,
            "export_model": self._tool_export_model,
            "session_state": self._tool_get_status,
        }
        handler = dispatch.get(short)
        if handler is None:
            return {"status": "error", "error": f"Unknown tool '{tool_name}'."}
        try:
            return handler(args)
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Internal Helpers & Part B5 Verification
    # ------------------------------------------------------------------ #

    def _require_session(self):
        if not self.agent.has_dataset or self.agent.assistant is None:
            return None
        return self.agent.assistant.session

    def _list_discoverable_files(self, limit: int = 15) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for base in self.agent.search_dirs:
            try:
                base_p = Path(base).resolve()
                if not base_p.is_dir():
                    continue
                for root, dirs, files in os.walk(base_p):
                    rel_depth = len(Path(root).relative_to(base_p).parts)
                    if rel_depth >= 2:
                        dirs[:] = []
                    dirs[:] = [
                        d for d in dirs
                        if not d.startswith(".") and d not in ("venv", ".venv", "node_modules", "__pycache__", "build", "dist")
                    ]
                    for fname in files:
                        if os.path.splitext(fname)[1].lower() in ALLOWED_DATA_EXTENSIONS:
                            fpath = Path(root) / fname
                            if is_within_allowed_dirs(fpath, self.agent.search_dirs):
                                r = str(fpath.resolve())
                                if r not in seen:
                                    seen.add(r)
                                    found.append(fname)
                                    if len(found) >= limit:
                                        return found
            except OSError:
                continue
        return found

    def _suggest_columns(self, query: str, columns: list[str]) -> list[str]:
        q_low = query.strip().lower()
        q_norm = textutil.normalize_token(q_low)
        substr = [
            c for c in columns
            if q_low in c.lower() or c.lower() in q_low or (q_norm and q_norm in textutil.normalize_token(c))
        ]
        fuzzy = difflib.get_close_matches(query.strip(), columns, n=3, cutoff=0.45)
        merged: list[str] = []
        for c in substr + fuzzy:
            if c not in merged:
                merged.append(c)
        return merged[:3]

    def _verify_training_output(self, s) -> list[str]:
        """Enforce Part B5 honesty checks on real training results."""
        warnings: list[str] = []
        tr = s.training_result
        if tr is None or not tr.leaderboard:
            return warnings

        best = tr.leaderboard[0]
        if best.cv_score >= 0.995:
            warnings.append(
                f"LEAKAGE WARNING: {tr.best_name} scored {best.cv_score:.4f} (~100%). "
                "Check whether an ID or post-outcome column leaked into features."
            )
        for m_name, m_val in best.test_metrics.items():
            if m_name in ("accuracy", "f1_weighted", "roc_auc", "r2") and m_val >= 0.995:
                warnings.append(
                    f"LEAKAGE WARNING: Hold-out {m_name} is {m_val:.4f} (~100%) — possible target leakage."
                )
                break

        if len(s.raw_df) < 500:
            warnings.append(
                f"SMALL DATASET NOTE: Dataset has {len(s.raw_df)} rows; results on a few hundred rows may not be stable."
            )

        # B5.1: Check if score is low or close to simple majority / mean baseline
        if tr.task == "classification":
            maj_share = 0.5
            src_df = s.cleaning_result.df if s.cleaning_result is not None else s.raw_df
            if s.target and s.target in src_df.columns:
                vc = src_df[s.target].value_counts(normalize=True, dropna=True)
                if len(vc) >= 1:
                    maj_share = float(vc.max())
                if len(vc) >= 2 and float(vc.min()) < 0.20:
                    warnings.append(
                        f"CLASS IMBALANCE: Minority class is {float(vc.min()) * 100:.1f}% — rely on F1/ROC-AUC rather than accuracy."
                    )
            if best.cv_score < 0.60 or best.cv_score <= maj_share + 0.03:
                warnings.append(
                    f"LOW OR BASELINE SCORE: Best {tr.scoring} ({best.cv_score:.4f}) is low or close to the simple majority baseline ({maj_share:.4f})."
                )
        elif tr.task == "regression":
            if best.cv_score <= 0.15:
                warnings.append(
                    f"LOW OR BASELINE SCORE: Best R2 ({best.cv_score:.4f}) is low or close to predicting the simple mean baseline (0.0)."
                )
        return warnings

    def _has_non_rebuildable_engineered_features(self, s) -> bool:
        """Check B5.5: engineered features active in trained model cannot be rebuilt for new raw data."""
        if s.training_result is None or s.feature_engineering_result is None:
            return False
        eng_feats = set(s.feature_engineering_result.new_features or [])
        if not eng_feats:
            return False
        model_feats = set(s.training_result.feature_names or [])
        return bool(eng_feats & model_feats)

    def _training_payload(self, s) -> dict[str, Any]:
        tr = s.training_result
        if tr is None or not tr.leaderboard:
            return {}
        top_features: list[str] = []
        if s.importance_result is None:
            try:
                self.agent.assistant.ask("show feature importance")
            except Exception:
                pass
        if s.importance_result is not None:
            top_features = [redact_identifier(k, kind="feature") for k in list(s.importance_result.importances.head(3).index)]
        best = tr.leaderboard[0]
        return {
            "task": tr.task,
            "target": redact_identifier(s.target, kind="column") if s.target else None,
            "rows": int(s.raw_df.shape[0]),
            "models_compared": len(tr.leaderboard),
            "best_model": tr.best_name,
            "cv_metric": tr.scoring,
            "cv_score": round(best.cv_score, 4),
            "cv_std": round(best.cv_std, 4),
            "test_metrics": {k: round(v, 4) for k, v in best.test_metrics.items()},
            "top_features": top_features,
            "why_winner": _measured_why_winner(tr.best_reason),
            "leaderboard": [
                {
                    "model": e.name,
                    "cv_score": round(e.cv_score, 4),
                    "test_metrics": {k: round(v, 4) for k, v in e.test_metrics.items()},
                }
                for e in tr.leaderboard[:6]
            ],
            "warnings": self._verify_training_output(s),
        }

    # ------------------------------------------------------------------ #
    # Canonical Part B2 Tools
    # ------------------------------------------------------------------ #

    def _tool_load_dataset(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("name") or args.get("path") or args.get("file") or "").strip()
        if not query:
            return {
                "status": "error",
                "error": "Missing dataset file name.",
                "discoverable_files": [wrap_untrusted(f) for f in self._list_discoverable_files()],
            }

        allow_outside = bool(args.get("allow_outside", False))
        p = Path(query)
        if allow_outside and p.is_file():
            msg = self.agent.load_file(str(p), allow_outside=True)
        else:
            cands = find_dataset_candidates(query, search_dirs=self.agent.search_dirs)
            if not cands:
                return {
                    "status": "error",
                    "error": f"No dataset matching '{query}' was found in the allowed folders.",
                    "discoverable_files": [wrap_untrusted(f) for f in self._list_discoverable_files()],
                }
            if len(cands) > 1:
                self.agent.pending_candidates = cands[:3]
                return {
                    "status": "disambiguate",
                    "matches": [wrap_untrusted(Path(c).name) for c in self.agent.pending_candidates],
                    "message": "Multiple files match. Ask the user to choose one of the top matches.",
                }
            msg = self.agent.load_file(cands[0], allow_outside=False)

        s = self._require_session()
        if s is None or msg.startswith("Blocked:") or msg.startswith("Failed") or msg.startswith("I couldn't"):
            return {"status": "error", "error": msg}

        warnings: list[str] = []
        if len(s.raw_df) < 500:
            warnings.append(
                f"SMALL DATASET NOTE: Dataset has {len(s.raw_df)} rows; model results may not be stable."
            )

        return {
            "status": "ok",
            "message": msg,
            "file": wrap_untrusted(os.path.basename(self.agent.current_file or query)),
            "rows": int(s.raw_df.shape[0]),
            "cols": int(s.raw_df.shape[1]),
            "columns": [wrap_untrusted(c) for c in s.raw_df.columns],
            "warnings": warnings,
        }

    def _tool_set_target(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset is loaded yet."}

        requested = str(args.get("column") or args.get("target") or "").strip()
        if not requested:
            return {
                "status": "error",
                "error": "No target column name provided.",
                "columns": [wrap_untrusted(c) for c in s.raw_df.columns],
            }

        raw_cols = [str(c) for c in s.raw_df.columns]
        req_norm = textutil.normalize_token(requested)

        # 1. Exact or normalized-exact match (e.g. 'would recommend' == 'Would_Recommend')
        exact_matches = [
            c for c in raw_cols
            if c.lower() == requested.lower() or (req_norm and textutil.normalize_token(c) == req_norm)
        ]
        if len(exact_matches) == 1:
            chosen = exact_matches[0]
            msg = self.agent.assistant.set_target(chosen)
            return {
                "status": "ok",
                "target": chosen,
                "task": s.dataset_report.task if s.dataset_report else None,
                "message": msg,
            }

        # 2. Ambiguous or partial match -> return suggestions so Mial asks "Did you mean ...?" (B3.6, B7 Ex 4)
        suggestions = self._suggest_columns(requested, raw_cols)
        if bool(args.get("confirmed", False)) and len(suggestions) == 1:
            chosen = suggestions[0]
            msg = self.agent.assistant.set_target(chosen)
            return {
                "status": "ok",
                "target": chosen,
                "task": s.dataset_report.task if s.dataset_report else None,
                "message": msg,
            }

        return {
            "status": "disambiguate_column",
            "requested": requested,
            "suggestions": suggestions,
            "columns": [wrap_untrusted(c) for c in raw_cols[:15]],
            "message": (
                f"Column '{requested}' was not found. "
                + (f"Did you mean: {', '.join(suggestions)}?" if suggestions else "Ask the user which column to predict.")
            ),
        }

    def _tool_ask_smartml(self, args: dict[str, Any], runner: Optional[Any] = None) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {
                "status": "error",
                "error": "No dataset is loaded yet. Offer to load one by name.",
                "discoverable_files": [wrap_untrusted(f) for f in self._list_discoverable_files()],
            }

        command = str(args.get("command") or "").strip()
        if not command:
            return {"status": "error", "error": "Empty SmartML command."}

        confirmed = bool(args.get("confirmed", False))
        cmd_low = command.lower()

        # B5.5: Block predict if model uses engineered features that cannot be rebuilt for new raw data
        is_predict_cmd = bool(re.search(r"\b(predict|score)\b", cmd_low)) and not bool(
            re.search(r"\b(train|build|fit|target)\b", cmd_low)
        )
        if is_predict_cmd and self._has_non_rebuildable_engineered_features(s):
            return {
                "status": "error",
                "error": (
                    "Cannot predict yet because the trained model uses engineered features "
                    "that cannot be rebuilt automatically for new data."
                ),
            }
        if is_predict_cmd:
            pred_path_m = re.search(
                r"([A-Za-z]:[\\/][\w\-./\\ ]+\.(?:csv|tsv|xlsx?|json|parquet|feather)|"
                r"(?:\.\.?[\\/]|/)[\w\-./\\]+\.(?:csv|tsv|xlsx?|json|parquet|feather)|"
                r"[\w\-./\\]+\.(?:csv|tsv|xlsx?|json|parquet|feather))",
                command,
                re.IGNORECASE,
            )
            if pred_path_m:
                raw_pred_path = pred_path_m.group(1).strip()
                if not is_within_allowed_dirs(Path(raw_pred_path), self.agent.search_dirs):
                    return {
                        "status": "error",
                        "error": f"Blocked: '{raw_pred_path}' is outside the allowed dataset folders.",
                    }

        # B3.5: Pre-mutation confirmation gate for data-changing commands or model overwrite
        is_mutating = bool(_MUTATING_CMD_RE.search(cmd_low))
        model_file = os.path.join(s.output_dir, "model.joblib")
        is_model_overwrite = bool(re.search(r"\b(save|export)\s+(?:the\s+)?model\b", cmd_low)) and os.path.exists(model_file)

        if (is_mutating or is_model_overwrite) and not confirmed:
            reason = (
                "A saved model already exists and would be overwritten."
                if is_model_overwrite
                else f"This command ('{command}') modifies the dataset."
            )
            return {
                "status": "needs_confirmation",
                "command": command,
                "question": f"{reason} Ask the user to confirm first, and mention that 'undo' is available.",
            }

        before_ver = s.wrangler.version if s.wrangler is not None else 0
        had_training_before = s.training_result is not None

        if runner is not None and is_heavy_command(command):
            reply, cancelled = runner.execute(command)
            if cancelled:
                return {"status": "cancelled", "message": "Stopped the running task."}
            s = self.agent.assistant.session
        else:
            reply = self.agent.assistant.ask(command, strict_security=True)

        after_ver = s.wrangler.version if s.wrangler is not None else 0
        data_changed = after_ver != before_ver

        if (
            "i'm not sure how to do that yet" in reply.lower()
            or reply.startswith("Could not load model:")
            or reply.startswith("Blocked:")
            or reply.startswith("Access denied:")
        ):
            return {"status": "error", "error": reply}

        if s.training_result is not None and reply == s.training_result.best_reason:
            reply = _measured_why_winner(reply) or reply

        out: dict[str, Any] = {
            "status": "ok",
            "command": command,
            "summary": reply,
            "rows": int(s.raw_df.shape[0]),
            "cols": int(s.raw_df.shape[1]),
            "warnings": [],
        }

        if is_mutating:
            out["undo_available"] = data_changed
            out["unchanged"] = not data_changed

        # Attach structured training / explainability fields whenever training exists or just ran
        if s.training_result is not None and (
            not had_training_before
            or any(w in cmd_low for w in ("train", "build", "fit", "tune", "why", "importan", "matters", "evaluat", "leaderboard"))
        ):
            out.update(self._training_payload(s))

        # B5.4: Flag forecast that fails to outperform naive persistence baseline
        if "model performance is close to naive persistence" in reply.lower():
            out["warnings"].append(
                "CAUTION: Forecast did not beat the naive baseline; advise the user to treat it with caution."
            )

        return out

    def _tool_run_full_pipeline(self, args: dict[str, Any], runner: Optional[Any] = None) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset is loaded yet."}
        if not s.target:
            return {
                "status": "error",
                "error": "Set a target column first before running the full pipeline.",
                "columns": [wrap_untrusted(c) for c in s.raw_df.columns[:15]],
            }

        model_file = os.path.join(s.output_dir, "model.joblib")
        if os.path.exists(model_file) and not bool(args.get("confirmed", False)):
            return {
                "status": "needs_confirmation",
                "question": "Running the full pipeline will overwrite the existing saved model. Ask the user to confirm first.",
            }

        cmd = "run full pipeline"
        if runner is not None:
            reply, cancelled = runner.execute(cmd)
            if cancelled:
                return {"status": "cancelled", "message": "Pipeline stopped."}
            s = self.agent.assistant.session
        else:
            reply = self.agent.assistant.ask(cmd)

        payload: dict[str, Any] = {
            "status": "ok",
            "summary": reply,
            "reports": dict(s.report_paths),
        }
        payload.update(self._training_payload(s))
        return payload

    def _tool_get_status(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {
                "status": "ok",
                "loaded": False,
                "discoverable_files": [wrap_untrusted(f) for f in self._list_discoverable_files()],
            }
        tr = s.training_result
        return {
            "status": "ok",
            "loaded": True,
            "file": wrap_untrusted(os.path.basename(self.agent.current_file or "dataset")),
            "rows": int(s.raw_df.shape[0]),
            "cols": int(s.raw_df.shape[1]),
            "columns": [wrap_untrusted(c) for c in s.raw_df.columns],
            "target": s.target,
            "cleaned": s.cleaning_result is not None,
            "engineered_features_active": self._has_non_rebuildable_engineered_features(s),
            "best_model": tr.best_name if tr else None,
            "models_compared": len(tr.leaderboard) if tr and tr.leaderboard else 0,
            "cv_metric": tr.scoring if tr else None,
            "last_cv_score": round(tr.leaderboard[0].cv_score, 4) if tr and tr.leaderboard else None,
            "reports": dict(s.report_paths),
        }

    def _tool_open_report(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset is loaded yet."}
        open_browser = bool(args.get("open", True))
        reply = self.agent.assistant.ask("show me the html report", format="html", open_browser=open_browser)
        return {"status": "ok", "report_paths": dict(s.report_paths), "summary": reply}

    def _tool_undo(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset is loaded yet."}
        reply = self.agent.assistant.ask("undo")
        return {
            "status": "ok",
            "rows": int(s.raw_df.shape[0]),
            "cols": int(s.raw_df.shape[1]),
            "summary": reply,
        }

    def _tool_stop(self, args: dict[str, Any], runner: Optional[Any] = None) -> dict[str, Any]:
        cancelled = False
        if runner is not None:
            cancelled = bool(runner.cancel())
        return {"status": "ok", "stopped": True, "cancelled_job": cancelled}

    # ------------------------------------------------------------------ #
    # Compatibility Tool Aliases (Safe SmartML Core Calls Only)
    # ------------------------------------------------------------------ #

    def _tool_profile_data(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        rep = analyzer.analyze(s.raw_df, target=s.target)
        s.dataset_report = rep
        missing_cols = {k: int(v) for k, v in rep.missing_counts.items() if v > 0}
        warnings = []
        if rep.n_rows < 500:
            warnings.append(f"SMALL DATASET NOTE: Dataset has {rep.n_rows} rows; scores may not be stable.")
        if rep.data_quality_score < 60:
            warnings.append(f"Low data quality score ({rep.data_quality_score}/100); clean before training.")
        return {
            "status": "ok",
            "rows": rep.n_rows,
            "cols": rep.n_cols,
            "numeric_cols": len(rep.numeric_cols),
            "categorical_cols": len(rep.categorical_cols),
            "datetime_cols": len(rep.datetime_cols),
            "total_missing": rep.total_missing,
            "missing_by_column": missing_cols,
            "duplicate_rows": rep.duplicate_rows,
            "quality_score": rep.data_quality_score,
            "target": s.target,
            "inferred_task": rep.task,
            "warnings": warnings,
        }

    def _tool_clean_data(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        reply = self.agent.assistant.ask("clean my dataset")
        steps = s.cleaning_result.steps if s.cleaning_result else []
        return {
            "status": "ok",
            "cleaned_rows": int(s.df.shape[0]),
            "cleaned_cols": int(s.df.shape[1]),
            "steps": steps,
            "summary": reply,
        }

    def _tool_eda(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        prof = self._tool_profile_data(args)
        cols = args.get("columns")
        col_summaries = {}
        if cols:
            col_list = [cols] if isinstance(cols, str) else list(cols)
            for c in col_list[:5]:
                matched = textutil.fuzzy_match_column(str(c), s.raw_df.columns)
                if matched:
                    cs = sml_stats.summarize_column(s.raw_df, matched)
                    col_summaries[matched] = {
                        "dtype": cs.dtype,
                        "missing": cs.missing,
                        "mean": cs.mean,
                        "median": cs.median,
                        "std": cs.std,
                        "skewness": cs.skewness,
                    }
        out_dir = os.path.join(s.output_dir, "plots")
        try:
            corr_path = visualizer.plot_correlation_heatmap(s.raw_df, out_dir)
            s.visualizations["Correlation Heatmap"] = corr_path
        except Exception:
            corr_path = None
        return {
            "status": "ok",
            "profile": prof,
            "column_summaries": col_summaries,
            "plot_saved": corr_path,
        }

    def _tool_engineer_features(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        reply = self.agent.assistant.ask("engineer features")
        new_feats = s.feature_engineering_result.new_features if s.feature_engineering_result else []
        return {
            "status": "ok",
            "new_features_count": len(new_feats),
            "new_features": new_feats[:15],
            "summary": reply,
        }

    def _tool_train_automl(self, args: dict[str, Any], runner: Optional[Any] = None) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}

        target_arg = args.get("target")
        if target_arg:
            t_res = self._tool_set_target({"column": target_arg, "confirmed": bool(args.get("confirmed", False))})
            if t_res.get("status") != "ok":
                return t_res

        if not s.target:
            return {
                "status": "error",
                "error": "Missing target column.",
                "columns": [wrap_untrusted(c) for c in s.raw_df.columns[:15]],
            }

        task_arg = str(args.get("task") or "").lower()
        cmd = "train the best model"
        if "classif" in task_arg:
            cmd = "train the best classification model"
        elif "regress" in task_arg:
            cmd = "train the best regression model"

        if runner is not None:
            reply, cancelled = runner.execute(cmd)
            if cancelled:
                return {"status": "cancelled", "message": "Training cancelled by user."}
            s = self.agent.assistant.session
        else:
            reply = self.agent.assistant.ask(cmd)

        if s.training_result is None:
            return {"status": "error", "error": reply}

        out = {"status": "ok"}
        out.update(self._training_payload(s))
        return out

    def _tool_compare_models(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        if s.training_result is None:
            return self._tool_train_automl(args)
        out = {"status": "ok"}
        out.update(self._training_payload(s))
        return out

    def _tool_tune(self, args: dict[str, Any], runner: Optional[Any] = None) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        if s.training_result is None:
            return {"status": "error", "error": "No model trained yet."}
        model_name = str(args.get("model") or s.training_result.best_name)
        return self._tool_ask_smartml({"command": f"tune {model_name}", "confirmed": True}, runner=runner)

    def _tool_evaluate(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None or s.training_result is None:
            return {"status": "error", "error": "Train a model first before evaluating."}
        out = {"status": "ok"}
        out.update(self._training_payload(s))
        return out

    def _tool_explain(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None or s.training_result is None:
            return {"status": "error", "error": "Train a model first so I can explain it."}
        self.agent.assistant.ask("show feature importance")
        imp = s.importance_result
        top_dict = (
            {redact_identifier(k, kind="feature"): round(float(v), 4) for k, v in imp.importances.head(8).items()}
            if imp is not None
            else {}
        )
        return {
            "status": "ok",
            "best_model": s.training_result.best_name,
            "models_compared": len(s.training_result.leaderboard),
            "why_winner": _measured_why_winner(s.training_result.best_reason),
            "method": imp.method if imp else "permutation",
            "top_features": top_dict,
        }

    def _tool_predict(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No session active."}
        if self._has_non_rebuildable_engineered_features(s):
            return {
                "status": "error",
                "error": (
                    "Cannot predict yet because the trained model uses engineered features "
                    "that cannot be rebuilt automatically for new data."
                ),
            }
        inp = args.get("input") or args.get("file_path") or args.get("path")
        if isinstance(inp, dict):
            row_df = pd.DataFrame([inp])
            reply = self.agent.assistant.ask("predict on this data", data=row_df)
        elif isinstance(inp, str) and inp.strip():
            cands = find_dataset_candidates(inp.strip(), search_dirs=self.agent.search_dirs)
            if not cands:
                return {"status": "error", "error": f"Input file '{inp}' not found in allowed folders."}
            reply = self.agent.assistant.ask(f"predict on {cands[0]}", file_path=cands[0])
        else:
            return {"status": "error", "error": "Provide a file path in allowed folders to predict on."}
        return {"status": "ok", "summary": reply}

    def _tool_plot(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        kind = str(args.get("kind") or "all")
        reply = self.agent.assistant.ask(f"plot {kind}")
        return {"status": "ok", "plots": dict(s.visualizations), "summary": reply}

    def _tool_generate_report(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None:
            return {"status": "error", "error": "No dataset loaded yet."}
        fmt = str(args.get("format") or "html").lower()
        open_browser = bool(args.get("open", True))
        cmd = f"show me the {fmt} report" if open_browser else f"generate {fmt} report"
        reply = self.agent.assistant.ask(cmd, format=fmt, open_browser=open_browser)
        return {"status": "ok", "report_paths": dict(s.report_paths), "summary": reply}

    def _tool_export_model(self, args: dict[str, Any]) -> dict[str, Any]:
        s = self._require_session()
        if s is None or s.training_result is None:
            return {"status": "error", "error": "No trained model to export yet."}
        dest = os.path.join(s.output_dir, "model.joblib")
        if os.path.exists(dest) and not bool(args.get("confirmed", False)):
            return {
                "status": "needs_confirmation",
                "question": "A saved model already exists. Ask the user if you should overwrite it.",
            }
        reply = self.agent.assistant.ask("save model")
        return {"status": "ok", "path": dest, "summary": reply}


