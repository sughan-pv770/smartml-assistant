"""
smartml.core.reporter
=======================
Builds an HTML (self-contained, base64-embedded images) or Markdown
report summarizing the full session: dataset, cleaning, model comparison,
explainability, and visualizations.
"""

from __future__ import annotations

import base64
import datetime as dt
import os
from typing import Optional


def _img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_html_report(session, out_path: str) -> str:
    """`session` is a smartml.assistant.Session (duck-typed)."""
    rows = ""
    if session.training_result:
        for e in session.training_result.leaderboard:
            metric_str = ", ".join(f"{k}: {v:.4f}" for k, v in e.test_metrics.items())
            highlight = ' style="background:#e8f5e9;font-weight:600;"' if e.name == session.training_result.best_name else ""
            rows += (
                f"<tr{highlight}><td>{e.name}</td><td>{e.cv_score:.4f} ± {e.cv_std:.4f}</td>"
                f"<td>{metric_str}</td></tr>"
            )

    cleaning_steps = "".join(f"<li>{s}</li>" for s in (session.cleaning_result.steps if session.cleaning_result else []))

    images_html = ""
    for label, path in session.visualizations.items():
        if path and os.path.exists(path):
            b64 = _img_to_base64(path)
            images_html += f'<h3>{label}</h3><img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:6px;margin-bottom:24px;"/>'

    importance_html = ""
    if session.importance_result is not None:
        rows_imp = "".join(
            f"<tr><td>{name}</td><td>{val:.4f}</td></tr>"
            for name, val in session.importance_result.importances.head(15).items()
        )
        importance_html = f"""
        <h2>Feature Importance</h2>
        <p>{session.importance_result.narrative}</p>
        <table><tr><th>Feature</th><th>Importance</th></tr>{rows_imp}</table>
        """

    best_section = ""
    if session.training_result:
        best_section = f"""
        <h2>Best Model</h2>
        <p><strong>{session.training_result.best_name}</strong></p>
        <p>{session.training_result.best_reason}</p>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SmartML Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 920px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.55; }}
  h1 {{ border-bottom: 3px solid #3b6ea5; padding-bottom: 8px; }}
  h2 {{ color: #2c4a6e; margin-top: 36px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #3b6ea5; color: white; }}
  tr:nth-child(even) {{ background: #f7f9fb; }}
  .meta {{ color: #666; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>SmartML Report</h1>
<p class="meta">Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<h2>Dataset Summary</h2>
<table>
<tr><th>Rows</th><td>{session.dataset_report.n_rows}</td></tr>
<tr><th>Columns</th><td>{session.dataset_report.n_cols}</td></tr>
<tr><th>Memory</th><td>{session.dataset_report.memory_mb} MB</td></tr>
<tr><th>Target</th><td>{session.target}</td></tr>
<tr><th>Task</th><td>{session.dataset_report.task}</td></tr>
</table>
<p>{session.dataset_report.task_reason}</p>

<h2>Cleaning Steps</h2>
<ul>{cleaning_steps or '<li>No cleaning has been run yet.</li>'}</ul>

<h2>Model Comparison</h2>
<table><tr><th>Model</th><th>CV Score</th><th>Test Metrics</th></tr>{rows or ''}</table>

{best_section}
{importance_html}

<h2>Visualizations</h2>
{images_html or '<p>No visualizations generated yet.</p>'}

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def generate_markdown_report(session, out_path: str) -> str:
    lines = ["# SmartML Report", "", f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_", ""]

    lines += [
        "## Dataset Summary", "",
        f"- Rows: {session.dataset_report.n_rows}",
        f"- Columns: {session.dataset_report.n_cols}",
        f"- Memory: {session.dataset_report.memory_mb} MB",
        f"- Target: `{session.target}`",
        f"- Task: **{session.dataset_report.task}**",
        f"- {session.dataset_report.task_reason}",
        "",
    ]

    lines += ["## Cleaning Steps", ""]
    if session.cleaning_result:
        lines += [f"- {s}" for s in session.cleaning_result.steps] or ["- No cleaning steps were required."]
    else:
        lines += ["- No cleaning has been run yet."]
    lines.append("")

    if session.training_result:
        lines += ["## Model Comparison", "", "| Model | CV Score | Test Metrics |", "|---|---|---|"]
        for e in session.training_result.leaderboard:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in e.test_metrics.items())
            marker = " **(best)**" if e.name == session.training_result.best_name else ""
            lines.append(f"| {e.name}{marker} | {e.cv_score:.4f} ± {e.cv_std:.4f} | {metric_str} |")
        lines.append("")
        lines += ["## Best Model", "", f"**{session.training_result.best_name}**", "", session.training_result.best_reason, ""]

    if session.importance_result is not None:
        lines += ["## Feature Importance", "", session.importance_result.narrative, "", "| Feature | Importance |", "|---|---|"]
        for name, val in session.importance_result.importances.head(15).items():
            lines.append(f"| {name} | {val:.4f} |")
        lines.append("")

    if session.visualizations:
        lines += ["## Visualizations", ""]
        for label, path in session.visualizations.items():
            lines.append(f"- {label}: `{path}`")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
