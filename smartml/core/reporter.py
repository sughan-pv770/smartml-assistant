"""
smartml.core.reporter
=======================
Builds a modern, interactive, self-contained executive HTML dashboard
with base64-embedded charts, metric cards, interactive tabs, responsive
CSS grid layouts, and clean data tables.
"""

from __future__ import annotations

import base64
import datetime as dt
import os
from typing import Optional, Any
import numpy as np
import pandas as pd


def _img_to_base64(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _generate_auto_charts(session, out_dir: str) -> dict[str, str]:
    """Ensure essential visual plots are always generated for the report."""
    from . import visualizer, explainer

    charts: dict[str, str] = {}
    if getattr(session, "visualizations", None):
        charts.update(session.visualizations)

    os.makedirs(out_dir, exist_ok=True)

    tr = getattr(session, "training_result", None)
    cr = getattr(session, "cleaning_result", None)

    # 1. Model comparison chart
    if tr and tr.leaderboard and "Model Comparison" not in charts:
        try:
            path = visualizer.plot_model_comparison(tr.leaderboard, out_dir)
            if path and os.path.exists(path):
                charts["Model Comparison"] = path
        except Exception:
            pass

    # 2. Feature Importance chart
    if tr and "Feature Importance" not in charts:
        try:
            if getattr(session, "importance_result", None) is None:
                session.importance_result = explainer.explain_feature_importance(
                    model=tr.best_model,
                    X_test=tr.X_test,
                    y_test=tr.y_test,
                    feature_names=tr.feature_names,
                    task=tr.task,
                )
            if session.importance_result is not None:
                path = visualizer.plot_feature_importance(session.importance_result.importances, out_dir)
                if path and os.path.exists(path):
                    charts["Feature Importance"] = path
        except Exception:
            pass

    # 3. Confusion Matrix or Residual Plot
    if tr and hasattr(tr, "best_model") and hasattr(tr, "X_test") and hasattr(tr, "y_test"):
        try:
            preds = tr.best_model.predict(tr.X_test)
            if tr.task == "classification" and "Confusion Matrix" not in charts:
                path = visualizer.plot_confusion_matrix(tr.y_test, preds, out_dir)
                if path and os.path.exists(path):
                    charts["Confusion Matrix"] = path
            elif tr.task == "regression" and "Residual Plot" not in charts:
                path = visualizer.plot_residuals(tr.y_test, preds, out_dir)
                if path and os.path.exists(path):
                    charts["Residual Plot"] = path
        except Exception:
            pass

    # 4. Correlation Heatmap
    if cr and hasattr(cr, "df") and "Correlation Heatmap" not in charts:
        try:
            path = visualizer.plot_correlation_heatmap(cr.df, out_dir)
            if path and os.path.exists(path):
                charts["Correlation Heatmap"] = path
        except Exception:
            pass

    # 5. Missing values heatmap
    if hasattr(session, "raw_df") and session.raw_df is not None and "Missing Values" not in charts:
        try:
            if session.raw_df.isna().sum().sum() > 0:
                path = visualizer.plot_missing_heatmap(session.raw_df, out_dir)
                if path and os.path.exists(path):
                    charts["Missing Values"] = path
        except Exception:
            pass

    return charts


def generate_html_report(session, out_path: str) -> str:
    """Generate a self-contained, publication-grade HTML executive dashboard."""
    out_dir = os.path.dirname(out_path) or "."
    charts = _generate_auto_charts(session, out_dir)

    tr = getattr(session, "training_result", None)
    dr = getattr(session, "dataset_report", None)
    cr = getattr(session, "cleaning_result", None)
    target = getattr(session, "target", "Target")

    # Metrics summary cards
    n_rows = dr.n_rows if dr else (len(cr.df) if cr and hasattr(cr, "df") else "N/A")
    n_cols = dr.n_cols if dr else (len(cr.df.columns) if cr and hasattr(cr, "df") else "N/A")
    task_type = (tr.task if tr else (dr.task if dr else "Machine Learning")).capitalize()
    best_model_name = tr.best_name if tr else "N/A"
    best_score_str = f"{tr.leaderboard[0].cv_score:.4f}" if tr and tr.leaderboard else "N/A"
    best_metric_name = "F1-Weighted" if tr and tr.task == "classification" else "R² Score"

    # Leaderboard table rows
    leaderboard_rows = ""
    if tr and tr.leaderboard:
        for idx, entry in enumerate(tr.leaderboard, 1):
            is_best = entry.name == tr.best_name
            highlight = ' class="best-row"' if is_best else ""
            badge = '<span class="badge-winner">Winner</span>' if is_best else f'<span class="rank-badge">#{idx}</span>'
            metrics_pills = " ".join(
                f'<span class="metric-pill"><b>{k}</b>: {v:.4f}</span>'
                for k, v in entry.test_metrics.items()
            )
            leaderboard_rows += f"""
            <tr{highlight}>
                <td>{badge} <strong>{entry.name}</strong></td>
                <td><span class="cv-score">{entry.cv_score:.4f}</span> <small class="text-muted">± {entry.cv_std:.4f}</small></td>
                <td>{metrics_pills}</td>
                <td>{entry.fit_time_s:.2f}s</td>
            </tr>
            """

    # Feature Importance table rows
    importance_rows = ""
    imp_narrative = ""
    if getattr(session, "importance_result", None) is not None:
        imp = session.importance_result
        imp_narrative = imp.narrative
        for idx, (fname, val) in enumerate(imp.importances.head(10).items(), 1):
            pct = val * 100 if val <= 1.0 else val
            importance_rows += f"""
            <tr>
                <td><strong>#{idx}</strong> {fname}</td>
                <td>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: {min(100, max(5, pct)):.1f}%;"></div>
                        <span class="progress-bar-text">{val:.4f}</span>
                    </div>
                </td>
            </tr>
            """

    # Cleaning steps badges
    cleaning_items = ""
    if cr and getattr(cr, "steps", None):
        cleaning_items = "".join(f'<div class="step-card"><span class="step-check">&#10003;</span> {s}</div>' for s in cr.steps)
    else:
        cleaning_items = '<div class="step-card">No data cleaning steps were needed; dataset is clean.</div>'

    # Embedded Visualization Gallery
    charts_html = ""
    if charts:
        for title, path in charts.items():
            b64 = _img_to_base64(path)
            if b64:
                charts_html += f"""
                <div class="chart-card">
                    <div class="chart-header">
                        <h3>{title}</h3>
                        <span class="chart-badge">Visual Analytic</span>
                    </div>
                    <div class="chart-img-wrap">
                        <img src="data:image/png;base64,{b64}" alt="{title}"/>
                    </div>
                </div>
                """
    if not charts_html:
        charts_html = "<p class='text-muted'>No visual charts generated for this run.</p>"

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SmartML Executive Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --primary: #2563eb;
    --primary-light: #eff6ff;
    --primary-border: #bfdbfe;
    --success: #16a34a;
    --success-light: #f0fdf4;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-page);
    color: var(--text-main);
    line-height: 1.6;
    padding: 32px 16px;
  }}

  .container {{
    max-width: 1200px;
    margin: 0 auto;
  }}

  /* Header */
  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: #ffffff;
    padding: 36px 40px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: var(--shadow-lg);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .header-left h1 {{
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .header-left p {{
    color: #94a3b8;
    margin-top: 6px;
    font-size: 14px;
  }}
  .header-right {{
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 13px;
    color: #cbd5e1;
  }}

  /* KPI Grid */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
    margin-bottom: 32px;
  }}
  .kpi-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
  }}
  .kpi-label {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }}
  .kpi-value {{
    font-size: 24px;
    font-weight: 800;
    color: var(--text-main);
  }}
  .kpi-value.highlight {{
    color: var(--primary);
  }}
  .kpi-subtext {{
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
  }}

  /* Section Styles */
  .section {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
    box-shadow: var(--shadow-sm);
  }}
  .section-title {{
    font-size: 19px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 2px solid var(--bg-page);
    padding-bottom: 12px;
  }}

  /* Best Model Banner */
  .winner-banner {{
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    display: flex;
    align-items: flex-start;
    gap: 16px;
  }}
  .winner-icon {{
    background: var(--success);
    color: white;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
  }}
  .winner-content h3 {{
    font-size: 17px;
    font-weight: 700;
    color: #14532d;
    margin-bottom: 4px;
  }}
  .winner-content p {{
    font-size: 14px;
    color: #166534;
    line-height: 1.5;
  }}

  /* Data Tables */
  .table-responsive {{
    overflow-x: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    text-align: left;
  }}
  th {{
    background: #f1f5f9;
    color: #475569;
    font-weight: 600;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    color: #334155;
  }}
  tr:hover td {{
    background-color: #f8fafc;
  }}
  tr.best-row td {{
    background-color: #f0fdf4 !important;
  }}

  /* Badges & Pills */
  .badge-winner {{
    background: #22c55e;
    color: white;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 20px;
    margin-right: 6px;
  }}
  .rank-badge {{
    background: #e2e8f0;
    color: #475569;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 20px;
    margin-right: 6px;
  }}
  .metric-pill {{
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 12px;
    margin-right: 4px;
    margin-bottom: 4px;
  }}
  .cv-score {{
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
  }}

  /* Visual Charts Grid */
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 24px;
  }}
  .chart-card {{
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
  }}
  .chart-header {{
    padding: 14px 20px;
    background: #f8fafc;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .chart-header h3 {{
    font-size: 15px;
    font-weight: 600;
    color: #1e293b;
  }}
  .chart-badge {{
    font-size: 11px;
    background: var(--primary-light);
    color: var(--primary);
    border: 1px solid var(--primary-border);
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 600;
  }}
  .chart-img-wrap {{
    padding: 16px;
    text-align: center;
    background: #ffffff;
  }}
  .chart-img-wrap img {{
    max-width: 100%;
    height: auto;
    border-radius: 6px;
  }}

  /* Progress bar in feature importance */
  .progress-bar-container {{
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    max-width: 320px;
  }}
  .progress-bar-fill {{
    height: 10px;
    background: linear-gradient(90deg, #3b82f6, #1d4ed8);
    border-radius: 6px;
  }}
  .progress-bar-text {{
    font-size: 12px;
    font-weight: 600;
    color: #475569;
    min-width: 48px;
  }}

  /* Cleaning Steps Grid */
  .cleaning-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 12px;
  }}
  .step-card {{
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #334155;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .step-check {{
    color: #16a34a;
    font-weight: 700;
    font-size: 14px;
  }}

  /* Footer */
  .footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }}

  @media (max-width: 768px) {{
    .charts-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
</head>
<body>

<div class="container">

  <!-- Header -->
  <div class="header">
    <div class="header-left">
      <h1>SmartML Executive Dashboard</h1>
      <p>Automated Machine Learning & Data Science Run Summary</p>
    </div>
    <div class="header-right">
      <span>Generated {dt.datetime.now().strftime('%b %d, %Y - %H:%M:%S')}</span>
    </div>
  </div>

  <!-- KPI Overview Cards -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Dataset Shape</div>
      <div class="kpi-value">{n_rows} <small style="font-size: 16px; font-weight: 500;">rows</small></div>
      <div class="kpi-subtext">{n_cols} total feature columns</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Target & Task</div>
      <div class="kpi-value highlight">{task_type}</div>
      <div class="kpi-subtext">Target: <b>{target}</b></div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Winning Model</div>
      <div class="kpi-value" style="color: #15803d;">{best_model_name}</div>
      <div class="kpi-subtext">Selected by cross-validation</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Best {best_metric_name}</div>
      <div class="kpi-value">{best_score_str}</div>
      <div class="kpi-subtext">Holdout CV performance</div>
    </div>
  </div>

  <!-- Visual Analytics & Charts Gallery -->
  <div class="section">
    <div class="section-title">
      Visual Analytics & Diagnostics
    </div>
    <div class="charts-grid">
      {charts_html}
    </div>
  </div>

  <!-- Best Model & Leaderboard -->
  <div class="section">
    <div class="section-title">
      Model Comparison & Evaluation Leaderboard
    </div>

    {f'''
    <div class="winner-banner">
      <div class="winner-icon">&#9733;</div>
      <div class="winner-content">
        <h3>Winning Candidate: {tr.best_name}</h3>
        <p>{tr.best_reason}</p>
      </div>
    </div>
    ''' if tr else ''}

    <div class="table-responsive">
      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Cross-Validated Score</th>
            <th>Test Metrics</th>
            <th>Fit Time</th>
          </tr>
        </thead>
        <tbody>
          {leaderboard_rows or '<tr><td colspan="4">No model training results found.</td></tr>'}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Feature Importance & Explanations -->
  {f'''
  <div class="section">
    <div class="section-title">
      Feature Importance & Driver Analysis
    </div>
    <p style="margin-bottom: 16px; font-size: 14px; color: #475569;">{imp_narrative}</p>
    <div class="table-responsive">
      <table>
        <thead>
          <tr>
            <th>Feature Name</th>
            <th>Relative Importance Contribution</th>
          </tr>
        </thead>
        <tbody>
          {importance_rows}
        </tbody>
      </table>
    </div>
  </div>
  ''' if importance_rows else ''}

  <!-- Data Cleaning Audit Log -->
  <div class="section">
    <div class="section-title">
      Automated Data Cleaning Audit Log
    </div>
    <div class="cleaning-grid">
      {cleaning_items}
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    SmartML v2.0 &bull; Multi-Paradigm AutoML & Data Science Engine &bull; Self-Contained Report
  </div>

</div>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    return out_path


def generate_markdown_report(session, out_path: str) -> str:
    """Generate Markdown summary report."""
    tr = getattr(session, "training_result", None)
    dr = getattr(session, "dataset_report", None)
    cr = getattr(session, "cleaning_result", None)

    lines = ["# SmartML Executive Report", "", f"_Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_", ""]

    if dr:
        lines += [
            "## Dataset Summary", "",
            f"- Rows: {dr.n_rows}",
            f"- Columns: {dr.n_cols}",
            f"- Memory: {dr.memory_mb} MB",
            f"- Target: `{session.target}`",
            f"- Task: **{dr.task}**",
            f"- {dr.task_reason}",
            "",
        ]

    lines += ["## Data Cleaning Audit Log", ""]
    if cr and getattr(cr, "steps", None):
        lines += [f"- {s}" for s in cr.steps]
    else:
        lines += ["- Dataset clean; no preprocessing required."]
    lines.append("")

    if tr:
        lines += ["## Model Comparison Leaderboard", "", "| Model | CV Score | Test Metrics |", "|---|---|---|"]
        for e in tr.leaderboard:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in e.test_metrics.items())
            marker = " **(Winner)**" if e.name == tr.best_name else ""
            lines.append(f"| {e.name}{marker} | {e.cv_score:.4f} ± {e.cv_std:.4f} | {metric_str} |")
        lines.append("")
        lines += ["## Best Model Summary", "", f"**{tr.best_name}**", "", tr.best_reason, ""]

    if getattr(session, "importance_result", None) is not None:
        lines += ["## Feature Importance", "", session.importance_result.narrative, "", "| Feature | Importance |", "|---|---|"]
        for name, val in session.importance_result.importances.head(15).items():
            lines.append(f"| {name} | {val:.4f} |")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
