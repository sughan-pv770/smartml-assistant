import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import smartml as sml
from smartml.core import explainer, visualizer

out_dir = r"f:\smartml_assistant_module\linkedin_showcase"
os.makedirs(out_dir, exist_ok=True)

# Load real dataset
csv_path = r"f:\smartml_assistant_module\archive\AI_Usage_and_Impact_on_Students_and_Professionals.csv"
df = sml.load(csv_path)

# ==========================================
# 1. Image 1: Raw Input Dataset Preview Table
# ==========================================
fig, ax = plt.subplots(figsize=(13, 6), dpi=220)
ax.axis('tight')
ax.axis('off')
sample_cols = ['User_Type', 'Country', 'Education_Level', 'AI_Tool', 'Productivity_Score', 'Satisfaction_Score', 'Would_Recommend']
preview_df = df[sample_cols].head(8)
table = ax.table(cellText=preview_df.values, colLabels=preview_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1.15, 1.7)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#0f172a')
        cell.set_text_props(color='#38bdf8', weight='bold')
    else:
        cell.set_facecolor('#f8fafc' if row % 2 == 0 else '#ffffff')
plt.title('Image 1: Input Dataset Preview (300 Rows, 19 Multimodal Features)', fontsize=14, weight='bold', pad=24, color='#0f172a')
p1 = os.path.join(out_dir, '1_input_dataset_preview.png')
plt.savefig(p1, bbox_inches='tight')
plt.close()

# ==========================================
# 2. Image 2: Minimalist 1-Line Python API Card
# ==========================================
fig, ax = plt.subplots(figsize=(11, 6), dpi=220)
ax.axis('off')
code_lines = [
    "# === SmartML v2.0: 1-Line Machine Learning & Data Science ===",
    "",
    "import smartml as sml",
    "",
    "# 1. Load any dataset (CSV, Excel, Parquet, JSON)",
    'df = sml.load("student_ai_impact.csv")',
    "",
    "# 2. Automatically profile, clean, tune, & compare 9+ ML models",
    'result = sml.auto(df, target="Would_Recommend")',
    "",
    "# 3. Print candidate leaderboard and natural language explanation",
    "print(result.summary)",
    "print(result.explain())",
    "",
    "# 4. Generate self-contained interactive visual HTML dashboard",
    'result.report(fmt="html")',
]
code_text = "\n".join(code_lines)
ax.text(
    0.05, 0.5, code_text, fontsize=12, family='monospace', verticalalignment='center',
    bbox=dict(boxstyle='round,pad=1.4', facecolor='#090d16', edgecolor='#38bdf8', linewidth=2),
    color='#f1f5f9'
)
plt.title('Image 2: Minimalist 1-Line Python Code', fontsize=14, weight='bold', pad=15, color='#0f172a')
p2 = os.path.join(out_dir, '2_python_code_snippet.png')
plt.savefig(p2, bbox_inches='tight')
plt.close()

# Train model to get live real output metrics
result = sml.auto(df, target='Would_Recommend', output_dir=out_dir)
tr = result.training_result

# ==========================================
# 3. Image 3: Model Comparison Leaderboard Chart
# ==========================================
names = [e.name for e in tr.leaderboard]
scores = [e.cv_score for e in tr.leaderboard]
stds = [e.cv_std for e in tr.leaderboard]
fig, ax = plt.subplots(figsize=(11, 6), dpi=220)
colors = ['#16a34a' if i == 0 else '#64748b' for i in range(len(names))]
y_pos = range(len(names))
bars = ax.barh(y_pos, scores[::-1], xerr=stds[::-1], color=colors[::-1], capsize=4, edgecolor='#1e293b', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(names[::-1], fontsize=10, weight='600')
ax.set_xlabel('Holdout Cross-Validated Score (Weighted F1)', fontsize=11, weight='600')
ax.set_title(f'Image 3: AutoML Model Comparison Leaderboard (Winner: {tr.best_name})', fontsize=13, weight='bold', pad=14, color='#0f172a')
ax.grid(axis='x', linestyle='--', alpha=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.015, bar.get_y() + bar.get_height()/2, f'{w:.4f}', ha='left', va='center', fontsize=9.5, weight='bold', color='#0f172a')
plt.tight_layout()
p3 = os.path.join(out_dir, '3_model_leaderboard_chart.png')
plt.savefig(p3, bbox_inches='tight')
plt.close()

# ==========================================
# 4. Image 4: Feature Importance Explainer Plot
# ==========================================
imp_res = explainer.explain_feature_importance(tr.best_model, tr.X_test, tr.y_test, tr.feature_names, tr.task)
top_imp = imp_res.importances.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=220)
bars = ax.barh(top_imp.index, top_imp.values, color='#2563eb', edgecolor='#1e40af', linewidth=0.8)
ax.set_xlabel('Relative Predictive Contribution', fontsize=10, weight='600')
ax.set_title('Image 4: Top Predictive Features Driving Predictions', fontsize=13, weight='bold', pad=14, color='#0f172a')
ax.grid(axis='x', linestyle='--', alpha=0.4)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.001, bar.get_y() + bar.get_height()/2, f'{w:.4f}', ha='left', va='center', fontsize=9.5, weight='bold', color='#0f172a')
plt.tight_layout()
p4 = os.path.join(out_dir, '4_feature_importance.png')
plt.savefig(p4, bbox_inches='tight')
plt.close()

# ==========================================
# 5. Image 5: Confusion Matrix Heatmap
# ==========================================
preds = tr.best_model.predict(tr.X_test)
p5_orig = visualizer.plot_confusion_matrix(tr.y_test, preds, out_dir)
p5 = os.path.join(out_dir, '5_confusion_matrix_diagnostics.png')
if os.path.exists(p5_orig):
    shutil.copy(p5_orig, p5)

# ==========================================
# 6. Image 6: Feature Correlation Heatmap
# ==========================================
p6_orig = visualizer.plot_correlation_heatmap(df, out_dir)
p6 = os.path.join(out_dir, '6_correlation_heatmap.png')
if os.path.exists(p6_orig):
    shutil.copy(p6_orig, p6)

print("SUCCESS: 6 high-res showcase images generated successfully!")
