"""Append §12 'Re-run rotations after relabel' to honest_eval_and_improve.ipynb.
One-shot — safe to delete after running.
"""
import json
from pathlib import Path

NB = Path(__file__).with_name('honest_eval_and_improve.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

md_src = """## 12. Re-run rotations after audios5 relabel

Workflow:
1. Make sure §3 (the original rotations A and B) has been run at least once with the **original** `audios5GT.csv` — that produces the baseline this cell compares against.
2. Re-listen to the suspect rows in `checkpoints_honest_eval/a5_label_audit_negatives.csv` and edit `audios5GT.csv` directly: change the label column from `not cheating` to `cheating` (or whatever the project convention is) for any audio you confirmed is mislabeled.
3. Run this cell. It will:
   - back up the original `audios5GT.csv` (once, on first run) to `audios5GT_baseline.csv`,
   - snapshot the original rotation results to `rotation_A_baseline.csv` / `rotation_B_baseline.csv` if not already snapshotted,
   - reload audios5 with the corrected GT,
   - re-run rotations A and B,
   - print a before/after delta table per model.

The headline at the bottom is the one number that matters: whisper's Rot B F1 gap. If the relabel hypothesis is right, it should drop from ~+0.13 toward 0."""

lines = [
    "import shutil",
    "",
    "GT_PATH    = NB_DIR / 'audios5GT.csv'",
    "GT_BACKUP  = NB_DIR / 'audios5GT_baseline.csv'",
    "BASELINE_A = SAVE_DIR / 'rotation_A_baseline.csv'",
    "BASELINE_B = SAVE_DIR / 'rotation_B_baseline.csv'",
    "ROT_A      = SAVE_DIR / 'rotation_A.csv'",
    "ROT_B      = SAVE_DIR / 'rotation_B.csv'",
    "",
    "# --- 1. Snapshot the original GT once (so a revert is always possible) ---",
    "if not GT_BACKUP.exists():",
    "    if not GT_PATH.exists():",
    "        raise RuntimeError(f'audios5GT.csv not found at {GT_PATH}')",
    "    shutil.copy(GT_PATH, GT_BACKUP)",
    "    print(f'  backed up original GT -> {GT_BACKUP.name}')",
    "else:",
    "    print(f'  GT backup already exists: {GT_BACKUP.name}  (untouched)')",
    "",
    "# --- 2. Snapshot baseline rotation metrics once ---",
    "if not BASELINE_A.exists():",
    "    if not ROT_A.exists() or not ROT_B.exists():",
    "        raise RuntimeError('Run section 3 first (writes rotation_A.csv and rotation_B.csv).')",
    "    shutil.copy(ROT_A, BASELINE_A)",
    "    shutil.copy(ROT_B, BASELINE_B)",
    "    print(f'  snapshotted baseline rotations -> {BASELINE_A.name}, {BASELINE_B.name}')",
    "else:",
    "    print(f'  rotation baselines already snapshotted  (untouched)')",
    "",
    "# --- 3. Reload audios5 from current (possibly corrected) GT ---",
    "import re as _re_rerun",
    "_RE_CAND_RR = _re_rerun.compile(r'^(.+)_(\\d{1,3})\\.[a-zA-Z0-9]+$')",
    "print('Reloading audios5 from audios5GT.csv (current state)...')",
    "batches['audios5'] = load_folder('audios5')",
    "batches['audios5']['candidate_id'] = batches['audios5']['filename'].astype(str).map(",
    "    lambda f: (_RE_CAND_RR.match(f).group(1) if _RE_CAND_RR.match(f) else None))",
    "_y5 = batches['audios5']['label_int'].values",
    "print(f'  audios5 now: n={len(_y5)}  cheat={int((_y5==1).sum())}  honest={int((_y5==0).sum())}')",
    "_base_gt = pd.read_csv(GT_BACKUP)",
    "_curr_gt = pd.read_csv(GT_PATH)",
    "_n_changed = (_base_gt.set_index(_base_gt.columns[0]).iloc[:,0] !=",
    "              _curr_gt.set_index(_curr_gt.columns[0]).iloc[:,0]).sum() if list(_base_gt.columns) == list(_curr_gt.columns) else 'unknown'",
    "print(f'  rows changed vs baseline GT: {_n_changed}')",
    "",
    "# --- 4. Re-run both rotations with current labels ---",
    "print('Running Rotation A: train=[a2,a4] CV=a4 test=a5 ...')",
    "df_A_corr, _ = run_rotation_base(['audios2','audios4'], 'audios4', 'audios5')",
    "print('Running Rotation B: train=[a2,a5] CV=a5 test=a4 ...')",
    "df_B_corr, _ = run_rotation_base(['audios2','audios5'], 'audios5', 'audios4')",
    "df_A_corr.to_csv(SAVE_DIR / 'rotation_A_corrected.csv', index=False)",
    "df_B_corr.to_csv(SAVE_DIR / 'rotation_B_corrected.csv', index=False)",
    "",
    "# --- 5. Before/after compare on F1 strategy ---",
    "def _compare(label, base_csv, corr_df):",
    "    base = pd.read_csv(base_csv)",
    "    base_f1 = base[base['strategy']=='F1'][['model','cv','te','gap']].rename(",
    "        columns={'cv':'cv_base','te':'te_base','gap':'gap_base'})",
    "    corr_f1 = corr_df[corr_df['strategy']=='F1'][['model','cv','te','gap']].rename(",
    "        columns={'cv':'cv_corr','te':'te_corr','gap':'gap_corr'})",
    "    m = base_f1.merge(corr_f1, on='model')",
    "    m['cv_delta']  = (m['cv_corr']  - m['cv_base']).round(3)",
    "    m['te_delta']  = (m['te_corr']  - m['te_base']).round(3)",
    "    m['gap_delta'] = (m['gap_corr'] - m['gap_base']).round(3)",
    "    print()",
    "    print('='*90)",
    "    print(f' {label}  (F1 strategy)')",
    "    print('='*90)",
    "    with pd.option_context('display.max_columns', None, 'display.width', 200):",
    "        print(m.to_string(index=False))",
    "    return m",
    "",
    "A_cmp = _compare('ROTATION A baseline vs corrected (CV=a4 test=a5)', BASELINE_A, df_A_corr)",
    "B_cmp = _compare('ROTATION B baseline vs corrected (CV=a5 test=a4)', BASELINE_B, df_B_corr)",
    "",
    "# --- 6. Headline ---",
    "print()",
    "print('='*90)",
    "print(' HEADLINE: did Rot B whisper gap shrink?  (the question this whole audit was for)')",
    "print('='*90)",
    "_w = B_cmp[B_cmp['model']=='whisper_wp_xgb'].iloc[0]",
    "print(f'  whisper_wp_xgb Rot B F1 gap:')",
    "print(f'    baseline   {_w[\"gap_base\"]:+.3f}')",
    "print(f'    corrected  {_w[\"gap_corr\"]:+.3f}')",
    "print(f'    delta      {_w[\"gap_delta\"]:+.3f}   (negative = gap shrunk = relabel hypothesis confirmed)')",
    "if 'wavlm_whole_ft' in B_cmp['model'].values:",
    "    _v = B_cmp[B_cmp['model']=='wavlm_whole_ft'].iloc[0]",
    "    print(f'  wavlm_whole_ft Rot B F1 gap:')",
    "    print(f'    baseline   {_v[\"gap_base\"]:+.3f}')",
    "    print(f'    corrected  {_v[\"gap_corr\"]:+.3f}')",
    "    print(f'    delta      {_v[\"gap_delta\"]:+.3f}')",
    "",
    "print()",
    "print('To revert the GT to the original:  shutil.copy(GT_BACKUP, GT_PATH)  then re-run this cell.')",
]
code_src = '\n'.join(lines)

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md_src.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": [ln + '\n' for ln in code_src.split('\n')]},
]
nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')

# AST-validate the new code cell
import ast
ast.parse(''.join(nb['cells'][-1]['source']))
print('APPENDED §12 — total cells:', len(nb['cells']), '— code AST-valid')
