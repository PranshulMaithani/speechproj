"""Append §15 'Fast threshold sweep' to honest_eval_and_improve.ipynb.
Tunable speed knobs at the top of the cell.
"""
import json, ast
from pathlib import Path

NB = Path(__file__).with_name('honest_eval_and_improve.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

md_src = """## 15. Fast min-speaking-time sweep

§14 takes ~2 hours per threshold because it runs all 7 base models × 3 seeds × 5 folds × bootstrap CIs × 2 rotations. This sweep is a slimmer version: a single F1 estimate per (model, rotation, threshold), no bootstrap, configurable folds/seeds/models. With the defaults it sweeps 6 thresholds in roughly 15-20 min on CPU.

The point of this cell is to find the *shape* of the gap-vs-threshold curve, not to produce final calibrated numbers. Once the optimal threshold is identified you go back to §14 with that one value for the precise estimate.

Speed knobs (top of the code cell — edit and re-run):
- `SWEEP_MODELS`: which models to include. Default skips wavlm_whole_ft (slow, follows whisper closely). Add it back if you want to confirm.
- `SWEEP_THRESHOLDS`: list of `MIN_SPEAKING_S` values to sweep. `0` = no filter (baseline).
- `SWEEP_N_FOLDS`: 3 for speed, 5 for parity with §3.
- `SWEEP_N_SEEDS`: 1 for speed, 3 for parity.
- `SWEEP_N_EST`: XGB n_estimators (200 fast / 400 default).

Test batch is always unfiltered. Same logic as §14: filter only training-eligible batches."""

lines = [
    "# ----- Speed knobs (edit before running) -----",
    "SWEEP_MODELS     = ['whisper_wp_xgb', 'text_stylo']        # add 'wavlm_whole_ft' for ~3x slower",
    "SWEEP_THRESHOLDS = [0, 15, 20, 25, 30, 35]                  # 0 = no filter (baseline)",
    "SWEEP_N_FOLDS    = 3                                        # 5 for parity with §3",
    "SWEEP_N_SEEDS    = 1                                        # 3 for parity",
    "SWEEP_N_EST      = 200                                      # 400 for parity",
    "",
    "if 'durations' not in dir():",
    "    raise RuntimeError('Run section 13 first (defines `durations`).')",
    "if 'best_f1_thr' not in dir() or 'metrics_at' not in dir():",
    "    raise RuntimeError('Run section 2 first (defines best_f1_thr / metrics_at).')",
    "",
    "import time as _t",
    "",
    "# Self-contained filter — independent of §14",
    "def _sw_filter(name, min_s):",
    "    if min_s <= 0:",
    "        return batches[name]",
    "    d = durations[name][['filename','speaking_time_s']]",
    "    keep = set(d[(d['speaking_time_s'] >= min_s) | (d['speaking_time_s'].isna())]['filename'])",
    "    return batches[name][batches[name]['filename'].isin(keep)].reset_index(drop=True)",
    "",
    "# Fast XGB factory honoring the speed knobs",
    "def _sw_xgb(n_feats, seed):",
    "    cs = 0.3 if n_feats > 500 else 0.8",
    "    return xgb.XGBClassifier(",
    "        n_estimators=SWEEP_N_EST, max_depth=4, learning_rate=0.05,",
    "        subsample=0.8, colsample_bytree=cs, min_child_weight=3,",
    "        scale_pos_weight=float(SPW_DEPLOY), eval_metric='logloss',",
    "        random_state=seed)",
    "",
    "# Single-rotation runner — F1 strategy only, no bootstrap",
    "def _sw_rot(view, train_folders, cv_target, test_folder, model_names):",
    "    df_cv  = view[cv_target].reset_index(drop=True)",
    "    y_cv   = df_cv['label_int'].values",
    "    groups = df_cv['candidate_id'].values",
    "    df_al  = pd.concat([view[b] for b in train_folders if b != cv_target], ignore_index=True) \\",
    "             if any(b != cv_target for b in train_folders) else None",
    "    df_te  = view[test_folder].reset_index(drop=True)",
    "    y_te   = df_te['label_int'].values",
    "    rows = []",
    "    for name in model_names:",
    "        X_fn, _ = BASE_REGISTRY[name]",
    "        n_feat  = X_fn(df_cv.head(1)).shape[1]",
    "        oof_sum = np.zeros(len(df_cv)); oof_cnt = np.zeros(len(df_cv))",
    "        for s in range(SWEEP_N_SEEDS):",
    "            skf = StratifiedGroupKFold(n_splits=SWEEP_N_FOLDS, shuffle=True, random_state=42+s)",
    "            for tr_idx, va_idx in skf.split(df_cv, y_cv, groups=groups):",
    "                df_tr = pd.concat([df_al, df_cv.iloc[tr_idx]], ignore_index=True) \\",
    "                        if df_al is not None else df_cv.iloc[tr_idx]",
    "                Xtr = X_fn(df_tr); ytr = df_tr['label_int'].values",
    "                Xva = X_fn(df_cv.iloc[va_idx])",
    "                sc = StandardScaler().fit(Xtr)",
    "                clf = _sw_xgb(n_feat, 42+s)",
    "                clf.fit(sc.transform(Xtr), ytr)",
    "                oof_sum[va_idx] += clf.predict_proba(sc.transform(Xva))[:,1]",
    "                oof_cnt[va_idx] += 1",
    "        oof = oof_sum / np.maximum(oof_cnt, 1)",
    "        df_tr_full = pd.concat([df_al, df_cv], ignore_index=True) if df_al is not None else df_cv",
    "        Xtr = X_fn(df_tr_full); ytr = df_tr_full['label_int'].values",
    "        sc = StandardScaler().fit(Xtr)",
    "        clf = _sw_xgb(n_feat, 42); clf.fit(sc.transform(Xtr), ytr)",
    "        p_te = clf.predict_proba(sc.transform(X_fn(df_te)))[:,1]",
    "        thr, cv_f1 = best_f1_thr(oof, y_cv)",
    "        te = metrics_at(p_te, y_te, thr)",
    "        rows.append({",
    "            'model': name, 'thr': round(thr,2),",
    "            'cv_f1': round(cv_f1,3), 'te_f1': round(te['f1'],3),",
    "            'gap':   round(cv_f1 - te['f1'], 3),",
    "        })",
    "    return pd.DataFrame(rows)",
    "",
    "# ---- Sweep ----",
    "results = []",
    "for min_s in SWEEP_THRESHOLDS:",
    "    t0 = _t.time()",
    "    print(f'--- MIN_SPEAKING_S = {min_s}s ---')",
    "    view = dict(batches)",
    "    view['audios2'] = _sw_filter('audios2', min_s)",
    "    view['audios4'] = _sw_filter('audios4', min_s)",
    "    print(f'  Rot A  a2:{len(view[\"audios2\"])}/{len(batches[\"audios2\"])}  a4:{len(view[\"audios4\"])}/{len(batches[\"audios4\"])}  test_a5:{len(view[\"audios5\"])}')",
    "    df_A = _sw_rot(view, ['audios2','audios4'], 'audios4', 'audios5', SWEEP_MODELS)",
    "    view = dict(batches)",
    "    view['audios2'] = _sw_filter('audios2', min_s)",
    "    view['audios5'] = _sw_filter('audios5', min_s)",
    "    print(f'  Rot B  a2:{len(view[\"audios2\"])}/{len(batches[\"audios2\"])}  a5:{len(view[\"audios5\"])}/{len(batches[\"audios5\"])}  test_a4:{len(view[\"audios4\"])}')",
    "    df_B = _sw_rot(view, ['audios2','audios5'], 'audios5', 'audios4', SWEEP_MODELS)",
    "    for r in df_A.itertuples():",
    "        results.append({'min_s': min_s, 'rot': 'A', 'model': r.model,",
    "                        'cv_f1': r.cv_f1, 'te_f1': r.te_f1, 'gap': r.gap})",
    "    for r in df_B.itertuples():",
    "        results.append({'min_s': min_s, 'rot': 'B', 'model': r.model,",
    "                        'cv_f1': r.cv_f1, 'te_f1': r.te_f1, 'gap': r.gap})",
    "    print(f'  done in {_t.time()-t0:.1f}s')",
    "",
    "results_df = pd.DataFrame(results)",
    "results_df.to_csv(SAVE_DIR / 'min_speaking_sweep.csv', index=False)",
    "",
    "def _piv(metric):",
    "    return results_df.pivot_table(index=['model','rot'], columns='min_s', values=metric).round(3)",
    "",
    "print()",
    "print('='*90)",
    "print(' GAP (cv - te)  per (model, rotation) across MIN_SPEAKING_S')",
    "print('='*90)",
    "print(_piv('gap').to_string())",
    "print()",
    "print('='*90)",
    "print(' TEST F1  per (model, rotation) across MIN_SPEAKING_S')",
    "print('='*90)",
    "print(_piv('te_f1').to_string())",
    "print()",
    "print('='*90)",
    "print(' CV F1  per (model, rotation) across MIN_SPEAKING_S')",
    "print('='*90)",
    "print(_piv('cv_f1').to_string())",
    "",
    "print()",
    "print(f'Saved -> {SAVE_DIR / \"min_speaking_sweep.csv\"}')",
    "print()",
    "print('Reading guide:')",
    "print('  - Find the column where Rot B gap is smallest AND test F1 is highest.')",
    "print('  - Monotonic gap shrink through 35 -> stricter is better; we are still removing confounds.')",
    "print('  - U-shape -> optimum somewhere in middle; past it we cut into real signal.')",
    "print('  - Sharp cv_f1 drop at any threshold -> training set too small at that level.')",
]

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md_src.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": [ln + '\n' for ln in '\n'.join(lines).split('\n')]},
]
nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
ast.parse(''.join(nb['cells'][-1]['source']))
print('APPENDED §15 — total cells:', len(nb['cells']), '— code AST-valid')
