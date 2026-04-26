"""Rewrite §14: combined relabel + filter test, dual test view (unfiltered + filtered)."""
import json, ast
from pathlib import Path

NB = Path(__file__).with_name('honest_eval_and_improve.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

# Find the §14 markdown index by header text
md_idx = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown' and '14. Min-speaking-time filter' in ''.join(c['source']):
        md_idx = i
        break
if md_idx is None:
    raise RuntimeError('Could not find §14 markdown header.')
code_idx = md_idx + 1
print(f'§14 markdown at cell {md_idx}, code at cell {code_idx}')

# New markdown
md_src = """## 14. Combined experiment — relabel + min-speaking-time filter, dual test views

This is the high-quality calibration cell (slow, multi-seed, all 5 strategies). Run it once with the chosen `MIN_SPEAKING_S` to get the precise post-intervention numbers.

What it does:
1. **Reloads `audios5` from disk** — picks up any `audios5GT.csv` edits you made after the §11 audit.
2. **Filters training-eligible batches** (always-train + CV target) at `MIN_SPEAKING_S`. Test batch left intact for full-test scoring.
3. **Runs both rotations** with the existing `run_rotation_base` (3 seeds × 5 folds × bootstrap CI per strategy).
4. **Refits each model on filtered training** to score the test set, then computes metrics two ways:
   - `te_full` — metrics on the **entire** test batch (no filter applied to test). This is the deployment-realistic number.
   - `te_filt` — metrics on the **subset** of test rows with `speaking_time_s ≥ MIN_SPEAKING_S`. This isolates "how good is the model on audios where the filter assumption holds?"
5. **Compares to baseline** = pre-relabel + pre-filter snapshot (`rotation_A.csv` / `rotation_B.csv` from §3's first run).

The delta between baseline and current = combined effect of (relabel + filter).
The delta between `te_full` and `te_filt` = how much the filter would help if it were also applied at deployment time (it isn't — but useful as a ceiling estimate).

Knobs at the top:
- `MIN_SPEAKING_S` — your chosen threshold (default 30, per §15 sweep).
- `MODELS_TO_RUN` — None for all 7 models (slow ~2 hr); pass a list to trim.
- `RELOAD_AUDIOS5_GT` — True to reload audios5 from disk (set False if you already reloaded earlier in the session)."""

# New code
lines = [
    "# ---------- KNOBS ----------",
    "MIN_SPEAKING_S    = 30.0",
    "MODELS_TO_RUN     = None     # None = all in BASE_REGISTRY; or list e.g. ['whisper_wp_xgb','wavlm_whole_ft','text_stylo']",
    "RELOAD_AUDIOS5_GT = True     # reload audios5 from disk so any GT edits take effect",
    "",
    "if 'durations' not in dir():",
    "    raise RuntimeError('Run section 13 first (defines `durations`).')",
    "if 'run_rotation_base' not in dir():",
    "    raise RuntimeError('Run section 3 first (defines `run_rotation_base`).')",
    "",
    "import re as _re14",
    "_RE_CAND14 = _re14.compile(r'^(.+)_(\\d{1,3})\\.[a-zA-Z0-9]+$')",
    "",
    "# ---------- 1. Reload audios5 to pick up GT corrections ----------",
    "if RELOAD_AUDIOS5_GT:",
    "    print('Reloading audios5 from disk to pick up any audios5GT.csv edits...')",
    "    new_a5 = load_folder('audios5')",
    "    new_a5['candidate_id'] = new_a5['filename'].astype(str).map(",
    "        lambda f: (_RE_CAND14.match(f).group(1) if _RE_CAND14.match(f) else None))",
    "    if 'label_int' in batches['audios5'].columns:",
    "        merged = batches['audios5'][['filename','label_int']].merge(",
    "            new_a5[['filename','label_int']], on='filename', suffixes=('_old','_new'))",
    "        n_changed = int((merged['label_int_old'] != merged['label_int_new']).sum())",
    "        print(f'  audios5 label changes vs in-memory copy: {n_changed} rows')",
    "    batches['audios5'] = new_a5",
    "    durations['audios5'] = compute_durations('audios5', force=True)",
    "    print(f'  audios5 now: n={len(new_a5)}  cheat={int((new_a5[\"label_int\"]==1).sum())}  honest={int((new_a5[\"label_int\"]==0).sum())}')",
    "",
    "# ---------- 2. Filter helper (NaN durations are kept) ----------",
    "def _filter_batch(name, min_s):",
    "    if min_s <= 0:",
    "        return batches[name]",
    "    d = durations[name][['filename','speaking_time_s']]",
    "    keep = set(d[(d['speaking_time_s'] >= min_s) | (d['speaking_time_s'].isna())]['filename'])",
    "    return batches[name][batches[name]['filename'].isin(keep)].reset_index(drop=True)",
    "",
    "# ---------- 3. Filter stats ----------",
    "filt_stats = []",
    "for b in BATCHES:",
    "    n_b = len(batches[b])",
    "    n_a = len(_filter_batch(b, MIN_SPEAKING_S))",
    "    pos_b = int((batches[b]['label_int']==1).sum())",
    "    pos_a = int((_filter_batch(b, MIN_SPEAKING_S)['label_int']==1).sum())",
    "    filt_stats.append({",
    "        'batch':       b,",
    "        'n_before':    n_b,",
    "        'n_after':     n_a,",
    "        'dropped':     n_b - n_a,",
    "        'pct_dropped': round((n_b - n_a) / max(n_b,1) * 100, 1),",
    "        'pos_before':  pos_b,",
    "        'pos_after':   pos_a,",
    "        'pos_dropped': pos_b - pos_a,",
    "        'neg_dropped': (n_b - pos_b) - (n_a - pos_a),",
    "    })",
    "print()",
    "print(f'Min speaking-time filter applied: keep audios with speaking_time_s >= {MIN_SPEAKING_S}')",
    "print(pd.DataFrame(filt_stats).to_string(index=False))",
    "",
    "# ---------- 4. Combined rotation: filtered training, dual-test scoring ----------",
    "model_list = MODELS_TO_RUN or list(BASE_REGISTRY)",
    "print()",
    "print(f'Models in this run ({len(model_list)}): {model_list}')",
    "",
    "def _dual_test_rotation(train_folders, cv_target, test_folder, min_s, models):",
    "    \"\"\"Filter training-eligible batches; score full + filtered test for each model (F1 strategy).\"\"\"",
    "    view = dict(batches)",
    "    for b in train_folders:",
    "        view[b] = _filter_batch(b, min_s)",
    "    df_cv  = view[cv_target].reset_index(drop=True)",
    "    y_cv   = df_cv['label_int'].values",
    "    groups = df_cv['candidate_id'].values",
    "    df_al  = pd.concat([view[b] for b in train_folders if b != cv_target], ignore_index=True) \\",
    "             if any(b != cv_target for b in train_folders) else None",
    "    df_te  = view[test_folder].reset_index(drop=True)   # unfiltered (full test)",
    "    y_te   = df_te['label_int'].values",
    "    sp_te  = durations[test_folder].set_index('filename')['speaking_time_s']",
    "    mask_keep = df_te['filename'].map(",
    "        lambda fn: pd.isna(sp_te.get(fn, np.nan)) or sp_te.get(fn, 0) >= min_s",
    "    ).values",
    "    rows = []",
    "    for name in models:",
    "        X_fn, factory = BASE_REGISTRY[name]",
    "        oof = repeated_kfold_oof(df_cv, df_al, X_fn, factory)",
    "        df_tr_full = pd.concat([df_al, df_cv], ignore_index=True) if df_al is not None else df_cv",
    "        Xtr = X_fn(df_tr_full); ytr = df_tr_full['label_int'].values",
    "        sc = StandardScaler().fit(Xtr)",
    "        clf = factory(42); clf.fit(sc.transform(Xtr), ytr)",
    "        p_te = clf.predict_proba(sc.transform(X_fn(df_te)))[:,1]",
    "        thr_pt, cv_f1 = best_f1_thr(oof, y_cv)",
    "        thr_md, thr_lo, thr_hi, _ = bootstrap_thr_ci(oof, y_cv, 'F1')",
    "        thr_use = thr_md if thr_md is not None else thr_pt",
    "        m_full = metrics_at(p_te, y_te, thr_use)",
    "        if mask_keep.sum() > 0:",
    "            m_filt = metrics_at(p_te[mask_keep], y_te[mask_keep], thr_use)",
    "        else:",
    "            m_filt = dict(prec=None, rec=None, f1=None)",
    "        gap_full = (cv_f1 - m_full['f1']) if m_full['f1'] is not None else None",
    "        gap_filt = (cv_f1 - m_filt['f1']) if m_filt['f1'] is not None else None",
    "        rows.append({",
    "            'model':         name,",
    "            'thr':           round(thr_use, 3) if thr_use is not None else None,",
    "            'thr_p10':       round(thr_lo, 3)  if thr_lo  is not None else None,",
    "            'thr_p90':       round(thr_hi, 3)  if thr_hi  is not None else None,",
    "            'cv':            round(cv_f1, 3),",
    "            'te_full':       round(m_full['f1'], 3) if m_full['f1'] is not None else None,",
    "            'te_prec_full':  round(m_full['prec'], 3) if m_full['prec'] is not None else None,",
    "            'te_rec_full':   round(m_full['rec'], 3)  if m_full['rec']  is not None else None,",
    "            'gap_full':      round(gap_full, 3) if gap_full is not None else None,",
    "            'n_te_full':     int(len(y_te)),",
    "            'te_filt':       round(m_filt['f1'], 3) if m_filt['f1'] is not None else None,",
    "            'te_prec_filt':  round(m_filt['prec'], 3) if m_filt['prec'] is not None else None,",
    "            'te_rec_filt':   round(m_filt['rec'], 3)  if m_filt['rec']  is not None else None,",
    "            'gap_filt':      round(gap_filt, 3) if gap_filt is not None else None,",
    "            'n_te_filt':     int(mask_keep.sum()),",
    "        })",
    "    return pd.DataFrame(rows)",
    "",
    "import time as _t14",
    "print()",
    "print('Running Rotation A (train=[a2,a4] filtered, CV=a4, test=a5) ...')",
    "_t0 = _t14.time()",
    "df_A_dual = _dual_test_rotation(['audios2','audios4'], 'audios4', 'audios5', MIN_SPEAKING_S, model_list)",
    "print(f'  done in {_t14.time()-_t0:.1f}s')",
    "",
    "print('Running Rotation B (train=[a2,a5] filtered, CV=a5, test=a4) ...')",
    "_t0 = _t14.time()",
    "df_B_dual = _dual_test_rotation(['audios2','audios5'], 'audios5', 'audios4', MIN_SPEAKING_S, model_list)",
    "print(f'  done in {_t14.time()-_t0:.1f}s')",
    "",
    "df_A_dual.to_csv(SAVE_DIR / f'rotation_A_minS{int(MIN_SPEAKING_S)}_dual.csv', index=False)",
    "df_B_dual.to_csv(SAVE_DIR / f'rotation_B_minS{int(MIN_SPEAKING_S)}_dual.csv', index=False)",
    "",
    "# ---------- 5. Compare to baseline (pre-relabel, pre-filter) ----------",
    "BASE_A_PATH = SAVE_DIR / 'rotation_A_baseline.csv' if (SAVE_DIR / 'rotation_A_baseline.csv').exists() else SAVE_DIR / 'rotation_A.csv'",
    "BASE_B_PATH = SAVE_DIR / 'rotation_B_baseline.csv' if (SAVE_DIR / 'rotation_B_baseline.csv').exists() else SAVE_DIR / 'rotation_B.csv'",
    "",
    "def _compare(label, base_path, dual_df, te_col, gap_col):",
    "    base = pd.read_csv(base_path)",
    "    base_f1 = base[base['strategy']=='F1'][['model','cv','te','gap']].rename(",
    "        columns={'cv':'cv_base','te':'te_base','gap':'gap_base'})",
    "    cur = dual_df[['model','cv', te_col, gap_col]].rename(",
    "        columns={'cv':'cv_cur', te_col:'te_cur', gap_col:'gap_cur'})",
    "    m = base_f1.merge(cur, on='model')",
    "    m['cv_delta']  = (m['cv_cur']  - m['cv_base']).round(3)",
    "    m['te_delta']  = (m['te_cur']  - m['te_base']).round(3)",
    "    m['gap_delta'] = (m['gap_cur'] - m['gap_base']).round(3)",
    "    print()",
    "    print('=' * 110)",
    "    print(f' {label}  (F1 strategy)')",
    "    print('  baseline = pre-relabel, pre-filter (from rotation_*.csv)')",
    "    print('=' * 110)",
    "    with pd.option_context('display.max_columns', None, 'display.width', 220):",
    "        print(m.to_string(index=False))",
    "    return m",
    "",
    "A_full = _compare(f'Rotation A vs baseline  -- TEST = full a5 (unfiltered, deployment-like)',",
    "                  BASE_A_PATH, df_A_dual, 'te_full', 'gap_full')",
    "A_filt = _compare(f'Rotation A vs baseline  -- TEST = a5 filtered to >= {MIN_SPEAKING_S}s only',",
    "                  BASE_A_PATH, df_A_dual, 'te_filt', 'gap_filt')",
    "B_full = _compare(f'Rotation B vs baseline  -- TEST = full a4 (unfiltered, deployment-like)',",
    "                  BASE_B_PATH, df_B_dual, 'te_full', 'gap_full')",
    "B_filt = _compare(f'Rotation B vs baseline  -- TEST = a4 filtered to >= {MIN_SPEAKING_S}s only',",
    "                  BASE_B_PATH, df_B_dual, 'te_filt', 'gap_filt')",
    "",
    "# ---------- 6. Headline ----------",
    "print()",
    "print('=' * 110)",
    "print(f' HEADLINE  (combined effect: any GT relabel + filter at MIN_SPEAKING_S={MIN_SPEAKING_S}s)')",
    "print('=' * 110)",
    "for model_name in ['whisper_wp_xgb','wavlm_whole_ft','text_stylo']:",
    "    if model_name not in df_B_dual['model'].values: continue",
    "    rB_full = B_full[B_full['model']==model_name].iloc[0]",
    "    rB_filt = B_filt[B_filt['model']==model_name].iloc[0]",
    "    print(f'  {model_name}')",
    "    print(f'    Rot B  full-test    gap  baseline {rB_full[\"gap_base\"]:+.3f}  ->  current {rB_full[\"gap_cur\"]:+.3f}   delta {rB_full[\"gap_delta\"]:+.3f}')",
    "    print(f'    Rot B  full-test    F1   baseline  {rB_full[\"te_base\"]:.3f}  ->  current  {rB_full[\"te_cur\"]:.3f}   delta {rB_full[\"te_delta\"]:+.3f}')",
    "    print(f'    Rot B  filtered-te  gap  baseline {rB_full[\"gap_base\"]:+.3f}  ->  current {rB_filt[\"gap_cur\"]:+.3f}   delta {rB_filt[\"gap_delta\"]:+.3f}   (n={int(df_B_dual[df_B_dual[\"model\"]==model_name].iloc[0][\"n_te_filt\"])} of {int(df_B_dual[df_B_dual[\"model\"]==model_name].iloc[0][\"n_te_full\"])})')",
    "    print(f'    Rot B  filtered-te  F1   baseline  {rB_full[\"te_base\"]:.3f}  ->  current  {rB_filt[\"te_cur\"]:.3f}   delta {rB_filt[\"te_delta\"]:+.3f}')",
    "    print()",
    "",
    "print('Reading guide:')",
    "print('  - full-test gap delta strongly negative -> intervention closed the deployment-realistic gap.')",
    "print('  - filtered-te F1 >> full-test F1 -> the model is genuinely better on audios where it has enough signal.')",
    "print('    The filtered-te number is what you would see if you abstained on short audios at deployment.')",
    "print('  - cv_delta strongly negative without te_delta improving -> filter cut into real signal; threshold should drop.')",
]
code_src = '\n'.join(lines)

nb['cells'][md_idx]['source']  = md_src.splitlines(keepends=True)
nb['cells'][code_idx]['source'] = [ln + '\n' for ln in code_src.split('\n')]

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
ast.parse(''.join(nb['cells'][code_idx]['source']))
print(f'§14 rewritten ({len("".join(nb["cells"][code_idx]["source"]))} chars) — AST-valid')
