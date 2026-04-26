"""Build experimentations_summary.ipynb from scratch.

Tells the chronological story of the CV/test rotation gap: baseline -> diagnostics ->
hypothesis tests -> final intervention. Self-contained — re-trains every model, no
loading from prior CSVs.

Heavy cells (full registry rotations) take ~2 hr each on CPU. Diagnostic cells are
cheap (<5 min each). All output CSVs go to checkpoints_story/ with friendlier column
names than the original honest_eval CSVs.

Safe to delete after running.
"""
import json, ast
from pathlib import Path

NB = Path(__file__).with_name('experimentations_summary.ipynb')
OUT_DIR = 'checkpoints_story'

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(lines):
    src = '\n'.join(lines)
    ast.parse(src)  # validate
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": [ln + '\n' for ln in src.split('\n')],
    })


# =========================================================================
md("""# Experimentations Summary — the CV vs test rotation gap

This notebook reproduces, end-to-end, the investigation into why our cheating-detection models showed a large gap between CV and held-out test performance, and how we narrowed it down to two confounds (one label noise, one acoustic) that we mitigated at training time.

We follow the actual chronology:

1. **Setup & data loading** — restore the original ground truth so the story works as it happened.
2. **The problem** — train every base model on the raw data, run both rotations, and look at the gap. This is the "before" picture.
3. **Diagnostic 1** — is one batch intrinsically harder than the other?
4. **Hypothesis A** — is `audios5` mislabeled in places?
5. **Test of hypothesis A** — apply the known correction and re-run. Did the gap close?
6. **Hypothesis B** — does `audios5` have an acoustic confound (very short audios)?
7. **Threshold sweep** — at what minimum speaking-time does the gap stop shrinking?
8. **Final intervention** — apply both fixes (corrected labels + filter at chosen threshold) and re-train every model.
9. **Before/after summary** — headline numbers.

The point of this notebook is *honest* numbers — every step retrains from scratch so we are not fooling ourselves with cached results. CSVs are written to `checkpoints_story/` with descriptive column names.
""")

# =========================================================================
md("""## 0. Setup — imports, paths, restore original ground truth

We restore `audios5GT.csv` from `audios5GT_baseline.csv` (which was snapshotted in the original audit) so this notebook starts in the **pre-correction** state. The user's manual corrections are preserved separately to `audios5GT_user_corrected.csv` so Section 5 can re-apply them.

Hard requirement: `checkpoints_honest_eval/audios5GT_baseline.csv` must exist. If it does not, run §12 of `honest_eval_and_improve.ipynb` once to create it (or copy your current GT there manually before deleting any state).
""")

code([
    "from pathlib import Path",
    "import json, re, shutil, warnings, time",
    "import numpy as np",
    "import pandas as pd",
    "",
    "import xgboost as xgb",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold",
    "from sklearn.metrics import (precision_score, recall_score, f1_score,",
    "                              confusion_matrix, roc_auc_score, average_precision_score)",
    "",
    "warnings.filterwarnings('ignore')",
    "",
    "NB_DIR     = Path('.').resolve()",
    f"SAVE_DIR   = NB_DIR / '{OUT_DIR}'",
    "SAVE_DIR.mkdir(parents=True, exist_ok=True)",
    "PRIOR_DIR  = NB_DIR / 'checkpoints_honest_eval'",
    "",
    "# --- Restore original GT so the story starts pre-correction ---",
    "GT_PATH      = NB_DIR / 'audios5GT.csv'",
    "GT_BASELINE  = PRIOR_DIR / 'audios5GT_baseline.csv'",
    "GT_USER_COPY = PRIOR_DIR / 'audios5GT_user_corrected.csv'",
    "",
    "if not GT_BASELINE.exists():",
    "    raise RuntimeError(f'Missing baseline GT: {GT_BASELINE}\\n'",
    "                       f'Run section 12 of honest_eval_and_improve.ipynb once to create it,\\n'",
    "                       f'or copy your current audios5GT.csv to that path manually.')",
    "",
    "# Snapshot the user's currently-corrected GT (only on first run of this notebook)",
    "if not GT_USER_COPY.exists():",
    "    shutil.copy(GT_PATH, GT_USER_COPY)",
    "    print(f'  snapshotted current GT (with your corrections) -> {GT_USER_COPY.name}')",
    "else:",
    "    print(f'  user-corrected GT snapshot already exists -> {GT_USER_COPY.name} (untouched)')",
    "",
    "shutil.copy(GT_BASELINE, GT_PATH)",
    "print(f'  restored audios5GT.csv from {GT_BASELINE.name} (pre-correction state)')",
    "",
    "# --- Project knobs (kept aligned with honest_eval_and_improve.ipynb) ---",
    "BATCHES         = ['audios2', 'audios4', 'audios5']",
    "DEPLOY_POS_RATE = 0.17",
    "SPW_DEPLOY      = (1.0 - DEPLOY_POS_RATE) / DEPLOY_POS_RATE",
    "N_SEEDS         = 3       # repeats of K-fold CV",
    "N_FOLDS         = 5",
    "BOOT_N          = 200     # bootstrap iterations for threshold CI",
    "STRATEGIES      = ['F1', 'P80', 'P90']",
    "PREC_FLOOR      = {'P80': 0.80, 'P90': 0.90}",
    "",
    "LABEL_MAP = {",
    "    'read': 1, 'cheating': 1, 'reading': 1, 'scripted': 1, 'yes': 1, '1': 1, 1: 1,",
    "    'spontaneous': 0, 'not cheating': 0, 'not_cheating': 0, 'no': 0, '0': 0, 0: 0, 'genuine': 0,",
    "}",
    "",
    "# Text feature lists (mirrors honest_eval cell 1)",
    "ALL_TEXT_FEATURES = [",
    "    'filler_rate','filler_count','repetition_rate','repair_rate','discourse_marker_rate','hedge_rate',",
    "    'ttr','mattr','mtld','complex_word_rate','avg_word_length','n_words','n_unique_words',",
    "    'avg_sentence_length','std_sentence_length','fragment_rate','n_sentences','self_ref_rate',",
    "    'noun_rate','verb_rate','adj_rate',",
    "    'pause_mean','pause_std','pause_median','pause_skew','long_pause_rate','pause_ratio','n_pauses',",
    "    'pause_regularity','pause_before_content_ratio','pause_before_function_ratio','mid_phrase_pause_rate',",
    "    'words_per_sec','articulation_rate','initial_pause','longest_pause',",
    "    'suspicious_gap_count','suspicious_gap_ratio',",
    "    'formal_transition_count','formal_transition_rate','ai_phrase_count','ai_phrase_rate',",
    "    'f0_mean','f0_std','f0_range','f0_skew','f0_slope','energy_mean','energy_std','speaking_rate_std',",
    "    'jitter_local','shimmer_local','hnr_mean',",
    "    'mean_perplexity','burstiness',",
    "]",
    "STYLO_FEATS = ['ttr','mattr','mtld','complex_word_rate','avg_word_length','n_words','n_unique_words',",
    "               'avg_sentence_length','std_sentence_length','fragment_rate','n_sentences','self_ref_rate',",
    "               'noun_rate','verb_rate','adj_rate']",
    "",
    "print()",
    "print(f'N_SEEDS={N_SEEDS}  N_FOLDS={N_FOLDS}  BOOT_N={BOOT_N}')",
    "print(f'SPW_DEPLOY={SPW_DEPLOY:.2f}  DEPLOY_POS_RATE={DEPLOY_POS_RATE}')",
    "print(f'Save dir: {SAVE_DIR}')",
])

# =========================================================================
md("""## 1. Load the three batches and attach candidate IDs

Three batches: `audios2` (always in train), `audios4`, `audios5`. Filename convention is `<candidate>_<question>.<ext>`, where each candidate has questions Q25 / Q26 / Q27. The **candidate ID** is what we use to group folds — without it, the same candidate's questions would split across train and validation and we would leak.

The two evaluation rotations are:
- **Rotation A** — train = [audios2, audios4], CV = audios4, test = audios5
- **Rotation B** — train = [audios2, audios5], CV = audios5, test = audios4
""")

code([
    "def load_gt(name):",
    "    gt = pd.read_csv(NB_DIR / f'{name}GT.csv')",
    "    fn_col  = next(c for c in gt.columns if c.lower() in ('filename','file','name'))",
    "    lbl_col = next(c for c in gt.columns if c.lower() in ('label','class','cheating','gt','label_int','ground_truth'))",
    "    gt = gt.rename(columns={fn_col: 'filename', lbl_col: 'label_raw'})",
    "    gt['label_int'] = gt['label_raw'].map(",
    "        lambda x: LABEL_MAP.get(x, LABEL_MAP.get(str(x).lower().strip(), -1)))",
    "    return gt[gt['label_int'].isin([0, 1])][['filename', 'label_int']]",
    "",
    "def _first_existing(candidates):",
    "    for c in candidates:",
    "        p = NB_DIR / c",
    "        if p.exists(): return p",
    "    return None",
    "",
    "WAVLM_CSV_CANDIDATES = {",
    "    'wpre': lambda n: [f'{n}_wavlm_whole.csv', f'{n}_whole_pretrained.csv'],",
    "    'wft' : lambda n: [f'{n}_whole_finetuned.csv'],",
    "}",
    "",
    "def load_folder(name):",
    "    gt   = load_gt(name)",
    "    text = pd.read_csv(NB_DIR / f'{name}_features.csv')",
    "    df = gt.merge(text, on='filename', how='inner')",
    "    for tag, cand_fn in WAVLM_CSV_CANDIDATES.items():",
    "        path = _first_existing(cand_fn(name))",
    "        if path is not None:",
    "            sub = pd.read_csv(path)",
    "            sub = sub.rename(columns={c: (c if c == 'filename' else f'{c}_{tag}') for c in sub.columns})",
    "            df = df.merge(sub, on='filename', how='inner')",
    "    whr = pd.read_csv(NB_DIR / f'{name}_whisper_whole.csv')",
    "    df = df.merge(whr.rename(columns={c: (c if c == 'filename' else f'{c}_whisper') for c in whr.columns}),",
    "                  on='filename', how='inner')",
    "    df['batch'] = name",
    "    return df",
    "",
    "_RE_CAND = re.compile(r'^(.+)_(\\d{1,3})\\.[a-zA-Z0-9]+$')",
    "def attach_candidate_id(df):",
    "    df = df.copy()",
    "    df['candidate_id'] = df['filename'].astype(str).map(",
    "        lambda f: (_RE_CAND.match(f).group(1) if _RE_CAND.match(f) else None))",
    "    return df",
    "",
    "batches = {b: attach_candidate_id(load_folder(b)) for b in BATCHES}",
    "",
    "# Discover feature column lists from audios2",
    "first = batches['audios2']",
    "WPRE_COLS = [c for c in first.columns if c.endswith('_wpre')  and c.startswith('wavlm_')]",
    "WFT_COLS  = [c for c in first.columns if c.endswith('_wft')   and c.startswith('wavlm_')]",
    "WH_COLS   = [c for c in first.columns if c.endswith('_whisper') and c.startswith('whisper_')]",
    "TEXT_ALL   = [c for c in ALL_TEXT_FEATURES if c in first.columns]",
    "TEXT_STYLO = [c for c in STYLO_FEATS       if c in first.columns]",
    "",
    "print('=== Per-batch row counts ===')",
    "for b, df in batches.items():",
    "    y = df['label_int'].values",
    "    n_cand = df['candidate_id'].nunique()",
    "    print(f'  {b}: n={len(df):4d}  cheat={int((y==1).sum()):3d}  honest={int((y==0).sum()):3d}  candidates={n_cand}')",
    "",
    "print()",
    "print('=== Feature dimensions ===')",
    "print(f'  Whisper whole-pretrained    : {len(WH_COLS)}')",
    "print(f'  WavLM whole-pretrained      : {len(WPRE_COLS)}')",
    "print(f'  WavLM whole-finetuned       : {len(WFT_COLS)}')",
    "print(f'  Text full / stylometric only: {len(TEXT_ALL)} / {len(TEXT_STYLO)}')",
])

# =========================================================================
md("""## 2. THE PROBLEM — baseline rotations on raw data (~2 hr)

We build the full base-model registry and run both rotations end-to-end with **StratifiedGroupKFold** (no candidate leakage), three repeated seeds, and 200 bootstrap iterations for the threshold CI.

Full registry: `whisper_wp_xgb`, `wavlm_whole_pre`, `wavlm_whole_ft` (if present), `text_all`, `text_stylo`.

Two threshold strategies per model:
- `F1` — picks the threshold that maximizes F1 on the CV set.
- `P90` — smallest threshold reaching CV precision ≥ 0.9, with at least 3 true positives.

Watch for the `cv_minus_test_gap` column. **Rotation A is expected to look fine. Rotation B is the problem** — that asymmetry is what the rest of this notebook investigates.
""")

code([
    "# ---------- XGB factory ----------",
    "def make_xgb(n_feats, seed=42, n_estimators=400):",
    "    colsample = 0.3 if n_feats > 500 else 0.8",
    "    return xgb.XGBClassifier(",
    "        n_estimators=n_estimators, max_depth=4, learning_rate=0.05,",
    "        subsample=0.8, colsample_bytree=colsample, min_child_weight=3,",
    "        scale_pos_weight=float(SPW_DEPLOY), eval_metric='logloss',",
    "        random_state=seed)",
    "",
    "def mk_X(cols): return lambda d: d[cols].fillna(0).values",
    "",
    "BASE_REGISTRY = {",
    "    'whisper_wp_xgb':  (mk_X(WH_COLS),   lambda s=42: make_xgb(len(WH_COLS), s)),",
    "    'wavlm_whole_pre': (mk_X(WPRE_COLS), lambda s=42: make_xgb(len(WPRE_COLS), s)) if WPRE_COLS else None,",
    "    'wavlm_whole_ft':  (mk_X(WFT_COLS),  lambda s=42: make_xgb(len(WFT_COLS), s))  if WFT_COLS  else None,",
    "    'text_all':        (mk_X(TEXT_ALL),   lambda s=42: make_xgb(len(TEXT_ALL), s)),",
    "    'text_stylo':      (mk_X(TEXT_STYLO), lambda s=42: make_xgb(len(TEXT_STYLO), s)),",
    "}",
    "BASE_REGISTRY = {k: v for k, v in BASE_REGISTRY.items() if v is not None}",
    "print(f'Models in registry: {list(BASE_REGISTRY)}')",
    "",
    "# ---------- Threshold pickers ----------",
    "def best_f1_thr(proba, y, grid=np.arange(0.20, 0.81, 0.01)):",
    "    bt, bf = 0.5, -1.0",
    "    for thr in grid:",
    "        f = f1_score(y, (proba >= thr).astype(int), zero_division=0)",
    "        if f > bf: bf, bt = f, float(thr)",
    "    return bt, bf",
    "",
    "def best_rec_at_prec(proba, y, target, min_tp=3):",
    "    best = None",
    "    for thr in np.arange(0.99, 0.10, -0.01):",
    "        pred = (proba >= thr).astype(int)",
    "        cm = confusion_matrix(y, pred, labels=[0, 1])",
    "        if cm[1, 1] < min_tp: continue",
    "        p = precision_score(y, pred, zero_division=0)",
    "        r = recall_score(y, pred, zero_division=0)",
    "        if p >= target and (best is None or r > best[1]):",
    "            best = (float(thr), float(r), float(p))",
    "    return best if best is not None else (None, None, None)",
    "",
    "def pick_thr(proba, y, strategy):",
    "    if strategy == 'F1':",
    "        thr, v = best_f1_thr(proba, y)",
    "        return thr, v",
    "    thr, rec, _ = best_rec_at_prec(proba, y, PREC_FLOOR[strategy])",
    "    return thr, rec",
    "",
    "def metrics_at(proba, y, thr):",
    "    if thr is None: return dict(prec=None, rec=None, f1=None)",
    "    pred = (proba >= thr).astype(int)",
    "    return dict(",
    "        prec=float(precision_score(y, pred, zero_division=0)),",
    "        rec =float(recall_score(y, pred, zero_division=0)),",
    "        f1  =float(f1_score(y, pred, zero_division=0)),",
    "    )",
    "",
    "# ---------- CV + bootstrap ----------",
    "def repeated_groupkfold_oof(df_cv, df_always, X_fn, factory, n_seeds=N_SEEDS, n_folds=N_FOLDS):",
    "    y_cv   = df_cv['label_int'].values",
    "    groups = df_cv['candidate_id'].values",
    "    oof_sum = np.zeros(len(df_cv)); oof_cnt = np.zeros(len(df_cv))",
    "    for s in range(n_seeds):",
    "        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42 + s)",
    "        for tr_idx, va_idx in skf.split(df_cv, y_cv, groups=groups):",
    "            df_tr_fold = df_cv.iloc[tr_idx]",
    "            df_va_fold = df_cv.iloc[va_idx]",
    "            df_tr = pd.concat([df_always, df_tr_fold], ignore_index=True) if df_always is not None and len(df_always) else df_tr_fold",
    "            Xtr = X_fn(df_tr); ytr = df_tr['label_int'].values",
    "            Xva = X_fn(df_va_fold)",
    "            sc = StandardScaler().fit(Xtr)",
    "            clf = factory(42 + s)",
    "            clf.fit(sc.transform(Xtr), ytr)",
    "            p = clf.predict_proba(sc.transform(Xva))[:, 1]",
    "            oof_sum[va_idx] += p",
    "            oof_cnt[va_idx] += 1",
    "    return oof_sum / np.maximum(oof_cnt, 1)",
    "",
    "def bootstrap_thr_ci(proba, y, strategy, n_boot=BOOT_N, seed=0):",
    "    rng = np.random.default_rng(seed)",
    "    thrs = []",
    "    for _ in range(n_boot):",
    "        b = rng.choice(np.arange(len(y)), size=len(y), replace=True)",
    "        if len(set(y[b].tolist())) < 2: continue",
    "        thr, _ = pick_thr(proba[b], y[b], strategy)",
    "        if thr is not None: thrs.append(thr)",
    "    if not thrs: return None, None, None",
    "    t = np.array(thrs)",
    "    return float(np.median(t)), float(np.percentile(t, 10)), float(np.percentile(t, 90))",
    "",
    "FRIENDLY_COLS = ['model_name','threshold_strategy','chosen_threshold',",
    "                 'threshold_ci_low','threshold_ci_high',",
    "                 'cv_score','test_score','test_precision','test_recall','cv_minus_test_gap']",
    "",
    "def run_full_rotation(view, train_folders, cv_target, test_folder, models=None, label=''):",
    "    models = models or list(BASE_REGISTRY)",
    "    df_cv  = view[cv_target].reset_index(drop=True)",
    "    df_al  = pd.concat([view[b] for b in train_folders if b != cv_target], ignore_index=True) \\",
    "             if any(b != cv_target for b in train_folders) else None",
    "    df_te  = view[test_folder].reset_index(drop=True)",
    "    y_cv   = df_cv['label_int'].values",
    "    y_te   = df_te['label_int'].values",
    "",
    "    print(f'  [{label}] cv={cv_target}({len(df_cv)})  always-train={[b for b in train_folders if b != cv_target]}  test={test_folder}({len(df_te)})')",
    "    rows = []",
    "    for name in models:",
    "        t0 = time.time()",
    "        X_fn, factory = BASE_REGISTRY[name]",
    "        oof = repeated_groupkfold_oof(df_cv, df_al, X_fn, factory)",
    "        df_tr_full = pd.concat([df_al, df_cv], ignore_index=True) if df_al is not None and len(df_al) else df_cv",
    "        Xtr = X_fn(df_tr_full); ytr = df_tr_full['label_int'].values",
    "        sc = StandardScaler().fit(Xtr)",
    "        clf = factory(42); clf.fit(sc.transform(Xtr), ytr)",
    "        p_te = clf.predict_proba(sc.transform(X_fn(df_te)))[:, 1]",
    "",
    "        for s in STRATEGIES:",
    "            thr_pt, cv_val = pick_thr(oof, y_cv, s)",
    "            thr_md, thr_lo, thr_hi = bootstrap_thr_ci(oof, y_cv, s)",
    "            thr_use = thr_md if thr_md is not None else thr_pt",
    "            te = metrics_at(p_te, y_te, thr_use)",
    "            te_val = te['f1'] if s == 'F1' else te['rec']",
    "            gap = (cv_val - te_val) if (cv_val is not None and te_val is not None) else None",
    "            rows.append({",
    "                'model_name':         name,",
    "                'threshold_strategy': s,",
    "                'chosen_threshold':   round(thr_use, 3) if thr_use is not None else None,",
    "                'threshold_ci_low':   round(thr_lo, 3)  if thr_lo  is not None else None,",
    "                'threshold_ci_high':  round(thr_hi, 3)  if thr_hi  is not None else None,",
    "                'cv_score':           round(cv_val, 3)  if cv_val  is not None else None,",
    "                'test_score':         round(te_val, 3)  if te_val  is not None else None,",
    "                'test_precision':     round(te['prec'], 3) if te['prec'] is not None else None,",
    "                'test_recall':        round(te['rec'], 3)  if te['rec']  is not None else None,",
    "                'cv_minus_test_gap':  round(gap, 3)       if gap        is not None else None,",
    "            })",
    "        print(f'    {name:18s} done in {time.time()-t0:5.1f}s')",
    "    return pd.DataFrame(rows)[FRIENDLY_COLS]",
])

code([
    "print('=' * 80)",
    "print(' SECTION 2  baseline rotations  (raw data, full registry)  -- expect ~2 hr')",
    "print('=' * 80)",
    "t_section = time.time()",
    "",
    "df_A_baseline = run_full_rotation(batches, ['audios2', 'audios4'], 'audios4', 'audios5', label='Rot A')",
    "df_B_baseline = run_full_rotation(batches, ['audios2', 'audios5'], 'audios5', 'audios4', label='Rot B')",
    "",
    "df_A_baseline.to_csv(SAVE_DIR / '2_baseline_rotation_A.csv', index=False)",
    "df_B_baseline.to_csv(SAVE_DIR / '2_baseline_rotation_B.csv', index=False)",
    "",
    "print()",
    "print('=== ROTATION A baseline (CV=audios4 test=audios5) ===')",
    "with pd.option_context('display.max_columns', None, 'display.width', 200):",
    "    print(df_A_baseline.to_string(index=False, na_rep='  --'))",
    "print()",
    "print('=== ROTATION B baseline (CV=audios5 test=audios4) ===')",
    "with pd.option_context('display.max_columns', None, 'display.width', 200):",
    "    print(df_B_baseline.to_string(index=False, na_rep='  --'))",
    "",
    "print()",
    "_a_f1 = df_A_baseline[df_A_baseline['threshold_strategy'] == 'F1'][['model_name','cv_minus_test_gap']]",
    "_b_f1 = df_B_baseline[df_B_baseline['threshold_strategy'] == 'F1'][['model_name','cv_minus_test_gap']]",
    "print('=== ROTATION GAP (F1 strategy) — Rot A vs Rot B ===')",
    "_cmp = _a_f1.merge(_b_f1, on='model_name', suffixes=('_rotA','_rotB'))",
    "print(_cmp.to_string(index=False))",
    "print()",
    "print(f'Section 2 took {(time.time()-t_section)/60:.1f} min')",
    "print(f'Saved: 2_baseline_rotation_A.csv  2_baseline_rotation_B.csv')",
])

# =========================================================================
md("""## 3. Diagnostic 1 — is one batch intrinsically harder?

The Rot B gap could mean one of two things:
- **(a)** `audios4` is *intrinsically* harder to classify than `audios5`. Then any model trained without seeing a4 would do worse on a4.
- **(b)** Including `audios5` in training *creates* the gap — the model learns something from a5 that hurts it on a4.

To tell them apart we train each base model on **`audios2` alone** (zero exposure to a4 or a5) and score both held-out batches. We compare prior-invariant metrics — AUC and PR-AUC — so the difference in positive rate between batches cannot fake an asymmetry.

If (a) were true we would expect **a4 AUC < a5 AUC**. If (b) is true we would expect them to be comparable, with a5 maybe slightly easier.
""")

code([
    "print('=' * 80)",
    "print(' SECTION 3  data-asymmetry diagnostic  (train on audios2 only)')",
    "print('=' * 80)",
    "",
    "DIAG_MODELS = ['whisper_wp_xgb', 'wavlm_whole_ft', 'text_stylo']",
    "DIAG_MODELS = [m for m in DIAG_MODELS if m in BASE_REGISTRY]",
    "",
    "def _train_on_audios2(name, X_fn, factory):",
    "    df_tr = batches['audios2']",
    "    Xtr   = X_fn(df_tr); ytr = df_tr['label_int'].values",
    "    sc    = StandardScaler().fit(Xtr)",
    "    clf   = factory(42); clf.fit(sc.transform(Xtr), ytr)",
    "    return clf, sc",
    "",
    "rows = []",
    "for name in DIAG_MODELS:",
    "    X_fn, factory = BASE_REGISTRY[name]",
    "    clf, sc = _train_on_audios2(name, X_fn, factory)",
    "    for tgt in ['audios4', 'audios5']:",
    "        df = batches[tgt]; y = df['label_int'].values",
    "        p = clf.predict_proba(sc.transform(X_fn(df)))[:, 1]",
    "        rows.append({",
    "            'model_name':       name,",
    "            'target_batch':     tgt,",
    "            'n':                len(df),",
    "            'positives':        int((y == 1).sum()),",
    "            'roc_auc':          round(float(roc_auc_score(y, p)), 3),",
    "            'pr_auc':           round(float(average_precision_score(y, p)), 3),",
    "            'mean_proba_pos':   round(float(p[y == 1].mean()), 3),",
    "            'mean_proba_neg':   round(float(p[y == 0].mean()), 3),",
    "        })",
    "",
    "diag_df = pd.DataFrame(rows)",
    "diag_df.to_csv(SAVE_DIR / '3_data_asymmetry.csv', index=False)",
    "print()",
    "with pd.option_context('display.max_columns', None, 'display.width', 160):",
    "    print(diag_df.to_string(index=False))",
    "",
    "print()",
    "print('Reading guide:')",
    "print('  - If audios4 AUC < audios5 AUC for every model -> a4 IS intrinsically harder (hypothesis a).')",
    "print('  - If they are comparable / a5 slightly higher  -> the Rot B gap is created by training on a5,')",
    "print('    not by a4 being harder. We then need to look at WHAT in a5 hurts the model.')",
    "print()",
    "print('Saved: 3_data_asymmetry.csv')",
])

# =========================================================================
md("""## 4. Hypothesis A — is `audios5` mislabeled in places?

If §3 told us the gap is created by *training* on `audios5`, the most obvious thing that could be wrong with a5 is the labels. We use the joint-disagreement test: score every a5 audio with **two** independent base models (whisper and finetuned WavLM, both trained on a2 alone) and flag rows where both models strongly disagree with the label.

Rule:
- **Labeled cheating but both models < 0.30** ⇒ likely honest (positive label suspect).
- **Labeled honest but both models > 0.70** ⇒ likely cheating (negative label suspect).

We save the full ranked list with absolute audio paths so re-listening is one click. Output rows are sorted with the most suspicious on top.
""")

code([
    "print('=' * 80)",
    "print(' SECTION 4  audios5 label-suspicion audit  (joint disagreement)')",
    "print('=' * 80)",
    "",
    "if 'wavlm_whole_ft' not in BASE_REGISTRY:",
    "    raise RuntimeError('wavlm_whole_ft not in BASE_REGISTRY -- joint audit needs both whisper and wavlm_ft.')",
    "",
    "def _score_audios5(name):",
    "    X_fn, factory = BASE_REGISTRY[name]",
    "    clf, sc = _train_on_audios2(name, X_fn, factory)",
    "    df = batches['audios5'].copy()",
    "    p = clf.predict_proba(sc.transform(X_fn(df)))[:, 1]",
    "    return df[['filename', 'candidate_id', 'label_int']].assign(proba=p)",
    "",
    "w = _score_audios5('whisper_wp_xgb')",
    "v = _score_audios5('wavlm_whole_ft')",
    "",
    "TARGET_BATCH = 'audios5'",
    "joint = w.rename(columns={'proba': 'whisper'}).merge(",
    "    v[['filename', 'proba']].rename(columns={'proba': 'wavlm_ft'}), on='filename')",
    "joint['avg_proba']   = (joint['whisper'] + joint['wavlm_ft']) / 2",
    "joint['min_proba']   = joint[['whisper', 'wavlm_ft']].min(axis=1)",
    "joint['max_proba']   = joint[['whisper', 'wavlm_ft']].max(axis=1)",
    "joint['audio_path']  = [str(NB_DIR / TARGET_BATCH / str(c) / str(f))",
    "                        for c, f in zip(joint['candidate_id'], joint['filename'])]",
    "",
    "POS_THR = 0.30",
    "NEG_THR = 0.70",
    "joint['suspect_type'] = ''",
    "joint.loc[(joint['label_int'] == 1) & (joint['max_proba'] < POS_THR), 'suspect_type'] = 'pos_label_likely_honest'",
    "joint.loc[(joint['label_int'] == 0) & (joint['min_proba'] > NEG_THR), 'suspect_type'] = 'neg_label_likely_cheating'",
    "",
    "n_pos_total = int((joint['label_int'] == 1).sum())",
    "n_neg_total = int((joint['label_int'] == 0).sum())",
    "n_pos_susp  = int((joint['suspect_type'] == 'pos_label_likely_honest').sum())",
    "n_neg_susp  = int((joint['suspect_type'] == 'neg_label_likely_cheating').sum())",
    "",
    "print()",
    "print(f'  audios5 total scored : {len(joint)}')",
    "print(f'  labeled cheating     : {n_pos_total}   suspect (both < {POS_THR}): {n_pos_susp}')",
    "print(f'  labeled honest       : {n_neg_total}   suspect (both > {NEG_THR}): {n_neg_susp}')",
    "",
    "save_cols = ['candidate_id','filename','label_int','whisper','wavlm_ft','avg_proba',",
    "             'min_proba','max_proba','suspect_type','audio_path']",
    "pos_df = joint[joint['label_int'] == 1].sort_values('min_proba')",
    "neg_df = joint[joint['label_int'] == 0].sort_values('max_proba', ascending=False)",
    "pos_df.to_csv(SAVE_DIR / '4_label_audit_positives.csv', columns=save_cols, index=False)",
    "neg_df.to_csv(SAVE_DIR / '4_label_audit_negatives.csv', columns=save_cols, index=False)",
    "joint.assign(disagree=(joint['label_int'] - joint['avg_proba']).abs()) \\",
    "     .sort_values('disagree', ascending=False) \\",
    "     .to_csv(SAVE_DIR / '4_label_audit_all.csv', columns=save_cols, index=False)",
    "",
    "print()",
    "print('Saved: 4_label_audit_positives.csv  4_label_audit_negatives.csv  4_label_audit_all.csv')",
    "print('Re-listen the top suspect rows; only confirmed mislabels should be flipped.')",
])

# =========================================================================
md("""## 5. Test of hypothesis A — apply known corrections, re-train, compare

We previously listened to the top suspects and confirmed exactly one mislabel (the rest were borderline, short, or the model was wrong). To test "label noise alone is the cause" honestly we apply that confirmed correction and re-train both rotations.

The corrections we previously verified are saved in `audios5GT_user_corrected.csv` (snapshotted in §0). This cell:
1. Diffs `audios5GT_user_corrected.csv` against the freshly-restored `audios5GT.csv` to find exactly which rows were corrected.
2. Applies those corrections to `audios5GT.csv`.
3. Reloads `batches['audios5']`.
4. Re-runs both rotations.
5. Prints the per-model delta vs the §2 baseline.

If label noise were the dominant cause of the Rot B gap, we'd see a large negative `gap_delta`. Spoiler from earlier work: the delta is small. That's what tells us we need to look elsewhere.
""")

code([
    "print('=' * 80)",
    "print(' SECTION 5  apply known correction, re-train, compare to baseline')",
    "print('=' * 80)",
    "",
    "_orig = pd.read_csv(GT_PATH)",
    "_corr = pd.read_csv(GT_USER_COPY)",
    "",
    "_fn_col_o  = next(c for c in _orig.columns if c.lower() in ('filename','file','name'))",
    "_lbl_col_o = next(c for c in _orig.columns if c.lower() in ('label','class','cheating','gt','label_int','ground_truth'))",
    "_fn_col_c  = next(c for c in _corr.columns if c.lower() in ('filename','file','name'))",
    "_lbl_col_c = next(c for c in _corr.columns if c.lower() in ('label','class','cheating','gt','label_int','ground_truth'))",
    "",
    "_o = _orig[[_fn_col_o, _lbl_col_o]].rename(columns={_fn_col_o: 'filename', _lbl_col_o: 'lbl_orig'})",
    "_c = _corr[[_fn_col_c, _lbl_col_c]].rename(columns={_fn_col_c: 'filename', _lbl_col_c: 'lbl_corr'})",
    "_diff = _o.merge(_c, on='filename', how='outer')",
    "_changed = _diff[_diff['lbl_orig'].astype(str) != _diff['lbl_corr'].astype(str)]",
    "print(f'  Rows changed by user vs baseline GT: {len(_changed)}')",
    "if len(_changed):",
    "    print(_changed.to_string(index=False))",
    "    shutil.copy(GT_USER_COPY, GT_PATH)",
    "    print(f'  Applied user corrections -> audios5GT.csv')",
    "else:",
    "    print('  No changes to apply -- baseline and user-corrected GT are identical for the label column.')",
    "    print('  Section 5 will still re-train and produce a CSV; expect ~zero delta.')",
    "",
    "# Reload audios5 only",
    "batches['audios5'] = attach_candidate_id(load_folder('audios5'))",
    "_y5 = batches['audios5']['label_int'].values",
    "print(f'  audios5 after relabel: n={len(_y5)}  cheat={int((_y5==1).sum())}  honest={int((_y5==0).sum())}')",
    "",
    "df_A_relabel = run_full_rotation(batches, ['audios2','audios4'], 'audios4', 'audios5', label='Rot A relabel')",
    "df_B_relabel = run_full_rotation(batches, ['audios2','audios5'], 'audios5', 'audios4', label='Rot B relabel')",
    "df_A_relabel.to_csv(SAVE_DIR / '5_after_relabel_rotation_A.csv', index=False)",
    "df_B_relabel.to_csv(SAVE_DIR / '5_after_relabel_rotation_B.csv', index=False)",
    "",
    "def _delta_table(label, base, after):",
    "    a = base[base['threshold_strategy'] == 'F1'][['model_name','cv_score','test_score','cv_minus_test_gap']]",
    "    b = after[after['threshold_strategy'] == 'F1'][['model_name','cv_score','test_score','cv_minus_test_gap']]",
    "    m = a.merge(b, on='model_name', suffixes=('_baseline','_relabel'))",
    "    m['cv_delta']  = (m['cv_score_relabel']         - m['cv_score_baseline']).round(3)",
    "    m['test_delta']= (m['test_score_relabel']       - m['test_score_baseline']).round(3)",
    "    m['gap_delta'] = (m['cv_minus_test_gap_relabel']- m['cv_minus_test_gap_baseline']).round(3)",
    "    print()",
    "    print('=' * 96)",
    "    print(f' {label}  (F1 strategy)')",
    "    print('=' * 96)",
    "    with pd.option_context('display.max_columns', None, 'display.width', 220):",
    "        print(m.to_string(index=False))",
    "    return m",
    "",
    "_delta_table('ROTATION A baseline -> relabel  (CV=a4 test=a5)', df_A_baseline, df_A_relabel)",
    "B_cmp = _delta_table('ROTATION B baseline -> relabel  (CV=a5 test=a4)', df_B_baseline, df_B_relabel)",
    "",
    "print()",
    "print('Read: gap_delta < 0 means the gap shrank. Big negative -> label noise was the cause.')",
    "print('      Small / near-zero -> label noise is NOT the dominant cause; keep searching.')",
    "print()",
    "print('Saved: 5_after_relabel_rotation_A.csv  5_after_relabel_rotation_B.csv')",
])

# =========================================================================
md("""## 6. Hypothesis B — is there a short-audio confound?

If label noise didn't explain it, the next thing to check is the audio itself. A very short answer (<30 s) tends to be acoustically flat and low-effort — easy to mistake for scripted speech regardless of the actual ground truth. If `audios2` has very few short audios but `audios4` and `audios5` both have many, then training on a5 teaches the model "short ⇒ cheating", which then misfires on the short rows in a4 (and vice versa). That mechanically produces the asymmetric Rot B gap.

This cell computes per-audio total duration and approximate speaking time from the ASR transcript JSONs (if present) or audio file directly, then reports the fraction of short audios per batch.
""")

code([
    "print('=' * 80)",
    "print(' SECTION 6  speaking-time per audio + per-batch short-audio share')",
    "print('=' * 80)",
    "",
    "try:",
    "    import soundfile as _sf",
    "    HAS_SF = True",
    "except ImportError:",
    "    HAS_SF = False",
    "    print('  soundfile not installed -- falling back to estimating from transcript JSON only')",
    "",
    "def _audio_duration(path):",
    "    if not HAS_SF or not Path(path).exists(): return np.nan",
    "    try:",
    "        info = _sf.info(str(path))",
    "        return float(info.frames / info.samplerate)",
    "    except Exception:",
    "        return np.nan",
    "",
    "def _speaking_time_from_transcript_json(json_path):",
    "    if not Path(json_path).exists(): return np.nan",
    "    try:",
    "        d = json.loads(Path(json_path).read_text(encoding='utf-8'))",
    "        segs = d.get('segments') or d.get('chunks') or []",
    "        if not segs: return np.nan",
    "        total = 0.0",
    "        for s in segs:",
    "            start = s.get('start', s.get('timestamp', [None, None])[0])",
    "            end   = s.get('end',   s.get('timestamp', [None, None])[1])",
    "            if start is None or end is None: continue",
    "            total += max(0.0, float(end) - float(start))",
    "        return float(total) if total > 0 else np.nan",
    "    except Exception:",
    "        return np.nan",
    "",
    "def _audio_path_for(batch, candidate_id, filename):",
    "    return NB_DIR / batch / str(candidate_id) / str(filename)",
    "",
    "def _transcript_path_for(batch, filename):",
    "    stem = Path(filename).stem",
    "    for cand in [NB_DIR / f'{batch}_transcripts' / f'{stem}.json',",
    "                 NB_DIR / batch / f'{stem}.json']:",
    "        if cand.exists(): return cand",
    "    return NB_DIR / f'{batch}_transcripts' / f'{stem}.json'",
    "",
    "duration_dfs = {}",
    "for b in BATCHES:",
    "    rows = []",
    "    for _, r in batches[b].iterrows():",
    "        ap = _audio_path_for(b, r['candidate_id'], r['filename'])",
    "        tp = _transcript_path_for(b, r['filename'])",
    "        td = _audio_duration(ap)",
    "        st = _speaking_time_from_transcript_json(tp)",
    "        if np.isnan(st) and not np.isnan(td):",
    "            st = td  # fallback: assume all of it is speech",
    "            src = 'audio_fallback'",
    "        elif not np.isnan(st):",
    "            src = 'transcript'",
    "        else:",
    "            src = 'unknown'",
    "        rows.append({",
    "            'filename': r['filename'], 'candidate_id': r['candidate_id'],",
    "            'label_int': int(r['label_int']),",
    "            'total_duration_s':  round(td, 2) if not np.isnan(td) else np.nan,",
    "            'speaking_time_s':   round(st, 2) if not np.isnan(st) else np.nan,",
    "            'speech_ratio':      round(st / td, 3) if (not np.isnan(st) and not np.isnan(td) and td > 0) else np.nan,",
    "            'source':            src,",
    "        })",
    "    df = pd.DataFrame(rows)",
    "    duration_dfs[b] = df",
    "    df.to_csv(SAVE_DIR / f'6_durations_{b}.csv', index=False)",
    "",
    "durations = duration_dfs   # alias for §7/§8",
    "",
    "print()",
    "print('=== Per-batch duration summary ===')",
    "summary = []",
    "for b, df in duration_dfs.items():",
    "    valid = df.dropna(subset=['speaking_time_s'])",
    "    summary.append({",
    "        'batch':              b,",
    "        'n_total':            len(df),",
    "        'n_valid_duration':   len(valid),",
    "        'speaking_s_mean':    round(valid['speaking_time_s'].mean(), 1),",
    "        'speaking_s_median':  round(valid['speaking_time_s'].median(), 1),",
    "        'pct_under_15s':      round(100 * (valid['speaking_time_s'] < 15).mean(), 1),",
    "        'pct_under_30s':      round(100 * (valid['speaking_time_s'] < 30).mean(), 1),",
    "    })",
    "summary_df = pd.DataFrame(summary)",
    "with pd.option_context('display.max_columns', None, 'display.width', 160):",
    "    print(summary_df.to_string(index=False))",
    "summary_df.to_csv(SAVE_DIR / '6_durations_summary.csv', index=False)",
    "",
    "print()",
    "print('Read: if pct_under_30s for a4 and a5 is several times higher than a2,')",
    "print('      the short-audio confound is plausible and worth filtering out at training time.')",
    "print()",
    "print('Saved: 6_durations_audios{2,4,5}.csv  6_durations_summary.csv')",
])

# =========================================================================
md("""## 7. Threshold sweep — at what minimum speaking-time does the gap stop shrinking?

Slim version of the rotation eval (no bootstrap, 1 seed, 3 folds, 200 trees, two-model subset) sweeping `MIN_SPEAKING_S` ∈ {0, 15, 20, 25, 30, 35}. The filter applies only to **training-eligible** rows in `audios2`/`audios4`/`audios5`; the held-out test batch stays unfiltered so the comparison is honest.

We're looking for one of three shapes:
- **Monotonic shrink** through 35 ⇒ stricter is strictly better; the confound is real and we haven't over-filtered yet. Keep raising.
- **U-shape** with a minimum somewhere in the middle ⇒ optimum found; past it we're cutting into real signal.
- **Sharp `cv_score` drop** at some threshold ⇒ training set has gone too small.
""")

code([
    "print('=' * 80)",
    "print(' SECTION 7  fast min-speaking-time sweep  (~15-20 min)')",
    "print('=' * 80)",
    "",
    "SWEEP_MODELS     = [m for m in ['whisper_wp_xgb', 'text_stylo'] if m in BASE_REGISTRY]",
    "SWEEP_THRESHOLDS = [0, 15, 20, 25, 30, 35]",
    "SWEEP_N_FOLDS    = 3",
    "SWEEP_N_SEEDS    = 1",
    "SWEEP_N_EST      = 200",
    "",
    "def _sw_filter(name, min_s):",
    "    if min_s <= 0: return batches[name]",
    "    d = durations[name][['filename','speaking_time_s']]",
    "    keep = set(d[(d['speaking_time_s'] >= min_s) | (d['speaking_time_s'].isna())]['filename'])",
    "    return batches[name][batches[name]['filename'].isin(keep)].reset_index(drop=True)",
    "",
    "def _sw_xgb(n_feats, seed):",
    "    cs = 0.3 if n_feats > 500 else 0.8",
    "    return xgb.XGBClassifier(",
    "        n_estimators=SWEEP_N_EST, max_depth=4, learning_rate=0.05,",
    "        subsample=0.8, colsample_bytree=cs, min_child_weight=3,",
    "        scale_pos_weight=float(SPW_DEPLOY), eval_metric='logloss',",
    "        random_state=seed)",
    "",
    "def _sw_rot(view, train_folders, cv_target, test_folder, model_names):",
    "    df_cv = view[cv_target].reset_index(drop=True)",
    "    y_cv  = df_cv['label_int'].values",
    "    grps  = df_cv['candidate_id'].values",
    "    df_al = pd.concat([view[b] for b in train_folders if b != cv_target], ignore_index=True) \\",
    "            if any(b != cv_target for b in train_folders) else None",
    "    df_te = view[test_folder].reset_index(drop=True)",
    "    y_te  = df_te['label_int'].values",
    "    rows = []",
    "    for name in model_names:",
    "        X_fn, _ = BASE_REGISTRY[name]",
    "        n_feat  = X_fn(df_cv.head(1)).shape[1]",
    "        oof_sum = np.zeros(len(df_cv)); oof_cnt = np.zeros(len(df_cv))",
    "        for s in range(SWEEP_N_SEEDS):",
    "            skf = StratifiedGroupKFold(n_splits=SWEEP_N_FOLDS, shuffle=True, random_state=42 + s)",
    "            for tr_idx, va_idx in skf.split(df_cv, y_cv, groups=grps):",
    "                df_tr = pd.concat([df_al, df_cv.iloc[tr_idx]], ignore_index=True) \\",
    "                        if df_al is not None else df_cv.iloc[tr_idx]",
    "                Xtr = X_fn(df_tr); ytr = df_tr['label_int'].values",
    "                Xva = X_fn(df_cv.iloc[va_idx])",
    "                sc  = StandardScaler().fit(Xtr)",
    "                clf = _sw_xgb(n_feat, 42 + s); clf.fit(sc.transform(Xtr), ytr)",
    "                oof_sum[va_idx] += clf.predict_proba(sc.transform(Xva))[:, 1]",
    "                oof_cnt[va_idx] += 1",
    "        oof = oof_sum / np.maximum(oof_cnt, 1)",
    "        df_tr_full = pd.concat([df_al, df_cv], ignore_index=True) if df_al is not None else df_cv",
    "        Xtr = X_fn(df_tr_full); ytr = df_tr_full['label_int'].values",
    "        sc = StandardScaler().fit(Xtr)",
    "        clf = _sw_xgb(n_feat, 42); clf.fit(sc.transform(Xtr), ytr)",
    "        p_te = clf.predict_proba(sc.transform(X_fn(df_te)))[:, 1]",
    "        thr, cv_f1 = best_f1_thr(oof, y_cv)",
    "        te = metrics_at(p_te, y_te, thr)",
    "        rows.append({'model_name': name, 'chosen_threshold': round(thr, 2),",
    "                     'cv_f1': round(cv_f1, 3), 'test_f1': round(te['f1'], 3),",
    "                     'cv_minus_test_gap': round(cv_f1 - te['f1'], 3)})",
    "    return pd.DataFrame(rows)",
    "",
    "results = []",
    "for min_s in SWEEP_THRESHOLDS:",
    "    t0 = time.time()",
    "    print(f'--- MIN_SPEAKING_S = {min_s}s ---')",
    "    view = dict(batches)",
    "    view['audios2'] = _sw_filter('audios2', min_s)",
    "    view['audios4'] = _sw_filter('audios4', min_s)",
    "    df_R_A = _sw_rot(view, ['audios2','audios4'], 'audios4', 'audios5', SWEEP_MODELS)",
    "    view = dict(batches)",
    "    view['audios2'] = _sw_filter('audios2', min_s)",
    "    view['audios5'] = _sw_filter('audios5', min_s)",
    "    df_R_B = _sw_rot(view, ['audios2','audios5'], 'audios5', 'audios4', SWEEP_MODELS)",
    "    for r in df_R_A.itertuples():",
    "        results.append({'min_speaking_s': min_s, 'rotation': 'A', 'model_name': r.model_name,",
    "                        'cv_f1': r.cv_f1, 'test_f1': r.test_f1, 'cv_minus_test_gap': r.cv_minus_test_gap})",
    "    for r in df_R_B.itertuples():",
    "        results.append({'min_speaking_s': min_s, 'rotation': 'B', 'model_name': r.model_name,",
    "                        'cv_f1': r.cv_f1, 'test_f1': r.test_f1, 'cv_minus_test_gap': r.cv_minus_test_gap})",
    "    print(f'  done in {time.time()-t0:.1f}s')",
    "",
    "sweep_df = pd.DataFrame(results)",
    "sweep_df.to_csv(SAVE_DIR / '7_threshold_sweep.csv', index=False)",
    "",
    "def _piv(metric):",
    "    return sweep_df.pivot_table(index=['model_name','rotation'], columns='min_speaking_s', values=metric).round(3)",
    "",
    "print()",
    "print('=== cv_minus_test_gap by (model, rotation) across MIN_SPEAKING_S ===')",
    "print(_piv('cv_minus_test_gap').to_string())",
    "print()",
    "print('=== test_f1 by (model, rotation) across MIN_SPEAKING_S ===')",
    "print(_piv('test_f1').to_string())",
    "print()",
    "print('=== cv_f1 by (model, rotation) across MIN_SPEAKING_S ===')",
    "print(_piv('cv_f1').to_string())",
    "print()",
    "print('Saved: 7_threshold_sweep.csv')",
])

# =========================================================================
md("""## 8. Final intervention — combined relabel + filter, dual test view (~2 hr)

Apply both fixes together with the full registry:
- Corrected `audios5GT.csv` (already in place from §5).
- Filter training rows at `MIN_SPEAKING_S = 30` (chosen from §7).
- Score each test batch in **two views**:
  - `test_full_*` — unfiltered. This is the deployment-realistic number — production won't pre-filter inputs.
  - `test_filtered_*` — same filter applied to test. This is the deployment-ceiling — the best the model could achieve if we also screened inputs.

The headline number is the change in `test_full_score` and `cv_minus_test_gap` for `whisper_wp_xgb` on Rot B compared to §2 baseline.
""")

code([
    "print('=' * 80)",
    "print(' SECTION 8  combined intervention  (corrected GT + min_speaking>=30, dual test)')",
    "print('=' * 80)",
    "t_section = time.time()",
    "",
    "MIN_SPEAKING_S = 30",
    "",
    "def _filter_view(name, min_s):",
    "    if min_s <= 0: return batches[name]",
    "    d = durations[name][['filename','speaking_time_s']]",
    "    keep = set(d[(d['speaking_time_s'] >= min_s) | (d['speaking_time_s'].isna())]['filename'])",
    "    return batches[name][batches[name]['filename'].isin(keep)].reset_index(drop=True)",
    "",
    "DUAL_COLS = ['model_name','threshold_strategy','chosen_threshold',",
    "             'threshold_ci_low','threshold_ci_high','cv_score',",
    "             'test_full_score','test_full_precision','test_full_recall','test_full_gap','n_test_full',",
    "             'test_filtered_score','test_filtered_precision','test_filtered_recall','test_filtered_gap','n_test_filtered']",
    "",
    "def run_dual_rotation(train_folders, cv_target, test_folder, label=''):",
    "    view = dict(batches)",
    "    for b in train_folders:",
    "        view[b] = _filter_view(b, MIN_SPEAKING_S)",
    "    df_cv  = view[cv_target].reset_index(drop=True)",
    "    df_al  = pd.concat([view[b] for b in train_folders if b != cv_target], ignore_index=True) \\",
    "             if any(b != cv_target for b in train_folders) else None",
    "    df_te_full = batches[test_folder].reset_index(drop=True)",
    "    df_te_filt = _filter_view(test_folder, MIN_SPEAKING_S).reset_index(drop=True)",
    "    y_cv      = df_cv['label_int'].values",
    "    y_te_full = df_te_full['label_int'].values",
    "    y_te_filt = df_te_filt['label_int'].values",
    "    print(f'  [{label}] cv={cv_target}({len(df_cv)})  test_full={test_folder}({len(df_te_full)})  test_filt={len(df_te_filt)}')",
    "",
    "    rows = []",
    "    for name in BASE_REGISTRY:",
    "        t0 = time.time()",
    "        X_fn, factory = BASE_REGISTRY[name]",
    "        oof = repeated_groupkfold_oof(df_cv, df_al, X_fn, factory)",
    "        df_tr_full = pd.concat([df_al, df_cv], ignore_index=True) if df_al is not None and len(df_al) else df_cv",
    "        Xtr = X_fn(df_tr_full); ytr = df_tr_full['label_int'].values",
    "        sc = StandardScaler().fit(Xtr)",
    "        clf = factory(42); clf.fit(sc.transform(Xtr), ytr)",
    "        p_te_full = clf.predict_proba(sc.transform(X_fn(df_te_full)))[:, 1]",
    "        p_te_filt = clf.predict_proba(sc.transform(X_fn(df_te_filt)))[:, 1]",
    "",
    "        for s in STRATEGIES:",
    "            thr_pt, cv_val = pick_thr(oof, y_cv, s)",
    "            thr_md, thr_lo, thr_hi = bootstrap_thr_ci(oof, y_cv, s)",
    "            thr_use = thr_md if thr_md is not None else thr_pt",
    "            te_f = metrics_at(p_te_full, y_te_full, thr_use)",
    "            te_l = metrics_at(p_te_filt, y_te_filt, thr_use)",
    "            te_full_val = te_f['f1'] if s == 'F1' else te_f['rec']",
    "            te_filt_val = te_l['f1'] if s == 'F1' else te_l['rec']",
    "            gap_full = (cv_val - te_full_val) if (cv_val is not None and te_full_val is not None) else None",
    "            gap_filt = (cv_val - te_filt_val) if (cv_val is not None and te_filt_val is not None) else None",
    "            rows.append({",
    "                'model_name':              name,",
    "                'threshold_strategy':      s,",
    "                'chosen_threshold':        round(thr_use, 3) if thr_use is not None else None,",
    "                'threshold_ci_low':        round(thr_lo, 3)  if thr_lo  is not None else None,",
    "                'threshold_ci_high':       round(thr_hi, 3)  if thr_hi  is not None else None,",
    "                'cv_score':                round(cv_val, 3)  if cv_val  is not None else None,",
    "                'test_full_score':         round(te_full_val, 3)  if te_full_val is not None else None,",
    "                'test_full_precision':     round(te_f['prec'], 3) if te_f['prec'] is not None else None,",
    "                'test_full_recall':        round(te_f['rec'], 3)  if te_f['rec']  is not None else None,",
    "                'test_full_gap':           round(gap_full, 3) if gap_full is not None else None,",
    "                'n_test_full':             len(df_te_full),",
    "                'test_filtered_score':     round(te_filt_val, 3)  if te_filt_val is not None else None,",
    "                'test_filtered_precision': round(te_l['prec'], 3) if te_l['prec'] is not None else None,",
    "                'test_filtered_recall':    round(te_l['rec'], 3)  if te_l['rec']  is not None else None,",
    "                'test_filtered_gap':       round(gap_filt, 3) if gap_filt is not None else None,",
    "                'n_test_filtered':         len(df_te_filt),",
    "            })",
    "        print(f'    {name:18s} done in {time.time()-t0:5.1f}s')",
    "    return pd.DataFrame(rows)[DUAL_COLS]",
    "",
    "df_A_final = run_dual_rotation(['audios2','audios4'], 'audios4', 'audios5', label='Rot A final')",
    "df_B_final = run_dual_rotation(['audios2','audios5'], 'audios5', 'audios4', label='Rot B final')",
    "df_A_final.to_csv(SAVE_DIR / '8_final_rotation_A.csv', index=False)",
    "df_B_final.to_csv(SAVE_DIR / '8_final_rotation_B.csv', index=False)",
    "",
    "print()",
    "print('=== ROTATION A final (CV=audios4 test=audios5, MIN_SPEAKING_S=30, corrected GT) ===')",
    "with pd.option_context('display.max_columns', None, 'display.width', 240):",
    "    print(df_A_final.to_string(index=False, na_rep='  --'))",
    "print()",
    "print('=== ROTATION B final (CV=audios5 test=audios4, MIN_SPEAKING_S=30, corrected GT) ===')",
    "with pd.option_context('display.max_columns', None, 'display.width', 240):",
    "    print(df_B_final.to_string(index=False, na_rep='  --'))",
    "print()",
    "print(f'Section 8 took {(time.time()-t_section)/60:.1f} min')",
    "print(f'Saved: 8_final_rotation_A.csv  8_final_rotation_B.csv')",
])

# =========================================================================
md("""## 9. Before / after summary

Pull the F1-strategy rows from the §2 baseline and the §8 final, line them up per model and per rotation, and report the deltas.

Headline columns:
- `cv_score_baseline` → `cv_score_final`
- `test_score_baseline` (full-test) → `test_full_score_final`
- `cv_minus_test_gap_baseline` → `test_full_gap_final`

Negative `gap_delta` = the gap shrank, which is what we wanted. Positive `test_score_delta` = absolute model performance also improved, not just the gap.
""")

code([
    "print('=' * 80)",
    "print(' SECTION 9  before/after summary')",
    "print('=' * 80)",
    "",
    "def _summary(label, base, final):",
    "    a = base [base ['threshold_strategy'] == 'F1'][['model_name','cv_score','test_score','cv_minus_test_gap']].copy()",
    "    b = final[final['threshold_strategy'] == 'F1'][['model_name','cv_score','test_full_score','test_full_gap','test_filtered_score','test_filtered_gap']].copy()",
    "    a = a.rename(columns={'cv_score':'cv_baseline','test_score':'test_baseline','cv_minus_test_gap':'gap_baseline'})",
    "    b = b.rename(columns={'cv_score':'cv_final','test_full_score':'test_full_final','test_full_gap':'gap_full_final',",
    "                          'test_filtered_score':'test_filtered_final','test_filtered_gap':'gap_filtered_final'})",
    "    m = a.merge(b, on='model_name')",
    "    m['cv_delta']        = (m['cv_final']       - m['cv_baseline']).round(3)",
    "    m['test_full_delta'] = (m['test_full_final']- m['test_baseline']).round(3)",
    "    m['gap_delta']       = (m['gap_full_final'] - m['gap_baseline']).round(3)",
    "    print()",
    "    print('=' * 100)",
    "    print(f' {label}')",
    "    print('=' * 100)",
    "    cols = ['model_name','cv_baseline','cv_final','cv_delta',",
    "            'test_baseline','test_full_final','test_full_delta',",
    "            'gap_baseline','gap_full_final','gap_delta',",
    "            'test_filtered_final','gap_filtered_final']",
    "    with pd.option_context('display.max_columns', None, 'display.width', 240):",
    "        print(m[cols].to_string(index=False))",
    "    return m[cols]",
    "",
    "sum_A = _summary('ROTATION A  baseline -> final  (CV=a4 test=a5)', df_A_baseline, df_A_final)",
    "sum_B = _summary('ROTATION B  baseline -> final  (CV=a5 test=a4)', df_B_baseline, df_B_final)",
    "",
    "sum_A.assign(rotation='A').to_csv(SAVE_DIR / '9_before_after_summary_rotation_A.csv', index=False)",
    "sum_B.assign(rotation='B').to_csv(SAVE_DIR / '9_before_after_summary_rotation_B.csv', index=False)",
    "",
    "print()",
    "print('=' * 100)",
    "print(' HEADLINE: whisper Rot B gap -- the gap this whole investigation was about')",
    "print('=' * 100)",
    "_w = sum_B[sum_B['model_name'] == 'whisper_wp_xgb']",
    "if len(_w):",
    "    _w = _w.iloc[0]",
    "    print(f'  whisper_wp_xgb Rot B (full-test view)')",
    "    print(f'    CV F1   baseline -> final : {_w[\"cv_baseline\"]:.3f} -> {_w[\"cv_final\"]:.3f}     (delta {_w[\"cv_delta\"]:+.3f})')",
    "    print(f'    test F1 baseline -> final : {_w[\"test_baseline\"]:.3f} -> {_w[\"test_full_final\"]:.3f}     (delta {_w[\"test_full_delta\"]:+.3f})')",
    "    print(f'    gap     baseline -> final : {_w[\"gap_baseline\"]:+.3f} -> {_w[\"gap_full_final\"]:+.3f}     (delta {_w[\"gap_delta\"]:+.3f})  <-- negative = gap shrunk')",
    "    print(f'    deployment ceiling (filtered test): F1 {_w[\"test_filtered_final\"]:.3f}  gap {_w[\"gap_filtered_final\"]:+.3f}')",
    "else:",
    "    print('  whisper_wp_xgb not in registry')",
    "",
    "print()",
    "print('Saved: 9_before_after_summary_rotation_A.csv  9_before_after_summary_rotation_B.csv')",
    "print()",
    "print('All artifacts in:', SAVE_DIR)",
])

# =========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')

# Final AST check on every code cell
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))

print(f'WROTE {NB.name} -- {len(nb["cells"])} cells, all code AST-valid')
