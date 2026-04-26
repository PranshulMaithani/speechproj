"""Build fusion_everything.ipynb from scratch in companylaptop/.
Runs once; safe to re-run (overwrites the target).
"""
import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'fusion_everything.ipynb'

def _id(): return uuid.uuid4().hex[:8]
def md(text):   return {'cell_type': 'markdown', 'metadata': {}, 'source': text, 'id': _id()}
def code(src):  return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src, 'id': _id()}

cells = []

# =====================================================================
# §0 — Title + overview
# =====================================================================
cells.append(md("""# Fusion: everything (8 bases × 2 train filters × 2 rotations × 2 test views)

Runs the full per-base + 2-way + 3-way fusion search across the cross-product:

- **8 base models**: `whisper_wp_xgb`, `wavlm_wp`, `wavlm_whole_ft`, `text_stylo`, `text_top10`, `text_top15`, `text_top20`, `text_all`
- **2 training conditions** per model: `unfilt` (no filter) and `filt` (only audios with `speaking_time_s >= 30`)
- **2 rotations**: A = (train [a2,a4], CV a4, test a5), B = (train [a2,a5], CV a5, test a4)
- **2 test views**: `_full` (deployment-realistic) and `_filt` (deployment-ceiling)

Threshold is picked on CV OOF (max-F1) and frozen, then applied to both test views.

## Outputs
- `checkpoints_everything/base_results.csv` — 32 rows (8 models × 2 train_filters × 2 rotations) with cv + dual test view + gap columns.
- `checkpoints_everything/fusion_2way_top20.csv` — top 20 by CV F1 per rotation, per train_filter (members must share train_filter).
- `checkpoints_everything/fusion_3way_top20.csv` — same shape for trios.
- `checkpoints_everything/frozen_configs.json` — weights + thresholds for the kept fusions.
- `checkpoints_everything/predictions_top_<rot>.csv` — per-file predictions (full audios{5,4}) for the top CV fusion of each rotation, with `speaking_time_s` and `in_filtered_view` columns.

## Reading guide
- `cv_F1` is OOF F1 on the CV target rows under that train_filter (filt: filtered val rows; unfilt: full val rows).
- `test_F1_full` = same threshold applied to the unfiltered test batch.
- `test_F1_filt` = same threshold applied to the filtered test batch.
- `gap_F1_full = cv_F1 - test_F1_full`. Positive = CV optimistic.
- `gap_F1_filt = cv_F1 - test_F1_filt`. The "ceiling" — what the model could do if test were also filtered.
"""))

# =====================================================================
# §1 — Setup
# =====================================================================
cells.append(md("## 0. Setup — config + imports"))

cells.append(code('''# ================================================================
# CONFIGURATION
# ================================================================
from pathlib import Path
import json, itertools, warnings, re, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score, average_precision_score)
warnings.filterwarnings('ignore')

NB_DIR        = Path('.').resolve()
SAVE_DIR      = NB_DIR / 'checkpoints_everything'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DURATIONS_DIR = NB_DIR / 'checkpoints_honest_eval'   # audios{2,4,5}_durations.csv

BATCHES   = ['audios2', 'audios4', 'audios5']
ROTATIONS = [
    # (rot_name, always_train_batches, cv_target, test_folder)
    ('A', ['audios2'], 'audios4', 'audios5'),
    ('B', ['audios2'], 'audios5', 'audios4'),
]

MIN_SPEAKING_S = 30
TRAIN_FILTERS  = ['unfilt', 'filt']    # the two model versions per base

DEPLOY_POS_RATE = 0.17
SPW_DEPLOY      = (1.0 - DEPLOY_POS_RATE) / DEPLOY_POS_RATE   # ~= 4.88

N_FOLDS       = 5
N_SEEDS       = 1                       # bump to 3 for tighter CV (~3x runtime)
RANDOM_SEED   = 42
TOP_K_FUSION  = 20                      # keep top-N per rotation per fusion arity
USE_GROUP_CV  = True

LABEL_MAP = {
    'read':1,'cheating':1,'reading':1,'scripted':1,'yes':1,'1':1,1:1,
    'spontaneous':0,'not cheating':0,'not_cheating':0,'no':0,'0':0,0:0,'genuine':0,
}

# Text feature catalog (for top-N ranking and text_all)
GROUPS = {
    'disfluency':  ['filler_rate','filler_count','repetition_rate','repair_rate',
                    'discourse_marker_rate','hedge_rate'],
    'stylometric': ['ttr','mattr','mtld','complex_word_rate','avg_word_length',
                    'n_words','n_unique_words','avg_sentence_length','std_sentence_length',
                    'fragment_rate','n_sentences','self_ref_rate',
                    'noun_rate','verb_rate','adj_rate'],
    'pause':       ['pause_mean','pause_std','pause_median','pause_skew','long_pause_rate',
                    'pause_ratio','n_pauses','pause_regularity',
                    'pause_before_content_ratio','pause_before_function_ratio',
                    'mid_phrase_pause_rate','words_per_sec','articulation_rate',
                    'initial_pause','longest_pause'],
    'formal_ai':   ['formal_transition_count','formal_transition_rate',
                    'ai_phrase_count','ai_phrase_rate'],
}
TEXT_RANK_GROUPS = ['stylometric', 'formal_ai', 'disfluency', 'pause']
TEXT_FEATURES_FOR_RANKING = [f for g in TEXT_RANK_GROUPS for f in GROUPS[g]]
STYLO_FEATS = GROUPS['stylometric']

ALL_TEXT_FEATURES = [
    # disfluency (6)
    'filler_rate','filler_count','repetition_rate','repair_rate',
    'discourse_marker_rate','hedge_rate',
    # stylometric (15)
    'ttr','mattr','mtld','complex_word_rate','avg_word_length','n_words',
    'n_unique_words','avg_sentence_length','std_sentence_length',
    'fragment_rate','n_sentences','self_ref_rate','noun_rate','verb_rate','adj_rate',
    # pause (15)
    'pause_mean','pause_std','pause_median','pause_skew','long_pause_rate',
    'pause_ratio','n_pauses','pause_regularity','pause_before_content_ratio',
    'pause_before_function_ratio','mid_phrase_pause_rate','words_per_sec',
    'articulation_rate','initial_pause','longest_pause',
    # suspicious (2)
    'suspicious_gap_count','suspicious_gap_ratio',
    # formal_ai (4)
    'formal_transition_count','formal_transition_rate',
    'ai_phrase_count','ai_phrase_rate',
    # prosodic (8)
    'f0_mean','f0_std','f0_range','f0_skew','f0_slope',
    'energy_mean','energy_std','speaking_rate_std',
    # voice_q (3)
    'jitter_local','shimmer_local','hnr_mean',
    # perplexity (2)
    'mean_perplexity','burstiness',
]

print(f'Save dir         : {SAVE_DIR}')
print(f'MIN_SPEAKING_S   : {MIN_SPEAKING_S}')
print(f'Train filters    : {TRAIN_FILTERS}')
print(f'Rotations        : {[r[0] for r in ROTATIONS]}')
print(f'CV folds / seeds : {N_FOLDS} / {N_SEEDS}')
print(f'SPW_DEPLOY       : {SPW_DEPLOY:.2f}')
'''))

# =====================================================================
# §1 — Load batches
# =====================================================================
cells.append(md("## 1. Load all three batches with candidate_id"))

cells.append(code('''_RE_CAND = re.compile(r'^(.+)_(\\d{1,3})\\.[a-zA-Z0-9]+$')
def _parse_candidate_id(fn):
    m = _RE_CAND.match(str(fn))
    return m.group(1) if m else None

def load_gt(name):
    gt = pd.read_csv(NB_DIR / f'{name}GT.csv')
    fn_col  = next(c for c in gt.columns if c.lower() in ('filename','file','name'))
    lbl_col = next(c for c in gt.columns if c.lower() in ('label','class','cheating','gt','label_int','ground_truth'))
    gt = gt.rename(columns={fn_col:'filename', lbl_col:'label_raw'})
    gt['label_int'] = gt['label_raw'].map(
        lambda x: LABEL_MAP.get(x, LABEL_MAP.get(str(x).lower().strip(), -1)))
    return gt[gt['label_int'].isin([0,1])][['filename','label_int']]

def _wavlm_wp_path(n):
    p = NB_DIR / f'{n}_whole_pretrained.csv'
    if p.exists(): return p
    alt = NB_DIR / f'{n}_wavlm_whole.csv'
    if alt.exists(): return alt
    raise FileNotFoundError(f'{p} or {alt}')

def _wavlm_wft_path(n):
    p = NB_DIR / f'{n}_whole_finetuned.csv'
    return p if p.exists() else None

def load_folder(name):
    gt      = load_gt(name)
    text    = pd.read_csv(NB_DIR / f'{name}_features.csv')
    wavlm   = pd.read_csv(_wavlm_wp_path(name))
    whisper = pd.read_csv(NB_DIR / f'{name}_whisper_whole.csv')
    df = (gt.merge(text,    on='filename', how='inner')
            .merge(wavlm,   on='filename', how='inner', suffixes=('','_wp'))
            .merge(whisper, on='filename', how='inner', suffixes=('','_wh')))
    wft_path = _wavlm_wft_path(name)
    if wft_path is not None:
        wft = pd.read_csv(wft_path)
        wft = wft.rename(columns={c: (c if c == 'filename' else f'{c}_wft') for c in wft.columns})
        df = df.merge(wft, on='filename', how='inner')
    df['candidate_id'] = df['filename'].astype(str).map(_parse_candidate_id)
    df['batch'] = name
    return df

batches_full = {b: load_folder(b) for b in BATCHES}

# Discover feature column lists from any batch (audios2 used)
_first = batches_full['audios2']
WP_COLS  = [c for c in _first.columns if c.startswith('wavlm_')
            and not c.startswith('wavlm_mean_') and not c.startswith('wavlm_std_')
            and not c.endswith('_wft')]
WFT_COLS = [c for c in _first.columns if c.endswith('_wft')]
WH_COLS  = [c for c in _first.columns if c.startswith('whisper_')]
TEXT_ALL_COLS    = [c for c in ALL_TEXT_FEATURES         if c in _first.columns]
TEXT_RANK_COLS   = [c for c in TEXT_FEATURES_FOR_RANKING if c in _first.columns]
TEXT_STYLO_COLS  = [c for c in STYLO_FEATS               if c in _first.columns]

print('Per-batch row counts (UNFILTERED):')
for b, df in batches_full.items():
    y = df['label_int'].values
    print(f'  {b}: n={len(df):4d}  cheat={int((y==1).sum()):3d}  honest={int((y==0).sum()):3d}  '
          f'candidates={df["candidate_id"].nunique()}')

print(f'\\nFeature dimensions:')
print(f'  whisper                : {len(WH_COLS)}')
print(f'  wavlm whole pretrained : {len(WP_COLS)}')
print(f'  wavlm whole finetuned  : {len(WFT_COLS)}')
print(f'  text_all / rank / stylo: {len(TEXT_ALL_COLS)} / {len(TEXT_RANK_COLS)} / {len(TEXT_STYLO_COLS)}')

if not WFT_COLS:
    print('\\n[WARN] wavlm_whole_ft features not found -- this base model will be skipped.')
'''))

# =====================================================================
# §2 — Filter views + durations
# =====================================================================
cells.append(md("""## 2. Build the filtered views (`speaking_time_s >= 30`)

Loads `audios{2,4,5}_durations.csv` from `checkpoints_honest_eval/` and produces a `batches_filt[b]` DataFrame for each batch — same schema as `batches_full[b]`, just with short audios removed.
"""))

cells.append(code('''def _load_durations(name):
    p = DURATIONS_DIR / f'{name}_durations.csv'
    if not p.exists():
        raise FileNotFoundError(
            f'Missing {p}. Run section 13 of honest_eval_and_improve.ipynb to generate the '
            f'audios{{2,4,5}}_durations.csv files.'
        )
    return pd.read_csv(p)[['filename', 'speaking_time_s']]

def _filter_df(df, dur, min_s):
    if min_s <= 0:
        return df.copy()
    keep = set(dur[(dur['speaking_time_s'] >= min_s) | (dur['speaking_time_s'].isna())]['filename'])
    return df[df['filename'].isin(keep)].reset_index(drop=True)

durations    = {b: _load_durations(b) for b in BATCHES}
batches_filt = {b: _filter_df(batches_full[b], durations[b], MIN_SPEAKING_S) for b in BATCHES}

print(f'Filter applied at speaking_time_s >= {MIN_SPEAKING_S} s')
print()
print(f'{"batch":<10}  {"n_full":>7}  {"n_filt":>7}  {"%kept":>7}   '
      f'{"cheat_full":>10}  {"cheat_filt":>10}')
for b in BATCHES:
    n_full = len(batches_full[b]); n_filt = len(batches_filt[b])
    cf = int((batches_full[b]["label_int"] == 1).sum())
    ck = int((batches_filt[b]["label_int"] == 1).sum())
    pct = 100.0 * n_filt / max(n_full, 1)
    print(f'{b:<10}  {n_full:>7}  {n_filt:>7}  {pct:>6.1f}%   {cf:>10}  {ck:>10}')

# Convenience map: train_filter -> {batch: df}
def view_for(train_filter):
    return batches_filt if train_filter == 'filt' else batches_full
'''))

# =====================================================================
# §3 — Top-N text feature ranking
# =====================================================================
cells.append(md("""## 3. Freeze top-10 / top-15 / top-20 text feature lists

XGB importance computed on `audios2` only (always-train batch in both rotations), **unfiltered** so the ranking is stable across train conditions. Same lists used for both rotations and both train_filters.
"""))

cells.append(code('''_df_rank = batches_full['audios2']
_X_rank  = _df_rank[TEXT_RANK_COLS].fillna(0).values
_y_rank  = _df_rank['label_int'].values
_sc_rank = StandardScaler().fit(_X_rank)
_rkr = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    scale_pos_weight=float(SPW_DEPLOY), eval_metric='logloss',
    random_state=RANDOM_SEED)
_rkr.fit(_sc_rank.transform(_X_rank), _y_rank)
_imp = pd.Series(_rkr.feature_importances_, index=TEXT_RANK_COLS).sort_values(ascending=False)

TOP10_FEATS = _imp.head(10).index.tolist()
TOP15_FEATS = _imp.head(15).index.tolist()
TOP20_FEATS = _imp.head(20).index.tolist()

print('Top-20 text features by audios2-only XGB importance:')
for i, f in enumerate(TOP20_FEATS, 1):
    mark = ''
    if i == 11: mark = '   <- top-15 boundary'
    if i == 16: mark = '   <- top-20 boundary (within top-15 slice)'
    print(f'  {i:2d}. {f:32s}  imp={_imp[f]:.4f}{mark}')
'''))

# =====================================================================
# §4 — Base model registry
# =====================================================================
cells.append(md("""## 4. Base model registry (8 models, all XGBoost)

All heads use `scale_pos_weight = SPW_DEPLOY` (deployment-prior loss reweighting). Same hyperparameters across all bases except `colsample_bytree` is reduced for the high-dimensional acoustic embeddings.
"""))

cells.append(code('''def make_xgb(n_feats, seed=RANDOM_SEED):
    cs = 0.3 if n_feats > 500 else 0.8
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=cs, min_child_weight=3,
        scale_pos_weight=float(SPW_DEPLOY), eval_metric='logloss',
        random_state=seed)

def _mk_X(cols):
    return lambda d: d[cols].fillna(0).values

# (X_fn, factory) for each base model. Skip wavlm_whole_ft if features missing.
BASE_REGISTRY = {
    'whisper_wp_xgb': (_mk_X(WH_COLS),         lambda s=RANDOM_SEED: make_xgb(len(WH_COLS), s)),
    'wavlm_wp':       (_mk_X(WP_COLS),         lambda s=RANDOM_SEED: make_xgb(len(WP_COLS), s)),
    'text_stylo':     (_mk_X(TEXT_STYLO_COLS), lambda s=RANDOM_SEED: make_xgb(len(TEXT_STYLO_COLS), s)),
    'text_top10':     (_mk_X(TOP10_FEATS),     lambda s=RANDOM_SEED: make_xgb(len(TOP10_FEATS), s)),
    'text_top15':     (_mk_X(TOP15_FEATS),     lambda s=RANDOM_SEED: make_xgb(len(TOP15_FEATS), s)),
    'text_top20':     (_mk_X(TOP20_FEATS),     lambda s=RANDOM_SEED: make_xgb(len(TOP20_FEATS), s)),
    'text_all':       (_mk_X(TEXT_ALL_COLS),   lambda s=RANDOM_SEED: make_xgb(len(TEXT_ALL_COLS), s)),
}
if WFT_COLS:
    BASE_REGISTRY['wavlm_whole_ft'] = (_mk_X(WFT_COLS), lambda s=RANDOM_SEED: make_xgb(len(WFT_COLS), s))

MODELS = list(BASE_REGISTRY.keys())
print(f'Base models ({len(MODELS)}):')
for m in MODELS:
    n = BASE_REGISTRY[m][0](batches_full["audios2"].head(1)).shape[1]
    print(f'  {m:18s}  n_feats={n}')
'''))

# =====================================================================
# §5 — Eval helpers
# =====================================================================
cells.append(md("## 5. Threshold + metric helpers"))

cells.append(code('''def best_f1_thr(proba, y, grid=np.arange(0.20, 0.81, 0.01)):
    bt, bf = 0.5, -1.0
    for thr in grid:
        f = f1_score(y, (proba >= thr).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(thr)
    return bt, bf

def metrics_at(proba, y, thr):
    pred = (proba >= thr).astype(int)
    return dict(
        prec = float(precision_score(y, pred, zero_division=0)),
        rec  = float(recall_score   (y, pred, zero_division=0)),
        f1   = float(f1_score       (y, pred, zero_division=0)),
    )

def fit_and_score(X_fn, factory, df_tr, df_te, seed=RANDOM_SEED):
    Xt, yt = X_fn(df_tr), df_tr['label_int'].values
    sc = StandardScaler().fit(Xt)
    m  = factory(seed); m.fit(sc.transform(Xt), yt)
    return m.predict_proba(sc.transform(X_fn(df_te)))[:, 1], (sc, m)

def fit_full(X_fn, factory, df_tr, seed=RANDOM_SEED):
    """Fit and return (scaler, model). Used for downstream test scoring."""
    Xt, yt = X_fn(df_tr), df_tr['label_int'].values
    sc = StandardScaler().fit(Xt)
    m  = factory(seed); m.fit(sc.transform(Xt), yt)
    return sc, m

def score_with(model_pair, X_fn, df):
    sc, m = model_pair
    return m.predict_proba(sc.transform(X_fn(df)))[:, 1]
'''))

# =====================================================================
# §6 — CV runner per (rot, train_filter, model) -- this is the heavy step
# =====================================================================
cells.append(md("""## 6. Run CV for every (rotation, train_filter, model)

Protocol A — `train_filter='filt'` filters BOTH the train and val sides of CV (deployment-ceiling surface). `train_filter='unfilt'` uses no filter anywhere. After CV, the final model is fit on `(always_train + cv_target)` under the same filter — this is what scores the test batch in §7.

Heavy step. With `N_SEEDS=1, N_FOLDS=5`, this runs `len(MODELS) * 2 train_filters * 2 rotations * 5 folds + final fits` = roughly 200 XGBoost fits. Estimated 30-60 min depending on hardware.
"""))

cells.append(code('''def _repeated_groupkfold_oof(df_cv, df_always, X_fn, factory):
    y_cv   = df_cv['label_int'].values
    groups = df_cv['candidate_id'].values
    oof_sum = np.zeros(len(df_cv)); oof_cnt = np.zeros(len(df_cv))
    use_group = USE_GROUP_CV and pd.notna(groups).all()
    for s in range(N_SEEDS):
        if use_group:
            skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED + s)
            it  = skf.split(df_cv, y_cv, groups=groups)
        else:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED + s)
            it  = skf.split(df_cv, y_cv)
        for tr_idx, va_idx in it:
            df_tr_fold = df_cv.iloc[tr_idx]
            df_va_fold = df_cv.iloc[va_idx]
            df_tr      = pd.concat([df_always, df_tr_fold], ignore_index=True) \\
                         if df_always is not None and len(df_always) else df_tr_fold
            p, _       = fit_and_score(X_fn, factory, df_tr, df_va_fold, seed=RANDOM_SEED + s)
            oof_sum[va_idx] += p
            oof_cnt[va_idx] += 1
    return oof_sum / np.maximum(oof_cnt, 1)

def run_one(rot_name, always_train, cv_target, test_folder, train_filter):
    view  = view_for(train_filter)
    df_al = pd.concat([view[b] for b in always_train], ignore_index=True) if always_train else None
    df_cv = view[cv_target].reset_index(drop=True)
    y_cv  = df_cv['label_int'].values

    # Fixed test views (always reference batches_full for full and batches_filt for filt)
    df_te_full = batches_full[test_folder].reset_index(drop=True)
    df_te_filt = batches_filt [test_folder].reset_index(drop=True)
    y_te_full  = df_te_full['label_int'].values
    y_te_filt  = df_te_filt['label_int'].values

    print(f'  [rot={rot_name} train_filter={train_filter:>6}] '
          f'cv={cv_target}({len(df_cv)}) always={always_train}({len(df_al) if df_al is not None else 0}) '
          f'te_full={test_folder}({len(df_te_full)}) te_filt({len(df_te_filt)})')

    oof, p_te_full, p_te_filt = {}, {}, {}
    for name in MODELS:
        t0 = time.time()
        X_fn, factory = BASE_REGISTRY[name]
        oof_arr = _repeated_groupkfold_oof(df_cv, df_al, X_fn, factory)
        # Final fit on full available train (always + cv_target) under the same filter.
        df_tr_full = pd.concat([df_al, df_cv], ignore_index=True) if df_al is not None and len(df_al) else df_cv
        sc, m = fit_full(X_fn, factory, df_tr_full)
        p_full = m.predict_proba(sc.transform(X_fn(df_te_full)))[:, 1]
        p_filt = m.predict_proba(sc.transform(X_fn(df_te_filt)))[:, 1]
        oof[name]       = oof_arr
        p_te_full[name] = p_full
        p_te_filt[name] = p_filt
        print(f'      {name:18s} done in {time.time()-t0:5.1f}s')
    return {
        'rot': rot_name, 'train_filter': train_filter,
        'always_train': always_train, 'cv_target': cv_target, 'test_folder': test_folder,
        'cv_y': y_cv,
        'cv_filenames': df_cv['filename'].values,
        'te_y_full':  y_te_full, 'te_filenames_full': df_te_full['filename'].values,
        'te_y_filt':  y_te_filt, 'te_filenames_filt': df_te_filt['filename'].values,
        'oof':        oof,        # {model: arr aligned to cv_y}
        'p_te_full':  p_te_full,  # {model: arr aligned to te_y_full}
        'p_te_filt':  p_te_filt,  # {model: arr aligned to te_y_filt}
    }

ALL = {}   # {(rot, train_filter): result-dict}

t_total = time.time()
for rot_name, always_train, cv_target, test_folder in ROTATIONS:
    print()
    print('=' * 100)
    print(f' ROTATION {rot_name}  always={always_train}  cv={cv_target}  test={test_folder}')
    print('=' * 100)
    for tf in TRAIN_FILTERS:
        ALL[(rot_name, tf)] = run_one(rot_name, always_train, cv_target, test_folder, tf)
print(f'\\nAll CV + final fits done in {(time.time()-t_total)/60:.1f} min.')
'''))

# =====================================================================
# §7 — Per-base table
# =====================================================================
cells.append(md("""## 7. Per-base results table

One row per `(rotation, model_name, train_filter)`. The 2x2 cross of `train_filter × test_filter` is two adjacent column blocks (`*_full`, `*_filt`) on the same row.
"""))

cells.append(code('''def _build_base_row(rot, train_filter, model):
    R = ALL[(rot, train_filter)]
    oof, y_cv  = R['oof'][model],       R['cv_y']
    p_full, y_full = R['p_te_full'][model], R['te_y_full']
    p_filt, y_filt = R['p_te_filt'][model], R['te_y_filt']
    thr, cv_F1 = best_f1_thr(oof, y_cv)
    m_full = metrics_at(p_full, y_full, thr)
    m_filt = metrics_at(p_filt, y_filt, thr)
    return {
        'rotation':            rot,
        'model_name':          model,
        'train_filter':        train_filter,
        'chosen_threshold':    round(thr, 3),
        'cv_F1':               round(cv_F1, 4),
        'n_cv':                int(len(y_cv)),
        'n_cv_pos':            int((y_cv == 1).sum()),
        'test_F1_full':        round(m_full['f1'], 4),
        'test_prec_full':      round(m_full['prec'], 4),
        'test_rec_full':       round(m_full['rec'], 4),
        'gap_F1_full':         round(cv_F1 - m_full['f1'], 4),
        'n_test_full':         int(len(y_full)),
        'test_F1_filt':        round(m_filt['f1'], 4),
        'test_prec_filt':      round(m_filt['prec'], 4),
        'test_rec_filt':       round(m_filt['rec'], 4),
        'gap_F1_filt':         round(cv_F1 - m_filt['f1'], 4),
        'n_test_filt':         int(len(y_filt)),
    }

base_rows = []
for rot_name, *_ in ROTATIONS:
    for tf in TRAIN_FILTERS:
        for model in MODELS:
            base_rows.append(_build_base_row(rot_name, tf, model))
base_df = pd.DataFrame(base_rows)

print('=== Per-base results (sorted by rotation, train_filter, cv_F1 desc) ===')
_show = base_df.sort_values(['rotation','train_filter','cv_F1'], ascending=[True, True, False])
with pd.option_context('display.max_columns', None, 'display.width', 240):
    print(_show.to_string(index=False))

base_df.to_csv(SAVE_DIR / 'base_results.csv', index=False)
print(f'\\nSaved: {SAVE_DIR / "base_results.csv"}')
'''))

# =====================================================================
# §8 — 2-way fusion search
# =====================================================================
cells.append(md("""## 8. 2-way fusion search (within same train_filter)

For each `(rotation, train_filter)`, sweep all `C(8,2) = 28` pairs at weight grid step `0.05` on CV OOF, pick the (weights, threshold) that maximise CV F1, then score on both test views. Top `TOP_K_FUSION` per rotation kept.
"""))

cells.append(code('''def _grid_2way(step=0.05):
    return [(round(w, 2), round(1 - w, 2)) for w in np.arange(0.0, 1.001, step)]

def _search_2way(R):
    y_cv = R['cv_y']
    out = []
    names = list(R['oof'])
    for a, b in itertools.combinations(names, 2):
        best = None
        for wa, wb in _grid_2way(step=0.05):
            p = wa * R['oof'][a] + wb * R['oof'][b]
            thr, f1 = best_f1_thr(p, y_cv)
            if best is None or f1 > best['cv_F1']:
                best = {'members': [a,b], 'weights': [wa,wb], 'thr': thr,
                        'cv_F1': f1, 'p_cv': p}
        out.append(best)
    return out

def _eval_fusion_on_test(R, members, weights, thr):
    p_full = sum(w * R['p_te_full'][m] for w, m in zip(weights, members))
    p_filt = sum(w * R['p_te_filt'][m] for w, m in zip(weights, members))
    m_full = metrics_at(p_full, R['te_y_full'], thr)
    m_filt = metrics_at(p_filt, R['te_y_filt'], thr)
    return p_full, p_filt, m_full, m_filt

print('=== 2-way fusion search ===')
fusion_2way_rows = []
for rot_name, *_ in ROTATIONS:
    rot_pool = []
    for tf in TRAIN_FILTERS:
        R = ALL[(rot_name, tf)]
        for cfg in _search_2way(R):
            _, _, m_full, m_filt = _eval_fusion_on_test(R, cfg['members'], cfg['weights'], cfg['thr'])
            rot_pool.append({
                'rotation':         rot_name,
                'train_filter':     tf,
                'members':          '+'.join(cfg['members']),
                'weights':          str(cfg['weights']),
                'chosen_threshold': round(cfg['thr'], 3),
                'cv_F1':            round(cfg['cv_F1'], 4),
                'test_F1_full':     round(m_full['f1'], 4),
                'test_prec_full':   round(m_full['prec'], 4),
                'test_rec_full':    round(m_full['rec'], 4),
                'gap_F1_full':      round(cfg['cv_F1'] - m_full['f1'], 4),
                'test_F1_filt':     round(m_filt['f1'], 4),
                'test_prec_filt':   round(m_filt['prec'], 4),
                'test_rec_filt':    round(m_filt['rec'], 4),
                'gap_F1_filt':      round(cfg['cv_F1'] - m_filt['f1'], 4),
            })
    rot_pool.sort(key=lambda r: -r['cv_F1'])
    fusion_2way_rows.extend(rot_pool[:TOP_K_FUSION])
    print(f'  rotation {rot_name}: kept top {TOP_K_FUSION} of {len(rot_pool)} (cv_F1 sorted desc)')

fusion_2way_df = pd.DataFrame(fusion_2way_rows)
fusion_2way_df.to_csv(SAVE_DIR / 'fusion_2way_top20.csv', index=False)

print()
with pd.option_context('display.max_columns', None, 'display.width', 240):
    print(fusion_2way_df.to_string(index=False))
print(f'\\nSaved: {SAVE_DIR / "fusion_2way_top20.csv"}')
'''))

# =====================================================================
# §9 — 3-way fusion search
# =====================================================================
cells.append(md("""## 9. 3-way fusion search (within same train_filter)

`C(8,3) = 56` trios per `(rotation, train_filter)`. Simplex weight grid at step `0.1` (~66 points per trio). Same protocol as §8.
"""))

cells.append(code('''def _grid_3way(step=0.1):
    out = []
    for w1 in np.arange(0, 1.001, step):
        for w2 in np.arange(0, 1.001 - w1 + 1e-9, step):
            w3 = max(0.0, 1.0 - w1 - w2)
            out.append((round(float(w1), 2), round(float(w2), 2), round(float(w3), 2)))
    return out

def _search_3way(R):
    y_cv = R['cv_y']
    grid = _grid_3way(step=0.1)
    out  = []
    names = list(R['oof'])
    for trio in itertools.combinations(names, 3):
        best = None
        for w in grid:
            p = w[0]*R['oof'][trio[0]] + w[1]*R['oof'][trio[1]] + w[2]*R['oof'][trio[2]]
            thr, f1 = best_f1_thr(p, y_cv)
            if best is None or f1 > best['cv_F1']:
                best = {'members': list(trio), 'weights': list(w), 'thr': thr,
                        'cv_F1': f1, 'p_cv': p}
        out.append(best)
    return out

print('=== 3-way fusion search ===')
fusion_3way_rows = []
for rot_name, *_ in ROTATIONS:
    rot_pool = []
    for tf in TRAIN_FILTERS:
        t0 = time.time()
        R = ALL[(rot_name, tf)]
        for cfg in _search_3way(R):
            _, _, m_full, m_filt = _eval_fusion_on_test(R, cfg['members'], cfg['weights'], cfg['thr'])
            rot_pool.append({
                'rotation':         rot_name,
                'train_filter':     tf,
                'members':          '+'.join(cfg['members']),
                'weights':          str(cfg['weights']),
                'chosen_threshold': round(cfg['thr'], 3),
                'cv_F1':            round(cfg['cv_F1'], 4),
                'test_F1_full':     round(m_full['f1'], 4),
                'test_prec_full':   round(m_full['prec'], 4),
                'test_rec_full':    round(m_full['rec'], 4),
                'gap_F1_full':      round(cfg['cv_F1'] - m_full['f1'], 4),
                'test_F1_filt':     round(m_filt['f1'], 4),
                'test_prec_filt':   round(m_filt['prec'], 4),
                'test_rec_filt':    round(m_filt['rec'], 4),
                'gap_F1_filt':      round(cfg['cv_F1'] - m_filt['f1'], 4),
            })
        print(f'  rot {rot_name} train_filter={tf}: {time.time()-t0:.1f}s')
    rot_pool.sort(key=lambda r: -r['cv_F1'])
    fusion_3way_rows.extend(rot_pool[:TOP_K_FUSION])

fusion_3way_df = pd.DataFrame(fusion_3way_rows)
fusion_3way_df.to_csv(SAVE_DIR / 'fusion_3way_top20.csv', index=False)

print()
with pd.option_context('display.max_columns', None, 'display.width', 240):
    print(fusion_3way_df.to_string(index=False))
print(f'\\nSaved: {SAVE_DIR / "fusion_3way_top20.csv"}')
'''))

# =====================================================================
# §10 — Headline summary panels
# =====================================================================
cells.append(md("""## 10. Headline summary panels

Per rotation, prints:
- Best base model by `test_F1_full`, per train_filter.
- Best 2-way and 3-way fusion by `test_F1_full`, per train_filter.
- Best by `cv_F1` per train_filter (the actual deployment candidate — selection is on CV).
- The four diagonal cells of the 2x2 (train × test) cross at the top base, so you can see whether `train=filt → test=full` is the deployment-risk case.
"""))

cells.append(code('''def _best(df, key, **filters):
    sub = df.copy()
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    if sub.empty: return None
    return sub.sort_values(key, ascending=False).iloc[0]

def _row_str(r, *cols):
    if r is None: return '  (none)'
    return '   '.join(f'{c}={r[c]}' for c in cols)

print('=' * 100)
print(' HEADLINE SUMMARY (per rotation)')
print('=' * 100)

for rot_name, *_ in ROTATIONS:
    print(f'\\n----- ROTATION {rot_name} -----')
    for tf in TRAIN_FILTERS:
        print(f'  train_filter = {tf}')
        b1 = _best(base_df,        'test_F1_full', rotation=rot_name, train_filter=tf)
        f2 = _best(fusion_2way_df, 'test_F1_full', rotation=rot_name, train_filter=tf)
        f3 = _best(fusion_3way_df, 'test_F1_full', rotation=rot_name, train_filter=tf)
        b1cv = _best(base_df,        'cv_F1', rotation=rot_name, train_filter=tf)
        f2cv = _best(fusion_2way_df, 'cv_F1', rotation=rot_name, train_filter=tf)
        f3cv = _best(fusion_3way_df, 'cv_F1', rotation=rot_name, train_filter=tf)

        print(f'    Best base    by test_F1_full : {b1["model_name"] if b1 is not None else "--":18s}  '
              f'cv_F1={b1["cv_F1"] if b1 is not None else "--"}  '
              f'test_F1_full={b1["test_F1_full"] if b1 is not None else "--"}  '
              f'gap_F1_full={b1["gap_F1_full"] if b1 is not None else "--"}')
        print(f'    Best 2-way   by test_F1_full : {f2["members"] if f2 is not None else "--":40s}  '
              f'cv_F1={f2["cv_F1"] if f2 is not None else "--"}  '
              f'test_F1_full={f2["test_F1_full"] if f2 is not None else "--"}')
        print(f'    Best 3-way   by test_F1_full : {f3["members"] if f3 is not None else "--":40s}  '
              f'cv_F1={f3["cv_F1"] if f3 is not None else "--"}  '
              f'test_F1_full={f3["test_F1_full"] if f3 is not None else "--"}')
        print(f'    Best base    by cv_F1        : {b1cv["model_name"] if b1cv is not None else "--":18s}  '
              f'cv_F1={b1cv["cv_F1"] if b1cv is not None else "--"}  '
              f'test_F1_full={b1cv["test_F1_full"] if b1cv is not None else "--"}')
        print(f'    Best 2-way   by cv_F1        : {f2cv["members"] if f2cv is not None else "--":40s}  '
              f'cv_F1={f2cv["cv_F1"] if f2cv is not None else "--"}  '
              f'test_F1_full={f2cv["test_F1_full"] if f2cv is not None else "--"}')
        print(f'    Best 3-way   by cv_F1        : {f3cv["members"] if f3cv is not None else "--":40s}  '
              f'cv_F1={f3cv["cv_F1"] if f3cv is not None else "--"}  '
              f'test_F1_full={f3cv["test_F1_full"] if f3cv is not None else "--"}')

# 2x2 cross at the top base per rotation (selected on cv_F1, train_filter='unfilt' as anchor)
print('\\n' + '=' * 100)
print(' 2x2 (train_filter x test_view) cross — top base by cv_F1 per rotation')
print('=' * 100)
for rot_name, *_ in ROTATIONS:
    sub = base_df[base_df['rotation'] == rot_name].sort_values('cv_F1', ascending=False)
    if sub.empty: continue
    top_model = sub.iloc[0]['model_name']
    rows = base_df[(base_df['rotation'] == rot_name) & (base_df['model_name'] == top_model)]
    print(f'\\nRotation {rot_name}  top model = {top_model}')
    print(rows[['train_filter','cv_F1','test_F1_full','gap_F1_full','test_F1_filt','gap_F1_filt']].to_string(index=False))
'''))

# =====================================================================
# §11 — Save artifacts (per-file predictions + frozen configs)
# =====================================================================
cells.append(md("""## 11. Save artifacts

- `frozen_configs.json` — the kept top-N fusion configs from §8 / §9 (members, weights, threshold, cv_F1, gaps).
- `predictions_top_<rot>.csv` — per-file predictions for the top CV fusion of each rotation, on the **full** test view, with `speaking_time_s` and `in_filtered_view` columns so error analysis can sort by audio length.
"""))

cells.append(code('''# --- frozen_configs.json ---
frozen = {
    'base_results':       base_df.to_dict(orient='records'),
    'fusion_2way_top20':  fusion_2way_df.to_dict(orient='records'),
    'fusion_3way_top20':  fusion_3way_df.to_dict(orient='records'),
}
with open(SAVE_DIR / 'frozen_configs.json', 'w') as f:
    json.dump(frozen, f, indent=2, default=str)
print(f'Saved: {SAVE_DIR / "frozen_configs.json"}')

# --- per-rotation top-fusion predictions (full test view) ---
def _top_fusion_per_rotation():
    """For each rotation, pick the highest-cv_F1 fusion across both train_filters and both arities.
    Returns dict rot -> dict with all the bits needed to reconstruct the prediction array.
    """
    out = {}
    pool = pd.concat([
        fusion_2way_df.assign(arity=2),
        fusion_3way_df.assign(arity=3),
    ], ignore_index=True)
    for rot_name, *_ in ROTATIONS:
        sub = pool[pool['rotation'] == rot_name].sort_values('cv_F1', ascending=False)
        if sub.empty: continue
        out[rot_name] = sub.iloc[0].to_dict()
    return out

top_per_rot = _top_fusion_per_rotation()

for rot_name, top in top_per_rot.items():
    R = ALL[(rot_name, top['train_filter'])]
    members = top['members'].split('+')
    weights = json.loads(top['weights'].replace("'", '"'))
    thr     = float(top['chosen_threshold'])
    p_full  = sum(w * R['p_te_full'][m] for w, m in zip(weights, members))
    pred    = (p_full >= thr).astype(int)

    # Per-file dataframe
    fn_full = R['te_filenames_full']
    y_full  = R['te_y_full']
    dur     = durations[R['test_folder']].set_index('filename')['speaking_time_s']
    dur_aligned = dur.reindex(fn_full).values

    review = pd.DataFrame({
        'filename':         fn_full,
        'current_gt':       y_full.astype(int),
        'speaking_time_s':  dur_aligned,
        'in_filtered_view': dur_aligned >= MIN_SPEAKING_S,
        'score':            np.round(p_full, 4),
        'pred':             pred,
    })
    for m in members:
        review[f'p_{m}'] = np.round(R['p_te_full'][m], 4)
    review['error_type'] = np.where(
        review['current_gt'] == review['pred'], 'CORRECT',
        np.where(review['current_gt'] == 0, 'FP', 'FN')
    )
    review = review.sort_values('score', ascending=False).reset_index(drop=True)
    review['rank'] = review.index + 1

    cols = ['rank','filename','current_gt','speaking_time_s','in_filtered_view',
            'score','pred','error_type'] + [f'p_{m}' for m in members]
    out_path = SAVE_DIR / f'predictions_top_{rot_name}.csv'
    review[cols].to_csv(out_path, index=False)
    print(f'\\n[{rot_name}] top fusion = {top["members"]}  weights={top["weights"]}  thr={top["chosen_threshold"]}  '
          f'train_filter={top["train_filter"]}  cv_F1={top["cv_F1"]}  '
          f'test_F1_full={top["test_F1_full"]}  gap_F1_full={top["gap_F1_full"]}')
    print(f'   per-file predictions -> {out_path}')

print('\\nAll artifacts saved to:', SAVE_DIR)
'''))

# =====================================================================
# Write notebook
# =====================================================================
nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Wrote {OUT}  ({len(cells)} cells)')
