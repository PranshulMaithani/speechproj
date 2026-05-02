#!/usr/bin/env python3
"""
Generates audios6_tier_b.ipynb.
Run:  python _build_audios6_tier_b.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / 'audios6_tier_b.ipynb'


def _c(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None,
            "source": src.strip('\n')}


def _m(src, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.strip('\n')}


C = []

# -----------------------------------------------------------------------------
# §0  CONFIG + IMPORTS
# -----------------------------------------------------------------------------
C.append(_m("# Tier B — content-feature ablation, prosodic-only base, GT audit\n\n"
            "Continuation of Tier A. Reads Tier A artifacts (read-only) from "
            "`checkpoints_tier_a/`. All new outputs go to `checkpoints_tier_b/`.", "md-tb-00"))

C.append(_m("## §0 — Config + Imports", "md-00"))

C.append(_c(r"""
import os, re, time, warnings, json, itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score)
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── constants ──────────────────────────────────────────────────────────────
SPW_DEPLOY        = (1 - 0.17) / 0.17   # 4.882...
RANDOM_SEED       = 42
N_FOLDS           = 5
MIN_SPEAKING_S    = 30
CALIB_SLICE_SIZE  = 50
N_REPEATS         = 20
CALIB_SEEDS       = list(range(N_REPEATS))      # 0-19, fixed across sections
AUDIT_SEED        = 42                          # single deterministic K for §6

NB_DIR     = Path('.').resolve()
TIER_A_DIR = NB_DIR / 'checkpoints_tier_a'
SAVE_DIR   = NB_DIR / 'checkpoints_tier_b'
SAVE_DIR.mkdir(exist_ok=True)

LABEL_MAP = {
    'read':1,'cheating':1,'reading':1,'scripted':1,'yes':1,'1':1, 1:1,
    'spontaneous':0,'not cheating':0,'not_cheating':0,'no':0,'0':0, 0:0,'genuine':0,
}
_RE = re.compile(r'^(.+)_(\d{1,3})\.[a-zA-Z0-9]+$')

print(f'NB_DIR:     {NB_DIR}')
print(f'TIER_A_DIR: {TIER_A_DIR}  exists={TIER_A_DIR.exists()}')
print(f'SAVE_DIR:   {SAVE_DIR}')
print(f'SPW_DEPLOY: {SPW_DEPLOY:.4f}')
""", "code-00"))

# ── feature groups (same as Tier A) ──
C.append(_c("""
FEAT_DISFLUENCY  = ['filler_rate','filler_count','repetition_rate','repair_rate',
                     'discourse_marker_rate','hedge_rate']
FEAT_STYLOMETRIC = ['ttr','mattr','mtld','complex_word_rate','avg_word_length','n_words',
                     'n_unique_words','avg_sentence_length','std_sentence_length',
                     'fragment_rate','n_sentences','self_ref_rate','noun_rate','verb_rate','adj_rate']
FEAT_PAUSE       = ['pause_mean','pause_std','pause_median','pause_skew','long_pause_rate',
                     'pause_ratio','n_pauses','pause_regularity','pause_before_content_ratio',
                     'pause_before_function_ratio','mid_phrase_pause_rate','words_per_sec',
                     'articulation_rate','initial_pause','longest_pause']
FEAT_FORMAL_AI   = ['formal_transition_count','formal_transition_rate',
                     'ai_phrase_count','ai_phrase_rate']
FEAT_PROSODIC    = ['f0_mean','f0_std','f0_range','f0_skew','f0_slope',
                     'energy_mean','energy_std','speaking_rate_std']
FEAT_VOICE_Q     = ['jitter_local','shimmer_local','hnr_mean']
FEAT_PERPLEXITY  = ['mean_perplexity','burstiness']
FEAT_SUSPICIOUS  = ['suspicious_gap_count','suspicious_gap_ratio']

FEAT_ALL_TEXT = (FEAT_DISFLUENCY + FEAT_STYLOMETRIC + FEAT_PAUSE +
                 FEAT_FORMAL_AI + FEAT_PROSODIC + FEAT_VOICE_Q +
                 FEAT_PERPLEXITY + FEAT_SUSPICIOUS)
FEAT_STYLO    = FEAT_STYLOMETRIC

# Content-invariant prosodic candidate set per Tier B brief.
# pause_before_{content,function}_ratio reference content categories so
# they're content-conditional — excluded explicitly.
FEAT_PROSODIC_BASE_CANDIDATES = [
    # pause group
    'pause_mean','pause_std','pause_median','pause_skew','long_pause_rate',
    'pause_ratio','n_pauses','pause_regularity','mid_phrase_pause_rate',
    'words_per_sec','articulation_rate','initial_pause','longest_pause',
    # voicing/F0
    'f0_mean','f0_std','f0_range','f0_skew','f0_slope',
    # energy
    'energy_mean','energy_std','speaking_rate_std',
    # voice quality
    'jitter_local','shimmer_local','hnr_mean',
    # disfluency
    'filler_rate','repetition_rate','repair_rate','hedge_rate',
]

print(f'Canonical text features: {len(FEAT_ALL_TEXT)}')
print(f'Prosodic-base candidates: {len(FEAT_PROSODIC_BASE_CANDIDATES)}')
""", "code-01"))

# ── utility functions (copied from Tier A; self-contained per brief) ──
C.append(_c(r"""
# ── data loading helpers ───────────────────────────────────────────────────

def load_gt(batch):
    p = NB_DIR / f'{batch}GT.csv'
    if not p.exists():
        raise FileNotFoundError(f'GT not found: {p}')
    gt = pd.read_csv(p)
    fn  = next((c for c in gt.columns if c.lower() in
                ('filename','file','name','audio','audio_file')), None)
    lbl = next((c for c in gt.columns if c.lower() in
                ('label','class','cheating','gt','label_int','ground_truth','label_raw')), None)
    if fn is None or lbl is None:
        raise ValueError(f'Cannot detect filename/label cols in {p}. Cols: {list(gt.columns)}')
    gt = gt.rename(columns={fn: 'filename', lbl: 'label_raw'})
    gt['label_int'] = gt['label_raw'].map(
        lambda x: LABEL_MAP.get(x, LABEL_MAP.get(str(x).lower().strip(), -1)))
    reg = next((c for c in gt.columns
                if c.lower() in ('region','country','locale','location')), None)
    if reg:
        gt['region'] = gt[reg].astype(str).str.upper().str.strip()
    keep = ['filename', 'label_int'] + (['region'] if reg else [])
    return gt[gt['label_int'].isin([0, 1])][keep].copy()


def attach_ids(df):
    df = df.copy()
    m   = df['filename'].map(lambda f: _RE.match(str(f)))
    bad = m.isna().sum()
    if bad / max(len(df), 1) > 0.05:
        raise ValueError(f'Filename regex failed {bad}/{len(df)} rows (>5%)')
    df['candidate_id'] = m.map(lambda x: x.group(1) if x else None)
    df['question_id']  = m.map(lambda x: int(x.group(2)) if x else None)
    return df


def load_embeddings(batch, kind):
    if kind == 'wavlm':
        cands = [f'{batch}_wavlm_whole.csv', f'{batch}_whole_pretrained.csv']
        pfx, bad = 'wavlm_', ('wavlm_mean_', 'wavlm_std_')
    else:
        cands = [f'{batch}_whisper_whole.csv']
        pfx, bad = 'whisper_', ()
    for cname in cands:
        p = NB_DIR / cname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        fn = next((c for c in df.columns if c.lower() in ('filename','file','name')), None)
        if fn:
            df = df.rename(columns={fn: 'filename'})
        cols = [c for c in df.columns
                if c.startswith(pfx) and not any(c.startswith(b) for b in bad)]
        if not cols:
            cols = [c for c in df.columns
                    if c != 'filename' and pd.api.types.is_numeric_dtype(df[c])]
        return df[['filename'] + cols], cols
    return None, []


def load_batch(batch, require_gt=True):
    gt        = load_gt(batch) if require_gt else None
    feat_path = NB_DIR / f'{batch}_features.csv'
    feat      = pd.read_csv(feat_path) if feat_path.exists() else None
    if feat is not None:
        fn = next((c for c in feat.columns if c.lower() in ('filename','file','name')), None)
        if fn:
            feat = feat.rename(columns={fn: 'filename'})
    wlm_df, wlm_cols = load_embeddings(batch, 'wavlm')
    wsp_df, wsp_cols = load_embeddings(batch, 'whisper')

    base = gt if gt is not None else (feat[['filename']] if feat is not None
                                      else wlm_df[['filename']] if wlm_df is not None
                                      else None)
    if base is None:
        raise ValueError(f'No data found for {batch}')
    df = base
    if feat   is not None: df = df.merge(feat,   on='filename', how='inner')
    if wlm_df is not None: df = df.merge(wlm_df, on='filename', how='inner')
    if wsp_df is not None: df = df.merge(wsp_df, on='filename', how='inner')
    df = attach_ids(df)
    return df, wlm_cols, wsp_cols


# ── model helpers ──────────────────────────────────────────────────────────

def make_xgb(n_feats=0, seed=RANDOM_SEED):
    cs = 0.3 if n_feats > 500 else 0.8
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=cs, min_child_weight=3,
        scale_pos_weight=float(SPW_DEPLOY),
        eval_metric='logloss', random_state=seed,
        use_label_encoder=False, verbosity=0)


def avail(cols, df):
    return [c for c in cols if c in df.columns]


def Xy(df, cols):
    c = avail(cols, df)
    y = df['label_int'].values if 'label_int' in df.columns else None
    return df[c].fillna(0).values, y, c


def fit_model(model, feat_cols, df_train):
    X, y, used = Xy(df_train, feat_cols)
    m = deepcopy(model)
    m.fit(X, y)
    return m, used


def predict_proba(model, used_cols, df):
    X, _, _ = Xy(df, used_cols)
    return model.predict_proba(X)[:, 1]


def a4_oof(model, feat_cols, df_a4, df_a2):
    df_a4 = df_a4.reset_index(drop=True)
    oof   = np.full(len(df_a4), np.nan)
    y     = df_a4['label_int'].values
    grps  = df_a4['candidate_id'].fillna('_unk').values
    sgkf  = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for tr_i, va_i in sgkf.split(df_a4, y, groups=grps):
        tr = pd.concat([df_a2, df_a4.iloc[tr_i]], ignore_index=True)
        va = df_a4.iloc[va_i]
        m, uc = fit_model(model, feat_cols, tr)
        oof[va_i] = predict_proba(m, uc, va)
    return oof, y


# ── metric helpers ─────────────────────────────────────────────────────────

def sweep_thr(y, s, lo=0.01, hi=0.99, step=0.01):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(lo, hi + step / 2, step):
        f = f1_score(y, (s >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return round(best_t, 3), round(best_f, 4)


def metrics_at(y, s, thr):
    # Return F1/P/R/AUC at threshold; sanity-check P/R against sklearn F1.
    p   = (s >= thr).astype(int)
    f1  = f1_score(y, p, zero_division=0)
    pr  = precision_score(y, p, zero_division=0)
    rc  = recall_score(y, p, zero_division=0)
    # P/R sanity check (Tier A had bugs here)
    if pr + rc > 0:
        f1_from_pr = 2 * pr * rc / (pr + rc)
        if abs(f1_from_pr - f1) > 1e-6:
            print(f'    WARN P/R/F1 mismatch: F1={f1:.6f} 2PR/(P+R)={f1_from_pr:.6f}')
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float('nan')
    return {'F1': round(f1, 4), 'P': round(pr, 4), 'R': round(rc, 4),
            'AUC': round(auc, 4), 'n_pos': int(y.sum()), 'n_eval': len(y)}


def bootstrap_ci(y, sa, ta, sb, tb, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    N   = len(y)
    d   = []
    for _ in range(n):
        i  = rng.choice(N, N, replace=True)
        fa = f1_score(y[i], (sa[i] >= ta).astype(int), zero_division=0)
        fb = f1_score(y[i], (sb[i] >= tb).astype(int), zero_division=0)
        d.append(fa - fb)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(d))


def fused(members, weights, score_dict):
    w  = np.array(weights, dtype=float)
    present = [m for m in members if m in score_dict]
    pw = np.array([w[i] for i, m in enumerate(members) if m in score_dict])
    pw = pw / pw.sum()
    s  = sum(pw[i] * score_dict[m] for i, m in enumerate(present))
    if len(present) < len(members):
        miss = set(members) - set(present)
        print(f'  WARN fusion missing {miss}; renormalized.')
    return s


def topk_predict(scores, slice_idx, y_a6):
    # Top-K% rule on remaining audios, K estimated from labelled slice.
    n_a6     = len(y_a6)
    rem_mask = np.ones(n_a6, bool); rem_mask[slice_idx] = False
    rem_idx  = np.where(rem_mask)[0]
    if len(rem_idx) == 0:
        return None, None, None
    K_hat = y_a6[slice_idx].mean()
    s_rem = scores[rem_idx]
    y_rem = y_a6[rem_idx]
    thr_k = float(np.percentile(s_rem, 100.0 * (1.0 - K_hat)))
    pred  = (s_rem >= thr_k).astype(int)
    return rem_idx, pred, thr_k


print('Utilities loaded.')
""", "code-02"))

# -----------------------------------------------------------------------------
# §1  LOAD + MERGE ALL BATCHES
# -----------------------------------------------------------------------------
C.append(_m("## §1 — Load + Merge All Batches", "md-01"))

C.append(_c("""
t0 = time.time()

BATCHES = ['audios2', 'audios4', 'audios5', 'audios6']
BD      = {}
WLM_COLS, WSP_COLS, TXT_COLS = [], [], []

for bat in BATCHES:
    try:
        df, wc, sc = load_batch(bat, require_gt=True)
        BD[bat] = df
        if not WLM_COLS and wc: WLM_COLS = wc
        if not WSP_COLS and sc: WSP_COLS = sc
        nc = df['label_int'].sum()
        nh = (df['label_int'] == 0).sum()
        print(f'\\n{bat}: {len(df)} rows | cheat={nc} honest={nh} '
              f'cands={df.candidate_id.nunique()} qs={df.question_id.nunique()}')
        if 'region' in df.columns:
            print(f'  regions: {df.region.value_counts().to_dict()}')
    except FileNotFoundError as e:
        print(f'  SKIP {bat}: {e}')

# detect text feature columns
for bat, df in BD.items():
    tc = [c for c in FEAT_ALL_TEXT if c in df.columns]
    if tc:
        TXT_COLS = tc
        print(f'\\nText cols from {bat}: {len(tc)}')
        break
if not TXT_COLS:
    print('WARNING: no canonical text cols found')

# optional duration filter
DUR_DIR = NB_DIR / 'checkpoints_honest_eval'
for bat, df in BD.items():
    p = DUR_DIR / f'{bat}_durations.csv'
    if not p.exists():
        if bat == 'audios2':
            print(f'WARNING: duration CSV missing for {bat}; unfiltered')
        continue
    dur = pd.read_csv(p)
    fn  = next((c for c in dur.columns if 'file' in c.lower()), None)
    dc  = next((c for c in dur.columns if 'dur' in c.lower() or 'speak' in c.lower()), None)
    if fn and dc:
        dur = dur.rename(columns={fn: 'filename', dc: 'dur'})
        pre = len(df)
        df  = df.merge(dur[['filename', 'dur']], on='filename', how='left')
        df  = df[df['dur'].isna() | (df['dur'] >= MIN_SPEAKING_S)].drop(columns=['dur'])
        BD[bat] = df.reset_index(drop=True)
        print(f'  {bat}: duration filter removed {pre - len(df)} rows')

TRAIN_POOL = pd.concat(
    [BD.get('audios2'), BD.get('audios4'), BD.get('audios5')],
    ignore_index=True).dropna(subset=['label_int'])
print(f'\\nTRAIN_POOL: {len(TRAIN_POOL)} rows | cheat={int(TRAIN_POOL.label_int.sum())}')
print(f'WavLM {len(WLM_COLS)}d  Whisper {len(WSP_COLS)}d  Text {len(TXT_COLS)} feats')

# ── question-id overlap matrix ────────────────────────────────────────────
print('\\nQuestion-id overlap matrix (rows=batch, cols=qid):')
q_rows = []
for bat in BATCHES:
    if bat not in BD: continue
    vc = BD[bat]['question_id'].value_counts()
    q_rows.append({'batch': bat, **{int(k): int(v) for k, v in vc.items()}})
qm = pd.DataFrame(q_rows).set_index('batch').fillna(0).astype(int).sort_index(axis=1)
print(qm.to_string())
qm.to_csv(SAVE_DIR / 'question_overlap_matrix.csv')

# explicit a6-vs-train overlap
if 'audios6' in BD:
    a6_qs    = set(BD['audios6']['question_id'].dropna().unique())
    train_qs = set(TRAIN_POOL['question_id'].dropna().unique())
    print(f'\\na6 questions:     {sorted(a6_qs)}')
    print(f'train questions:  {sorted(train_qs)}')
    print(f'a6 ∩ train:       {sorted(a6_qs & train_qs)}')
    print(f'a6 \\\\ train:      {sorted(a6_qs - train_qs)}  (NEW in a6)')

print(f'\\nSection 1 done in {time.time()-t0:.1f}s')
""", "code-10"))

# -----------------------------------------------------------------------------
# §2  Baseline references from Tier A
# -----------------------------------------------------------------------------
C.append(_m("## §2 — Baseline references (Tier A floor)", "md-02"))

C.append(_c("""
t0 = time.time()

df_a2 = BD.get('audios2', pd.DataFrame())
df_a4 = BD.get('audios4', pd.DataFrame())
df_a6 = BD.get('audios6', pd.DataFrame())
HAS_A6 = len(df_a6) > 0
y_a6_full = df_a6['label_int'].values if HAS_A6 else np.array([])

# ── Tier A best fusion scores on a6 (the floor any Tier B condition must beat) ──
TIER_A_BEST_NAME = 'A2A3_combined_topk'
TIER_A_BEST_F1   = float('nan')
TIER_A_BEST_A6   = None      # per-audio combined score
TIER_A_PICK_A6   = {}        # pick_name -> per-audio score (baselines)

# 1) load combined a6 scores produced by Tier A §6
comb_p = TIER_A_DIR / 'combined_a6_scores.csv'
if comb_p.exists():
    cdf = pd.read_csv(comb_p)
    # align with df_a6 row order via filename
    if HAS_A6 and 'filename' in cdf.columns and 'score_combined' in cdf.columns:
        merged = df_a6[['filename']].merge(cdf[['filename','score_combined']],
                                            on='filename', how='left')
        TIER_A_BEST_A6 = merged['score_combined'].values
        print(f'Loaded Tier A combined a6 scores: {comb_p}')
        miss = pd.isna(TIER_A_BEST_A6).sum()
        if miss > 0:
            print(f'  WARN {miss}/{len(TIER_A_BEST_A6)} rows missing combined score')
else:
    print(f'WARN: {comb_p} not found — Tier A floor unavailable from disk.')

# 2) Tier A summary table for floor F1
ts_p = TIER_A_DIR / 'tier_a_summary.csv'
if ts_p.exists():
    ta = pd.read_csv(ts_p)
    overall = ta[ta['region'] == 'overall'].sort_values('F1', ascending=False)
    if len(overall) > 0:
        TIER_A_BEST_F1   = float(overall.iloc[0]['F1'])
        TIER_A_BEST_NAME = f"{overall.iloc[0]['condition']}/{overall.iloc[0]['pick_name']}"
        print(f'\\nTier A best (from summary): {TIER_A_BEST_NAME}  F1={TIER_A_BEST_F1:.4f}')
    print('\\nTier A overall top 5:')
    print(overall[['condition','pick_name','F1','P','R','AUC']].head(5).to_string(index=False))
else:
    print(f'WARN: {ts_p} not found.')

# ── load η² table ────────────────────────────────────────────────────────
ETA2 = {}
eta_p = TIER_A_DIR / 'feature_variance_explained.csv'
if eta_p.exists():
    edf = pd.read_csv(eta_p)
    ETA2 = dict(zip(edf['feature'], edf['eta_squared']))
    print(f'\\nLoaded η² for {len(ETA2)} features. Top 10:')
    print(edf.head(10).to_string(index=False))
else:
    print(f'WARN: {eta_p} not found — DROP_FEATS will use brief defaults only.')

# ── shared calibration slices (matches Tier A §3) ────────────────────────
CALIB_SLICE_IDX = {}
if HAS_A6:
    n_a6 = len(y_a6_full)
    pos_idx = np.where(y_a6_full == 1)[0]
    neg_idx = np.where(y_a6_full == 0)[0]
    for seed in CALIB_SEEDS:
        rng     = np.random.RandomState(seed)
        ss      = CALIB_SLICE_SIZE
        n_pos_s = max(1, round(ss * y_a6_full.mean()))
        n_neg_s = ss - n_pos_s
        n_pos_s = min(n_pos_s, len(pos_idx))
        n_neg_s = min(n_neg_s, len(neg_idx))
        sl_idx  = np.concatenate([
            rng.choice(pos_idx, n_pos_s, replace=False),
            rng.choice(neg_idx, n_neg_s, replace=False),
        ])
        CALIB_SLICE_IDX[seed] = sl_idx
    print(f'\\nBuilt {len(CALIB_SLICE_IDX)} calibration slices (size {CALIB_SLICE_SIZE}).')

print(f'\\nSection 2 done in {time.time()-t0:.1f}s')
""", "code-20"))

# -----------------------------------------------------------------------------
# §3  TIER B1 — content-feature ablation
# -----------------------------------------------------------------------------
C.append(_m("## §3 — Tier B1: Content-Feature Ablation", "md-03"))

C.append(_c("""
t0 = time.time()

# ── build DROP_FEATS list ────────────────────────────────────────────────
DROP_FEATS_BRIEF = ['self_ref_rate', 'avg_word_length', 'complex_word_rate',
                     'noun_rate', 'adj_rate']
DROP_FEATS_HIGH_ETA = sorted({f for f, e in ETA2.items() if e > 0.10})
DROP_FEATS = sorted(set(DROP_FEATS_BRIEF) | set(DROP_FEATS_HIGH_ETA))

print('B1 drop list:')
print(f'  brief defaults:    {DROP_FEATS_BRIEF}')
print(f'  η² > 0.10 (Tier A): {DROP_FEATS_HIGH_ETA}')
print(f'  FINAL DROP_FEATS:  {DROP_FEATS}')

avail_txt = [c for c in TXT_COLS if c in TRAIN_POOL.columns]
TXT_KEEP  = [c for c in avail_txt if c not in DROP_FEATS]
print(f'\\nKept text features: {len(TXT_KEEP)} / {len(avail_txt)}')

# ── re-compute text_top20 ranking on the kept set ────────────────────────
_imp = make_xgb(n_feats=len(TXT_KEEP))
_X = TRAIN_POOL[TXT_KEEP].fillna(0).values
_y = TRAIN_POOL['label_int'].values
_imp.fit(_X, _y)
_top = np.argsort(_imp.feature_importances_)[::-1][:20]
TXT_TOP20_DROP = [TXT_KEEP[i] for i in _top]
TXT_STYLO_DROP = [c for c in FEAT_STYLO if c in TXT_KEEP]
TXT_ALL_DROP   = TXT_KEEP

print(f'text_top20_drop: {TXT_TOP20_DROP[:5]} ...')
print(f'text_stylo_drop: {len(TXT_STYLO_DROP)} feats')
print(f'text_all_drop:   {len(TXT_ALL_DROP)} feats')

# ── train B1 text bases + whisper_wp + wavlm_wp (latter same as Tier A) ──
B1_DEF = {
    'text_top20_drop': TXT_TOP20_DROP,
    'text_stylo_drop': TXT_STYLO_DROP,
    'text_all_drop':   TXT_ALL_DROP,
    'whisper_wp':      WSP_COLS,
    'wavlm_wp':        WLM_COLS,
}

B1_OOF, B1_A6 = {}, {}
print('\\nTraining B1 bases...')
for name, fcols in B1_DEF.items():
    if not fcols:
        print(f'  SKIP {name}: no cols'); continue
    t1  = time.time()
    mdl = make_xgb(n_feats=len(fcols))
    if len(df_a4) > 0:
        oof, y_a4 = a4_oof(mdl, fcols, df_a4, df_a2)
        B1_OOF[name] = (oof, y_a4)
        try:    auc = roc_auc_score(y_a4, oof)
        except: auc = float('nan')
    else:
        auc = float('nan')
    m, uc = fit_model(mdl, fcols, TRAIN_POOL)
    if HAS_A6:
        B1_A6[name] = predict_proba(m, uc, df_a6)
    print(f'  {name}: {len(uc)} feats  OOF-AUC={auc:.3f}  {time.time()-t1:.1f}s')

# save OOF + a6 score matrices
oof_df = df_a4[['filename']].copy() if len(df_a4) > 0 else pd.DataFrame()
for n, (o, _) in B1_OOF.items():
    oof_df[n] = o
oof_df.to_csv(SAVE_DIR / 'b1_oof.csv', index=False)

if HAS_A6:
    a6_df = df_a6[['filename','label_int']].copy()
    for n, sc in B1_A6.items():
        a6_df[n] = sc
    a6_df.to_csv(SAVE_DIR / 'b1_a6_scores.csv', index=False)

print(f'\\n§3a (B1 bases) done in {time.time()-t0:.1f}s')
""", "code-30"))

C.append(_c("""
# ── §3b: B1 fusion search (2-way + 3-way over the 5 B1 bases) ────────────
t1 = time.time()

def grid_2(step=0.05):
    for w in np.arange(step, 1.0, step):
        yield [round(float(w), 4), round(float(1 - w), 4)]


def grid_3(step=0.1):
    for w1 in np.arange(0, 1.0 + step / 2, step):
        for w2 in np.arange(0, 1.0 - w1 + step / 2, step):
            w3 = round(1.0 - w1 - w2, 6)
            if w3 < -1e-9 or (w1 + w2 + w3) > 1.0 + 1e-9:
                continue
            if w1 > 0 and w2 > 0 and w3 > 0:
                yield [round(float(w1), 2), round(float(w2), 2), round(float(w3), 2)]


def search_fusion(bases, oof_dict, a6_dict, y_a4_ref, label):
    rows = []
    for k, gen in [(2, grid_2), (3, grid_3)]:
        for combo in itertools.combinations(bases, k):
            if not all(m in oof_dict for m in combo): continue
            for ws in gen():
                od = {m: oof_dict[m] for m in combo}
                fo = fused(list(combo), ws, od)
                thr, f1 = sweep_thr(y_a4_ref, fo)
                if HAS_A6 and all(m in a6_dict for m in combo):
                    a6d  = {m: a6_dict[m] for m in combo}
                    a6s  = fused(list(combo), ws, a6d)
                    a6f1 = metrics_at(y_a6_full, a6s, thr)['F1']
                    a6f1_ora = sweep_thr(y_a6_full, a6s)[1]
                else:
                    a6f1 = a6f1_ora = float('nan')
                rows.append({'label':label,'k':k,'members':'+'.join(combo),
                              'weights':json.dumps(ws),'cv_thr':thr,'cv_f1':f1,
                              'a6_f1_cv_thr':a6f1,'a6_f1_oracle':a6f1_ora})
    return pd.DataFrame(rows)


y_a4_ref = next(iter(B1_OOF.values()))[1]
b1_search = search_fusion(list(B1_DEF.keys()),
                           {k: v[0] for k, v in B1_OOF.items()},
                           B1_A6, y_a4_ref, 'B1')
b1_search = b1_search.sort_values('cv_f1', ascending=False)
b1_search.to_csv(SAVE_DIR / 'b1_search.csv', index=False)
print(f'B1 search: {len(b1_search)} combinations')
print('\\nTop 5 B1 by CV F1:')
print(b1_search[['k','members','weights','cv_thr','cv_f1','a6_f1_cv_thr','a6_f1_oracle']].head(5).to_string(index=False))
print('\\nTop 5 B1 by a6 F1 @ CV-thr:')
print(b1_search.sort_values('a6_f1_cv_thr', ascending=False)[
    ['k','members','weights','cv_thr','cv_f1','a6_f1_cv_thr']].head(5).to_string(index=False))

# ── B1 best (by CV F1) → Top-K% on a6 ────────────────────────────────────
B1_BEST_ROW   = b1_search.iloc[0]
B1_BEST_MEM   = B1_BEST_ROW['members'].split('+')
B1_BEST_W     = json.loads(B1_BEST_ROW['weights'])
B1_BEST_THR   = float(B1_BEST_ROW['cv_thr'])

if HAS_A6 and all(m in B1_A6 for m in B1_BEST_MEM):
    B1_BEST_A6 = fused(B1_BEST_MEM, B1_BEST_W,
                        {m: B1_A6[m] for m in B1_BEST_MEM})
    f1s, ps, rs, f1s_ind, f1s_php = [], [], [], [], []
    for seed in CALIB_SEEDS:
        sl = CALIB_SLICE_IDX[seed]
        ri, pred, _ = topk_predict(B1_BEST_A6, sl, y_a6_full)
        if ri is None: continue
        y_r = y_a6_full[ri]
        f1s.append(f1_score(y_r, pred, zero_division=0))
        ps.append(precision_score(y_r, pred, zero_division=0))
        rs.append(recall_score(y_r, pred, zero_division=0))
        if 'region' in df_a6.columns:
            reg_s = df_a6['region'].values[ri]
            for reg, lst in [('IND', f1s_ind), ('PHP', f1s_php)]:
                mk = reg_s == reg
                if mk.sum() >= 5:
                    lst.append(f1_score(y_r[mk], pred[mk], zero_division=0))
    print(f'\\nB1 best ({"+".join(B1_BEST_MEM)} w={B1_BEST_W}):')
    print(f'  Top-K%@50: F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}  '
          f'P={np.mean(ps):.3f}  R={np.mean(rs):.3f}')
    if f1s_ind: print(f'    IND F1: {np.mean(f1s_ind):.3f}')
    if f1s_php: print(f'    PHP F1: {np.mean(f1s_php):.3f}')
    B1_TOPK_F1 = float(np.mean(f1s))
else:
    B1_BEST_A6 = None; B1_TOPK_F1 = float('nan')

# pick winning B1 text base for B2 inputs
B1_TXT_BASES = ['text_top20_drop','text_stylo_drop','text_all_drop']
B1_TXT_STANDALONE = {n: roc_auc_score(B1_OOF[n][1], B1_OOF[n][0])
                      for n in B1_TXT_BASES if n in B1_OOF}
B1_TXT_BEST = max(B1_TXT_STANDALONE, key=B1_TXT_STANDALONE.get) if B1_TXT_STANDALONE else None
print(f'\\nB1 standalone text OOF-AUC: {B1_TXT_STANDALONE}')
print(f'→ B1 winning text base for §4: {B1_TXT_BEST}')

print(f'\\n§3b done in {time.time()-t1:.1f}s')
""", "code-31"))

# -----------------------------------------------------------------------------
# §4  TIER B2 — prosodic-only base
# -----------------------------------------------------------------------------
C.append(_m("## §4 — Tier B2: Prosodic-Only Base", "md-04"))

C.append(_c("""
t0 = time.time()

# ── auto-drop any prosodic candidate with η² > 0.10 ──────────────────────
PROS_DROPPED = sorted({f for f in FEAT_PROSODIC_BASE_CANDIDATES
                       if ETA2.get(f, 0.0) > 0.10})
PROS_KEEP    = [f for f in FEAT_PROSODIC_BASE_CANDIDATES
                if f in TRAIN_POOL.columns and f not in PROS_DROPPED]

print(f'Prosodic auto-dropped (η² > 0.10): {PROS_DROPPED}')
print(f'Prosodic kept ({len(PROS_KEEP)} feats): {PROS_KEEP}')

# Per-feature η² for the kept set (visibility for the user)
print('\\nKept prosodic features by η²:')
for f in PROS_KEEP:
    print(f'  {f:<35s}  η²={ETA2.get(f, float("nan")):.4f}')

# ── train prosodic_xgb on TRAIN_POOL ─────────────────────────────────────
prosodic_mdl = make_xgb(n_feats=len(PROS_KEEP))
if len(df_a4) > 0:
    pros_oof, y_a4 = a4_oof(prosodic_mdl, PROS_KEEP, df_a4, df_a2)
    try:    pros_auc = roc_auc_score(y_a4, pros_oof)
    except: pros_auc = float('nan')
else:
    pros_oof, y_a4 = np.array([]), np.array([]); pros_auc = float('nan')
m_pros, uc_pros = fit_model(prosodic_mdl, PROS_KEEP, TRAIN_POOL)
pros_a6 = predict_proba(m_pros, uc_pros, df_a6) if HAS_A6 else None
print(f'\\nprosodic_xgb: {len(uc_pros)} feats  OOF-AUC={pros_auc:.3f}')

# standalone prosodic F1 on a6
if HAS_A6:
    thr_pros, f1_pros_cv = sweep_thr(y_a4, pros_oof)
    metr = metrics_at(y_a6_full, pros_a6, thr_pros)
    print(f'  standalone a6 @ CV-thr={thr_pros:.2f}: '
          f'F1={metr["F1"]:.3f}  P={metr["P"]:.3f}  R={metr["R"]:.3f}  AUC={metr["AUC"]:.3f}')

# ── feature importances (top 15) ─────────────────────────────────────────
imp = pd.DataFrame({'feat': uc_pros, 'imp': m_pros.feature_importances_})
imp = imp.sort_values('imp', ascending=False).head(15)
print('\\nProsodic-base top-15 importances:')
print(imp.to_string(index=False))

# add prosodic to OOF/a6 dicts (alongside §3 winning text + whisper + wavlm)
B2_OOF = {'prosodic_xgb': (pros_oof, y_a4)}
B2_A6  = {'prosodic_xgb': pros_a6} if HAS_A6 else {}

# save
oof_out = df_a4[['filename']].copy() if len(df_a4) > 0 else pd.DataFrame()
oof_out['prosodic_xgb'] = pros_oof
oof_out.to_csv(SAVE_DIR / 'b2_prosodic_oof.csv', index=False)
if HAS_A6:
    sc_out = df_a6[['filename','label_int']].copy()
    sc_out['prosodic_xgb'] = pros_a6
    sc_out.to_csv(SAVE_DIR / 'b2_prosodic_a6_scores.csv', index=False)

print(f'\\n§4a done in {time.time()-t0:.1f}s')
""", "code-40"))

C.append(_c("""
# ── §4b: B2 fusion search ────────────────────────────────────────────────
# Pool = winning B1 text base + whisper_wp + wavlm_wp + prosodic_xgb
t1 = time.time()

if B1_TXT_BEST is None:
    print('No B1 text base; skipping B2 search.')
    b2_search = pd.DataFrame()
    B2_BEST_A6, B2_TOPK_F1 = None, float('nan')
else:
    B2_BASES = [B1_TXT_BEST, 'whisper_wp', 'wavlm_wp', 'prosodic_xgb']
    B2_OOF_ALL = {}
    B2_A6_ALL  = {}
    for n in B2_BASES:
        if n in B1_OOF: B2_OOF_ALL[n] = B1_OOF[n][0]
        elif n in B2_OOF: B2_OOF_ALL[n] = B2_OOF[n][0]
        if n in B1_A6: B2_A6_ALL[n] = B1_A6[n]
        elif n in B2_A6: B2_A6_ALL[n] = B2_A6[n]

    y_a4_ref = next(iter(B1_OOF.values()))[1]
    b2_search = search_fusion(B2_BASES, B2_OOF_ALL, B2_A6_ALL, y_a4_ref, 'B2')
    b2_search = b2_search.sort_values('cv_f1', ascending=False)
    b2_search.to_csv(SAVE_DIR / 'b2_search.csv', index=False)
    print(f'B2 search: {len(b2_search)} combinations')
    print('\\nTop 5 B2 by CV F1:')
    print(b2_search[['k','members','weights','cv_thr','cv_f1','a6_f1_cv_thr','a6_f1_oracle']].head(5).to_string(index=False))

    # B2 best → Top-K%
    B2_BEST_ROW = b2_search.iloc[0]
    B2_BEST_MEM = B2_BEST_ROW['members'].split('+')
    B2_BEST_W   = json.loads(B2_BEST_ROW['weights'])
    if HAS_A6 and all(m in B2_A6_ALL for m in B2_BEST_MEM):
        B2_BEST_A6 = fused(B2_BEST_MEM, B2_BEST_W,
                            {m: B2_A6_ALL[m] for m in B2_BEST_MEM})
        f1s = []
        for seed in CALIB_SEEDS:
            sl = CALIB_SLICE_IDX[seed]
            ri, pred, _ = topk_predict(B2_BEST_A6, sl, y_a6_full)
            if ri is None: continue
            f1s.append(f1_score(y_a6_full[ri], pred, zero_division=0))
        B2_TOPK_F1 = float(np.mean(f1s)) if f1s else float('nan')
        print(f'\\nB2 best ({"+".join(B2_BEST_MEM)} w={B2_BEST_W}):')
        print(f'  Top-K%@50: F1={B2_TOPK_F1:.3f}±{np.std(f1s):.3f}')
    else:
        B2_BEST_A6 = None; B2_TOPK_F1 = float('nan')

# ── B1+B2 unified fresh search (both interventions on) ───────────────────
print('\\n── B1+B2 unified search ──')
B12_BASES = ['text_top20_drop','text_stylo_drop','text_all_drop',
             'whisper_wp','wavlm_wp','prosodic_xgb']
B12_OOF = {}
B12_A6  = {}
for n in B12_BASES:
    if n in B1_OOF: B12_OOF[n] = B1_OOF[n][0]
    elif n in B2_OOF: B12_OOF[n] = B2_OOF[n][0]
    if n in B1_A6: B12_A6[n] = B1_A6[n]
    elif n in B2_A6: B12_A6[n] = B2_A6[n]

y_a4_ref = next(iter(B1_OOF.values()))[1]
b12_search = search_fusion(B12_BASES, B12_OOF, B12_A6, y_a4_ref, 'B1+B2')
b12_search = b12_search.sort_values('cv_f1', ascending=False)
b12_search.to_csv(SAVE_DIR / 'b1b2_search.csv', index=False)
print(f'B1+B2 search: {len(b12_search)} combinations')
print('\\nTop 5 B1+B2 by CV F1:')
print(b12_search[['k','members','weights','cv_thr','cv_f1','a6_f1_cv_thr','a6_f1_oracle']].head(5).to_string(index=False))

B12_BEST_ROW = b12_search.iloc[0]
B12_BEST_MEM = B12_BEST_ROW['members'].split('+')
B12_BEST_W   = json.loads(B12_BEST_ROW['weights'])
if HAS_A6 and all(m in B12_A6 for m in B12_BEST_MEM):
    B12_BEST_A6 = fused(B12_BEST_MEM, B12_BEST_W,
                         {m: B12_A6[m] for m in B12_BEST_MEM})
    f1s = []
    for seed in CALIB_SEEDS:
        sl = CALIB_SLICE_IDX[seed]
        ri, pred, _ = topk_predict(B12_BEST_A6, sl, y_a6_full)
        if ri is None: continue
        f1s.append(f1_score(y_a6_full[ri], pred, zero_division=0))
    B12_TOPK_F1 = float(np.mean(f1s)) if f1s else float('nan')
    print(f'\\nB1+B2 best ({"+".join(B12_BEST_MEM)} w={B12_BEST_W}):')
    print(f'  Top-K%@50: F1={B12_TOPK_F1:.3f}±{np.std(f1s):.3f}')
else:
    B12_BEST_A6 = None; B12_TOPK_F1 = float('nan')

print(f'\\n§4 done in {time.time()-t1:.1f}s')
""", "code-41"))

# -----------------------------------------------------------------------------
# §5  TIER B3 — region-conditional (diagnostic)
# -----------------------------------------------------------------------------
C.append(_m("## §5 — Tier B3: Region-Conditional Fusion (diagnostic only)", "md-05"))

C.append(_c("""
t0 = time.time()

if not HAS_A6 or 'region' not in df_a6.columns:
    print('region column unavailable; skipping §5')
    B3_RAN = False
else:
    B3_RAN = True
    # pick the best per-section a6 to use as the candidate set
    # use B1+B2 unified bases since they're a strict superset of B1, B2
    bases_for_b3 = [m for m in B12_BASES if m in B12_A6]
    print(f'B3 candidate bases ({len(bases_for_b3)}): {bases_for_b3}')

    region_vals = df_a6['region'].values
    rows = []

    def fit_global_then_region(bases, scores_a6, y_a6):
        # global = best B1+B2 weights on all a6 (already known)
        return None  # placeholder; we compute per-region directly below

    for reg in sorted(set(region_vals)):
        mk = region_vals == reg
        if mk.sum() < 30:
            print(f'  {reg}: n={mk.sum()} too small, skip')
            continue
        y_r       = y_a6_full[mk]
        sub_score = {b: B12_A6[b][mk] for b in bases_for_b3}

        # within-region 5-fold CV: pick best 2-way and 3-way per region
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        # generate per-fold OOF for each base by re-fitting on TRAIN_POOL only once,
        # then evaluating fold thresholds on a6 (model is fixed; CV here is over a6 sub).
        # That collapses to: search weights to maximise within-region OOF F1 via
        # k-fold threshold sweeps on a6 sub-region scores.
        best_2 = {'cv_f1': -1.0}
        best_3 = {'cv_f1': -1.0}
        for k, gen in [(2, grid_2), (3, grid_3)]:
            for combo in itertools.combinations(bases_for_b3, k):
                for ws in gen():
                    fo = fused(list(combo), ws, {m: sub_score[m] for m in combo})
                    f1s_fold = []
                    for tr, va in skf.split(np.zeros_like(y_r), y_r):
                        thr_fold, _ = sweep_thr(y_r[tr], fo[tr])
                        f1s_fold.append(f1_score(y_r[va], (fo[va] >= thr_fold).astype(int),
                                                  zero_division=0))
                    cv_f1 = float(np.mean(f1s_fold))
                    if k == 2 and cv_f1 > best_2['cv_f1']:
                        best_2 = {'k':2,'members':list(combo),'weights':ws,'cv_f1':cv_f1}
                    elif k == 3 and cv_f1 > best_3['cv_f1']:
                        best_3 = {'k':3,'members':list(combo),'weights':ws,'cv_f1':cv_f1}

        # global weights = B1+B2 unified best (applied within region)
        global_fo = B12_BEST_A6[mk] if B12_BEST_A6 is not None else None
        if global_fo is not None:
            thr_g, _ = sweep_thr(y_r, global_fo)
            f1_global = metrics_at(y_r, global_fo, thr_g)['F1']
        else:
            f1_global = float('nan')

        for tag, best in [('2way', best_2), ('3way', best_3)]:
            rows.append({
                'region': reg, 'n': int(mk.sum()), 'tag': tag,
                'global_f1': round(f1_global, 4),
                'region_specific_members': '+'.join(best['members']),
                'region_specific_weights': json.dumps(best['weights']),
                'region_specific_f1': round(best['cv_f1'], 4),
                'gap_vs_global': round(best['cv_f1'] - f1_global, 4),
            })
            print(f'  {reg} {tag}: global F1={f1_global:.3f}  '
                  f'region-specific F1={best["cv_f1"]:.3f}  '
                  f'gap={best["cv_f1"]-f1_global:+.3f}  '
                  f'members={"+".join(best["members"])}  w={best["weights"]}')

    pd.DataFrame(rows).to_csv(SAVE_DIR / 'b3_region_diagnostic.csv', index=False)
    print(f'\\nDIAGNOSTIC ONLY — region-specific weights NOT shipped (region unknown at inference).')

print(f'\\nSection 5 done in {time.time()-t0:.1f}s')
""", "code-50"))

# -----------------------------------------------------------------------------
# §6  GT AUDIT
# -----------------------------------------------------------------------------
C.append(_m("## §6 — GT Audit (per-a6-audio review file)", "md-06"))

C.append(_c("""
t0 = time.time()

# ── pick BEST Tier B condition by overall a6 F1 ──────────────────────────
# Candidates: B1 best, B2 best, B1+B2 unified best. (§5 is diagnostic, excluded.)
TB_CANDS = []
if B1_BEST_A6 is not None:
    TB_CANDS.append(('B1',     B1_BEST_MEM,  B1_BEST_W,   B1_BEST_A6,  B1_TOPK_F1))
if B2_BEST_A6 is not None:
    TB_CANDS.append(('B2',     B2_BEST_MEM,  B2_BEST_W,   B2_BEST_A6,  B2_TOPK_F1))
if B12_BEST_A6 is not None:
    TB_CANDS.append(('B1+B2', B12_BEST_MEM, B12_BEST_W, B12_BEST_A6, B12_TOPK_F1))

if not TB_CANDS:
    print('No Tier B candidates; skipping audit.')
    AUDIT = pd.DataFrame()
else:
    # rank by Top-K% F1 (the metric the brief uses for "best Tier B")
    TB_CANDS.sort(key=lambda x: (x[4] if not np.isnan(x[4]) else -1), reverse=True)
    BEST_TAG, BEST_MEM, BEST_W, BEST_A6, BEST_TOPK = TB_CANDS[0]
    print(f'Audit fusion: {BEST_TAG}  members={"+".join(BEST_MEM)}  '
          f'w={BEST_W}  Top-K%@50 F1={BEST_TOPK:.3f}')

    # ── per-base score columns for the audit ────────────────────────────
    BASE_A6_ALL = {**B1_A6, **B2_A6}
    base_score_cols = {}
    if 'text_top20_drop' in BEST_MEM or 'text_stylo_drop' in BEST_MEM or 'text_all_drop' in BEST_MEM:
        for b in ('text_top20_drop','text_stylo_drop','text_all_drop'):
            if b in BEST_MEM and b in BASE_A6_ALL:
                base_score_cols['p_text'] = BASE_A6_ALL[b]; break
    if 'whisper_wp' in BEST_MEM and 'whisper_wp' in BASE_A6_ALL:
        base_score_cols['p_whisper'] = BASE_A6_ALL['whisper_wp']
    if 'wavlm_wp' in BEST_MEM and 'wavlm_wp' in BASE_A6_ALL:
        base_score_cols['p_wavlm'] = BASE_A6_ALL['wavlm_wp']
    if 'prosodic_xgb' in BEST_MEM and 'prosodic_xgb' in BASE_A6_ALL:
        base_score_cols['p_prosodic'] = BASE_A6_ALL['prosodic_xgb']

    # ── single deterministic K from seed=42 50-audio slice ──────────────
    sl42 = CALIB_SLICE_IDX[AUDIT_SEED]
    K42  = y_a6_full[sl42].mean()
    THR_AUDIT = float(np.percentile(BEST_A6, 100.0 * (1.0 - K42)))
    pred_audit = (BEST_A6 >= THR_AUDIT).astype(int)
    print(f'  audit K={K42:.3f}  threshold={THR_AUDIT:.4f}')

    # ── build audit dataframe ───────────────────────────────────────────
    AUDIT = df_a6[['filename','candidate_id','question_id','label_int']].copy()
    if 'region' in df_a6.columns:
        AUDIT['region'] = df_a6['region'].values
    if 'speaking_time_s' in df_a6.columns:
        AUDIT['speaking_time_s'] = df_a6['speaking_time_s'].values
    AUDIT = AUDIT.rename(columns={'label_int':'gt_label'})
    AUDIT['fusion_score']  = BEST_A6
    AUDIT['pred_at_topk']  = pred_audit
    AUDIT['fusion_rank']   = (-AUDIT['fusion_score']).rank(method='min').astype(int)
    err = []
    for gt, pr in zip(AUDIT['gt_label'].values, pred_audit):
        if   gt == 1 and pr == 1: err.append('TP')
        elif gt == 0 and pr == 0: err.append('TN')
        elif gt == 0 and pr == 1: err.append('FP')
        else:                      err.append('FN')
    AUDIT['error_type'] = err

    for col, vals in base_score_cols.items():
        AUDIT[col] = vals

    base_mat = np.column_stack(list(base_score_cols.values())) if base_score_cols else None
    if base_mat is not None and base_mat.shape[1] > 0:
        AUDIT['bases_above_05_count'] = (base_mat > 0.5).sum(axis=1)
        AUDIT['bases_below_05_count'] = (base_mat < 0.5).sum(axis=1)
        AUDIT['unanimous_cheat']      = (base_mat > 0.6).all(axis=1)
        AUDIT['unanimous_honest']     = (base_mat < 0.3).all(axis=1)
        AUDIT['base_score_std']       = base_mat.std(axis=1)
    else:
        AUDIT['bases_above_05_count'] = 0
        AUDIT['bases_below_05_count'] = 0
        AUDIT['unanimous_cheat']      = False
        AUDIT['unanimous_honest']     = False
        AUDIT['base_score_std']       = float('nan')

    # ── gt_suspect_score ────────────────────────────────────────────────
    suspect = np.full(len(AUDIT), np.nan)
    for i in range(len(AUDIT)):
        gt = AUDIT['gt_label'].iat[i]
        fs = AUDIT['fusion_score'].iat[i]
        if gt == 0 and AUDIT['unanimous_cheat'].iat[i]:
            suspect[i] = +fs
        elif gt == 1 and AUDIT['unanimous_honest'].iat[i]:
            suspect[i] = -fs
    AUDIT['gt_suspect_score'] = suspect

    # ── sort: |gt_suspect_score| desc (NaN last), then fusion_rank asc ──
    sort_key = AUDIT['gt_suspect_score'].abs()
    AUDIT = AUDIT.assign(_sk=sort_key).sort_values(
        ['_sk','fusion_rank'], ascending=[False, True], na_position='last').drop(columns=['_sk'])
    AUDIT.to_csv(SAVE_DIR / 'gt_audit_a6.csv', index=False)

    # ── top 30 GT-suspects printed ──────────────────────────────────────
    top_susp = AUDIT[AUDIT['gt_suspect_score'].notna()].head(30)
    show_cols = ['filename','candidate_id','question_id','gt_label','error_type',
                 'fusion_score'] + list(base_score_cols.keys()) + \
                ['gt_suspect_score','unanimous_cheat','unanimous_honest']
    show_cols = [c for c in show_cols if c in top_susp.columns]
    print(f'\\nTop {len(top_susp)} GT-suspects (relabel-priority):')
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print(top_susp[show_cols].to_string(index=False))
    pd.reset_option('display.float_format')

    # ── slim files: top FP, top FN ──────────────────────────────────────
    top_fp = AUDIT[AUDIT['error_type'] == 'FP'].sort_values(
        'fusion_score', ascending=False).head(30)
    top_fn = AUDIT[AUDIT['error_type'] == 'FN'].sort_values(
        'fusion_score', ascending=True).head(30)
    top_fp.to_csv(SAVE_DIR / 'gt_audit_top_FP.csv', index=False)
    top_fn.to_csv(SAVE_DIR / 'gt_audit_top_FN.csv', index=False)

    n_fp_unanim = int(top_fp['unanimous_cheat'].sum()) if 'unanimous_cheat' in top_fp else 0
    n_fn_unanim = int(top_fn['unanimous_honest'].sum()) if 'unanimous_honest' in top_fn else 0
    print(f'\\nOf {len(top_fp)} top FP candidates, {n_fp_unanim} have unanimous CHEAT '
          f'consensus across all bases.')
    print(f'Of {len(top_fn)} top FN candidates, {n_fn_unanim} have unanimous HONEST '
          f'consensus.')
    print('These are highest-priority for label re-verification.')

print(f'\\nSection 6 done in {time.time()-t0:.1f}s')
""", "code-60"))

# -----------------------------------------------------------------------------
# §7  SUMMARY TABLE
# -----------------------------------------------------------------------------
C.append(_m("## §7 — Summary Table", "md-07"))

C.append(_c("""
t0 = time.time()
summary = []

def _push(condition, pick, region, F1, P=float('nan'), R=float('nan'),
           AUC=float('nan'), n_pos=float('nan'), n_eval=float('nan'),
           thr=float('nan'), notes=''):
    delta = round(F1 - TIER_A_BEST_F1, 4) if not np.isnan(TIER_A_BEST_F1) else float('nan')
    summary.append({
        'condition': condition, 'pick_name': pick, 'region': region,
        'F1': round(F1, 4), 'P': round(P, 4) if not np.isnan(P) else P,
        'R': round(R, 4) if not np.isnan(R) else R,
        'AUC': round(AUC, 4) if not np.isnan(AUC) else AUC,
        'n_pos': n_pos, 'n_eval': n_eval, 'threshold': thr,
        'notes': notes, 'delta_vs_tierA_best': delta,
    })

# baseline_combined_topk = Tier A best (copied)
if not np.isnan(TIER_A_BEST_F1):
    _push('baseline_combined_topk', TIER_A_BEST_NAME, 'overall',
          TIER_A_BEST_F1, notes='Tier A floor (copied from tier_a_summary.csv)')

# B1 Top-K% (best B1 fusion via mean over slices)
def _topk_metrics(scores, y, slices, region_mask=None):
    f1s, ps, rs = [], [], []
    n_eval_acc = 0
    for seed in slices:
        sl = CALIB_SLICE_IDX[seed]
        ri, pred, _ = topk_predict(scores, sl, y)
        if ri is None: continue
        y_r = y[ri]
        if region_mask is not None:
            mk = region_mask[ri]
            if mk.sum() < 5: continue
            y_r = y_r[mk]; pred = pred[mk]
        n_eval_acc = max(n_eval_acc, len(y_r))
        f1s.append(f1_score(y_r, pred, zero_division=0))
        ps.append(precision_score(y_r, pred, zero_division=0))
        rs.append(recall_score(y_r, pred, zero_division=0))
    if not f1s: return None
    return (np.mean(f1s), np.mean(ps), np.mean(rs), n_eval_acc)


def _push_topk(label, members_str, scores):
    if scores is None: return
    overall = _topk_metrics(scores, y_a6_full, CALIB_SEEDS)
    if overall:
        _push(label, members_str, 'overall', overall[0], P=overall[1], R=overall[2],
              n_eval=overall[3], notes=f'mean over {N_REPEATS} draws')
    if 'region' in df_a6.columns:
        for reg in sorted(df_a6['region'].dropna().unique()):
            rm = (df_a6['region'].values == reg)
            r_metrics = _topk_metrics(scores, y_a6_full, CALIB_SEEDS, region_mask=rm)
            if r_metrics:
                _push(label, members_str, reg, r_metrics[0], P=r_metrics[1],
                      R=r_metrics[2], n_eval=r_metrics[3],
                      notes=f'mean over {N_REPEATS} draws')


_push_topk('B1_topk',     '+'.join(B1_BEST_MEM)  if B1_BEST_A6  is not None else '', B1_BEST_A6)
_push_topk('B2_topk',     '+'.join(B2_BEST_MEM)  if B2_BEST_A6  is not None else '', B2_BEST_A6)
_push_topk('B1+B2_topk', '+'.join(B12_BEST_MEM) if B12_BEST_A6 is not None else '', B12_BEST_A6)

# B3 region row (if ran) — best region-specific F1 across regions
if 'B3_RAN' in dir() and B3_RAN:
    b3_path = SAVE_DIR / 'b3_region_diagnostic.csv'
    if b3_path.exists():
        b3df = pd.read_csv(b3_path)
        for _, r in b3df.iterrows():
            _push('B3_region', r['region_specific_members'], str(r['region']),
                  r['region_specific_f1'], notes=f'region-specific weights (DIAGNOSTIC, not shipped); gap_vs_global={r["gap_vs_global"]:+.3f}')

S = pd.DataFrame(summary)

# best_overall row (whichever Tier B condition wins overall)
overall_b = S[(S['region'] == 'overall') &
               (S['condition'].isin(['B1_topk','B2_topk','B1+B2_topk']))]
if len(overall_b) > 0:
    bw = overall_b.sort_values('F1', ascending=False).iloc[0]
    _push('best_overall', bw['pick_name'], 'overall', bw['F1'],
          P=bw['P'], R=bw['R'], n_eval=bw['n_eval'],
          notes=f'winner = {bw["condition"]}')
    S = pd.DataFrame(summary)

S = S.sort_values(['region','F1'], ascending=[True, False])
S.to_csv(SAVE_DIR / 'tier_b_summary.csv', index=False)

# ── print master table ───────────────────────────────────────────────────
print('=' * 80)
print('TIER B SUMMARY (sorted by overall F1)')
print('=' * 80)
ov = S[S['region'] == 'overall'].sort_values('F1', ascending=False)
print(ov[['condition','pick_name','F1','P','R','AUC','threshold','delta_vs_tierA_best','notes']].to_string(index=False))

# ── flag conditions beating Tier A by >= 0.02 with bootstrap CI excluding 0 ─
print('\\n' + '=' * 80)
print('SIGNIFICANCE CHECK (bootstrap 95% CI, n=1000)')
print('=' * 80)

if TIER_A_BEST_A6 is not None and not np.isnan(TIER_A_BEST_F1):
    # threshold for Tier A floor: oracle on a6 (matches how its F1 was reported)
    ta_thr, ta_f1 = sweep_thr(y_a6_full, TIER_A_BEST_A6)
    print(f'Tier A floor recomputed: F1={ta_f1:.4f}  thr={ta_thr:.3f}')

    cands = [
        ('B1_topk',    B1_BEST_A6,  B1_TOPK_F1),
        ('B2_topk',    B2_BEST_A6,  B2_TOPK_F1),
        ('B1+B2_topk', B12_BEST_A6, B12_TOPK_F1),
    ]
    flagged = False
    for label, sc, f1 in cands:
        if sc is None or np.isnan(f1): continue
        delta = f1 - ta_f1
        if abs(delta) < 0.02:
            print(f'  {label}: F1={f1:.4f}  Δ={delta:+.4f}  (under threshold; no CI)')
            continue
        # use seed=42 audit threshold for the Tier B condition
        sl42      = CALIB_SLICE_IDX[AUDIT_SEED]
        K42       = y_a6_full[sl42].mean()
        thr_b     = float(np.percentile(sc, 100.0 * (1.0 - K42)))
        lo, hi, mu = bootstrap_ci(y_a6_full, sc, thr_b, TIER_A_BEST_A6, ta_thr)
        excl0     = (lo > 0) or (hi < 0)
        flag      = ' ***' if excl0 else ''
        print(f'  {label}: F1={f1:.4f}  Δ={delta:+.4f}  '
              f'CI=[{lo:+.4f}, {hi:+.4f}]  μ={mu:+.4f}{flag}')
        if excl0 and delta > 0:
            flagged = True
            print(f'    → {label} BEATS Tier A best by {delta:+.4f} (CI excludes 0)')

    if not flagged:
        print('\\nNo Tier B condition beats Tier A best with bootstrap CI excluding 0.')
else:
    print('Tier A floor a6 scores not available; skipping CI check.')

# ── conclusion line ──────────────────────────────────────────────────────
print('\\n' + '=' * 80)
ov = pd.DataFrame(summary)
ov_topb = ov[(ov['region']=='overall') &
              (ov['condition'].isin(['B1_topk','B2_topk','B1+B2_topk']))]
if len(ov_topb) > 0:
    best = ov_topb.sort_values('F1', ascending=False).iloc[0]
    if not np.isnan(TIER_A_BEST_F1):
        print(f'CONCLUSION: best Tier B = {best["condition"]} ({best["pick_name"]})  '
              f'F1={best["F1"]:.4f}  vs Tier A floor F1={TIER_A_BEST_F1:.4f}  '
              f'Δ={best["F1"]-TIER_A_BEST_F1:+.4f}')
    else:
        print(f'CONCLUSION: best Tier B = {best["condition"]} F1={best["F1"]:.4f}  '
              f'(Tier A floor F1 unavailable)')
print('=' * 80)
print(f'\\nSection 7 done in {time.time()-t0:.1f}s')
print(f'All artifacts saved to {SAVE_DIR}')
""", "code-70"))

# -----------------------------------------------------------------------------
# Build notebook JSON
# -----------------------------------------------------------------------------
NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8.0"}
    },
    "cells": C,
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)

print(f'Written: {OUT}')
print(f'Cells:   {len(C)}')
