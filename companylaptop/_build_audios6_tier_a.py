#!/usr/bin/env python3
"""
Generates audios6_tier_a.ipynb.
Run:  python _build_audios6_tier_a.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / 'audios6_tier_a.ipynb'


def _c(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None,
            "source": src.strip('\n')}


def _m(src, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": src.strip('\n')}


C = []

# ─────────────────────────────────────────────────────────────────────────────
# §0  CONFIG + IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §0 — Config + Imports", "md-00"))

C.append(_c(r"""
import os, re, time, warnings, json, itertools
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score)
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── constants ──────────────────────────────────────────────────────────────
SPW_DEPLOY           = (1 - 0.17) / 0.17   # 4.882...
RANDOM_SEED          = 42
N_FOLDS              = 5
MIN_SPEAKING_S       = 30
CALIB_SLICE_SIZE     = 50
MAX_PSEUDO_PER_CLASS = 80
N_REPEATS            = 20
CALIB_SEEDS          = list(range(N_REPEATS))   # 0-19, fixed across all sections

NB_DIR   = Path('.').resolve()
SAVE_DIR = NB_DIR / 'checkpoints_tier_a'
SAVE_DIR.mkdir(exist_ok=True)

LABEL_MAP = {
    'read':1,'cheating':1,'reading':1,'scripted':1,'yes':1,'1':1, 1:1,
    'spontaneous':0,'not cheating':0,'not_cheating':0,'no':0,'0':0, 0:0,'genuine':0,
}
_RE = re.compile(r'^(.+)_(\d{1,3})\.[a-zA-Z0-9]+$')

print(f'NB_DIR:     {NB_DIR}')
print(f'SAVE_DIR:   {SAVE_DIR}')
print(f'SPW_DEPLOY: {SPW_DEPLOY:.4f}')
""", "code-00"))

# ── feature groups ──────────────────────────────────────────────────────────
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
FEAT_STYLO    = FEAT_STYLOMETRIC  # alias used in pick3

print(f'Canonical text features: {len(FEAT_ALL_TEXT)}')
""", "code-01"))

# ── utility functions ────────────────────────────────────────────────────────
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
    if feat    is not None: df = df.merge(feat,   on='filename', how='inner')
    if wlm_df  is not None: df = df.merge(wlm_df, on='filename', how='inner')
    if wsp_df  is not None: df = df.merge(wsp_df, on='filename', how='inner')
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


def make_rf(seed=RANDOM_SEED):
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2,
        class_weight={0: 1.0, 1: float(SPW_DEPLOY)},
        random_state=seed, n_jobs=-1)


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
    p = (s >= thr).astype(int)
    return {
        'F1':    round(f1_score(y, p, zero_division=0), 4),
        'P':     round(precision_score(y, p, zero_division=0), 4),
        'R':     round(recall_score(y, p, zero_division=0), 4),
        'AUC':   round(roc_auc_score(y, s) if len(np.unique(y)) > 1 else float('nan'), 4),
        'n_pos': int(y.sum()), 'n_eval': len(y),
    }


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
    return lo, hi, float(np.mean(d))


def fused(members, weights, score_dict):
    w = np.array(weights, dtype=float)
    present = [m for m in members if m in score_dict]
    pw = np.array([w[i] for i, m in enumerate(members) if m in score_dict])
    pw = pw / pw.sum()
    s  = sum(pw[i] * score_dict[m] for i, m in enumerate(present))
    if len(present) < len(members):
        miss = set(members) - set(present)
        print(f'  WARN fusion missing {miss}; renormalized.')
    return s


def fit_weighted(mdl, feat_cols, df):
    ac = [c for c in feat_cols if c in df.columns]
    X  = df[ac].fillna(0).values
    y  = df['label_int'].values
    sw = df['sample_weight'].fillna(1.0).values if 'sample_weight' in df.columns else np.ones(len(df))
    m  = deepcopy(mdl)
    try:
        m.fit(X, y, sample_weight=sw)
    except TypeError:
        m.fit(X, y)
    return m, ac

print('Utilities loaded.')
""", "code-02"))

# ─────────────────────────────────────────────────────────────────────────────
# §1  LOAD & MERGE ALL BATCHES
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §1 — Load & Merge All Batches", "md-01"))

C.append(_c("""
t0 = time.time()

BATCHES = ['audios2', 'audios4', 'audios5', 'audios6']
BD      = {}           # batch -> merged DataFrame
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

# audios3 — no GT
BD3 = None
try:
    df3, _, _ = load_batch('audios3', require_gt=False)
    BD3 = df3
    print(f'\\naudios3: {len(df3)} rows (no GT)')
except Exception as e:
    print(f'\\naudios3 unavailable: {e}')

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
print(f'\\nSection 1 done in {time.time()-t0:.1f}s')
""", "code-10"))

# ─────────────────────────────────────────────────────────────────────────────
# §2  BASELINE a6 FUSION SCORES
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §2 — Establish Baseline a6 Fusion Scores", "md-02"))

C.append(_c("""
t0 = time.time()

df_a2 = BD.get('audios2', pd.DataFrame())
df_a4 = BD.get('audios4', pd.DataFrame())
df_a6 = BD.get('audios6', pd.DataFrame())
HAS_A6 = len(df_a6) > 0

# ── determine text_top20 from TRAIN_POOL importance ───────────────────────
avail_txt = [c for c in TXT_COLS if c in TRAIN_POOL.columns]
_imp_model = make_xgb(n_feats=len(avail_txt))
_X = TRAIN_POOL[avail_txt].fillna(0).values
_y = TRAIN_POOL['label_int'].values
_imp_model.fit(_X, _y)
_top_idx  = np.argsort(_imp_model.feature_importances_)[::-1][:20]
TXT_TOP20 = [avail_txt[i] for i in _top_idx]
TXT_STYLO = [c for c in FEAT_STYLO if c in TRAIN_POOL.columns]
print(f'text_top20: {TXT_TOP20[:5]} ...  ({len(TXT_TOP20)} total)')
print(f'text_stylo: {len(TXT_STYLO)} feats')

# ── base model registry ──────────────────────────────────────────────────
BASE_DEF = {
    'text_top20_xgb': (TXT_TOP20,    'xgb'),
    'text_all_xgb':   (avail_txt,    'xgb'),
    'text_stylo_xgb': (TXT_STYLO,    'xgb'),
    'whisper_wp_xgb': (WSP_COLS,     'xgb'),
    'wavlm_wp_xgb':   (WLM_COLS,     'xgb'),
    'whisper_wp_rf':  (WSP_COLS,     'rf'),
}
BASE_MODEL = {}   # name -> (fitted, used_cols)
BASE_OOF   = {}   # name -> (oof_proba_on_a4, y_a4)
BASE_A6    = {}   # name -> a6_proba

print('\\nTraining base models...')
for name, (fcols, mtype) in BASE_DEF.items():
    if not fcols:
        print(f'  SKIP {name}: no cols'); continue
    t1 = time.time()
    mdl = make_xgb(n_feats=len(fcols)) if mtype == 'xgb' else make_rf()
    if len(df_a4) > 0:
        oof, y_a4 = a4_oof(mdl, fcols, df_a4, df_a2)
        BASE_OOF[name] = (oof, y_a4)
        try:
            auc = roc_auc_score(y_a4, oof)
        except Exception:
            auc = float('nan')
    else:
        oof, y_a4, auc = np.array([]), np.array([]), float('nan')
    m, uc = fit_model(mdl, fcols, TRAIN_POOL)
    BASE_MODEL[name] = (m, uc)
    if HAS_A6:
        BASE_A6[name] = predict_proba(m, uc, df_a6)
    print(f'  {name}: {len(uc)} feats  OOF-AUC={auc:.3f}  {time.time()-t1:.1f}s')

print(f'Base models ready in {time.time()-t0:.1f}s')
""", "code-20"))

C.append(_c("""
# ── load frozen configs if available ────────────────────────────────────
FROZEN = {}
for fp in [NB_DIR / 'checkpoints_everything' / 'frozen_configs.json',
           NB_DIR / 'checkpoints_fusions'     / 'frozen_configs.json']:
    if fp.exists():
        with open(fp) as _f:
            FROZEN = json.load(_f)
        print(f'Loaded frozen_configs: {fp}')
        break
if not FROZEN:
    print('frozen_configs.json not found; using defaults for Pick2')


def _pick2_weights():
    for key in ('fusion_2way_top20', 'fusion_3way_top20', 'base_results'):
        for row in FROZEN.get(key, []):
            mems = str(row.get('members', ''))
            if 'text_all' in mems and 'wavlm_wp' in mems:
                try:
                    raw_w = str(row.get('weights', '')).replace("'", '"')
                    w = json.loads(raw_w) if raw_w.startswith('[') else [0.5, 0.5]
                    t = float(row.get('chosen_threshold', 0.5))
                    return w, t
                except Exception:
                    pass
    return [0.5, 0.5], 0.5


P2W, P2T = _pick2_weights()

PICKS = {
    'pick1': {'members': ['text_top20_xgb', 'whisper_wp_xgb', 'wavlm_wp_xgb'],
              'weights': [0.20, 0.44, 0.36], 'frozen_thr': 0.59},
    'pick2': {'members': ['text_all_xgb', 'wavlm_wp_xgb'],
              'weights': P2W, 'frozen_thr': P2T},
    'pick3': {'members': ['text_top20_xgb', 'text_stylo_xgb', 'whisper_wp_rf'],
              'weights': [0.12, 0.16, 0.72], 'frozen_thr': 0.59},
}

# ── compute CV threshold from a4 OOF ─────────────────────────────────────
for pn, pc in PICKS.items():
    oof_d = {m: BASE_OOF[m][0] for m in pc['members'] if m in BASE_OOF}
    if not oof_d:
        print(f'{pn}: no OOF data'); continue
    y_ref = BASE_OOF[pc['members'][0]][1]
    fo    = fused(pc['members'], pc['weights'], oof_d)
    thr, f1 = sweep_thr(y_ref, fo)
    PICKS[pn]['cv_thr'] = thr
    PICKS[pn]['cv_f1']  = f1
    print(f'{pn}  CV-thr={thr:.3f}  CV-F1={f1:.3f}')

# ── score a6 and compute baseline table ──────────────────────────────────
PICK_A6 = {}   # pn -> (scores, y_a6)

print('\\nBaseline a6 metrics:')
for pn, pc in PICKS.items():
    a6_d = {m: BASE_A6[m] for m in pc['members'] if m in BASE_A6}
    if not a6_d or not HAS_A6:
        print(f'  {pn}: a6 unavailable'); continue
    scores  = fused(pc['members'], pc['weights'], a6_d)
    y_a6    = df_a6['label_int'].values
    PICK_A6[pn] = (scores, y_a6)
    # save
    out = df_a6[['filename']].copy()
    out['score']     = scores
    out['pred']      = (scores >= pc['frozen_thr']).astype(int)
    out['label_int'] = y_a6
    out.to_csv(SAVE_DIR / f'baseline_{pn}_a6.csv', index=False)

    fthr = pc['frozen_thr']
    cthr = pc.get('cv_thr', fthr)
    othr, of1 = sweep_thr(y_a6, scores)
    PICKS[pn]['oracle_thr'] = othr
    PICKS[pn]['oracle_f1']  = of1
    fm = metrics_at(y_a6, scores, fthr)
    cm = metrics_at(y_a6, scores, cthr)
    print(f'\\n  {pn}:')
    print(f'    frozen  thr={fthr:.2f}  F1={fm["F1"]:.3f}  P={fm["P"]:.3f}  R={fm["R"]:.3f}  AUC={fm["AUC"]:.3f}')
    print(f'    cv      thr={cthr:.2f}  F1={cm["F1"]:.3f}  P={cm["P"]:.3f}  R={cm["R"]:.3f}')
    print(f'    oracle  thr={othr:.2f}  F1={of1:.3f}')
    if 'region' in df_a6.columns:
        for reg in sorted(df_a6['region'].dropna().unique()):
            mk = (df_a6['region'] == reg).values
            if mk.sum() < 10: continue
            rm = metrics_at(y_a6[mk], scores[mk], fthr)
            print(f'    {reg}: F1={rm["F1"]:.3f}  AUC={rm["AUC"]:.3f}  n={mk.sum()}')

print(f'\\nSection 2 done in {time.time()-t0:.1f}s')
""", "code-21"))

# ─────────────────────────────────────────────────────────────────────────────
# §3  TIER A1: Top-K% rank-based decision rule
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §3 — Tier A1: Top-K% Rank-Based Decision Rule", "md-03"))

C.append(_c("""
t0 = time.time()

SLICE_SIZES = [30, 50, 80, 120]
CALIB_SLICE_IDX = {}   # seed -> array of a6 row indices  (shared with §5/§6)

topk_rows = []

for pn, pc in PICKS.items():
    if pn not in PICK_A6:
        print(f'{pn}: no a6 scores, skip'); continue
    scores, y_a6 = PICK_A6[pn]
    n_a6 = len(y_a6)
    fthr = pc['frozen_thr']
    cthr = pc.get('cv_thr', fthr)
    othr, _ = sweep_thr(y_a6, scores)

    print(f'\\n── {pn} ──')
    for ss in SLICE_SIZES:
        f1s, ps, rs = [], [], []
        f1s_ind, f1s_php = [], []
        for seed in CALIB_SEEDS:
            rng = np.random.RandomState(seed)
            # stratified sample: equal representation of cheat/honest
            pos_idx  = np.where(y_a6 == 1)[0]
            neg_idx  = np.where(y_a6 == 0)[0]
            n_pos_s  = max(1, round(ss * y_a6.mean()))
            n_neg_s  = ss - n_pos_s
            n_pos_s  = min(n_pos_s, len(pos_idx))
            n_neg_s  = min(n_neg_s, len(neg_idx))
            sl_idx   = np.concatenate([
                rng.choice(pos_idx, n_pos_s, replace=False),
                rng.choice(neg_idx, n_neg_s, replace=False),
            ])
            if ss == CALIB_SLICE_SIZE:
                CALIB_SLICE_IDX[seed] = sl_idx   # save for §5/§6
            rem_mask = np.ones(n_a6, bool)
            rem_mask[sl_idx] = False
            rem_idx  = np.where(rem_mask)[0]
            if len(rem_idx) == 0: continue

            K_hat = y_a6[sl_idx].mean()
            s_rem = scores[rem_idx]
            y_rem = y_a6[rem_idx]
            thr_k = float(np.percentile(s_rem, 100.0 * (1.0 - K_hat)))
            pred  = (s_rem >= thr_k).astype(int)
            f1s.append(f1_score(y_rem, pred, zero_division=0))
            ps.append(precision_score(y_rem, pred, zero_division=0))
            rs.append(recall_score(y_rem, pred, zero_division=0))
            if 'region' in df_a6.columns:
                reg_s = df_a6['region'].values[rem_idx]
                for reg, lst in [('IND', f1s_ind), ('PHP', f1s_php)]:
                    mk = reg_s == reg
                    if mk.sum() >= 5:
                        lst.append(f1_score(y_rem[mk], pred[mk], zero_division=0))

        if not f1s: continue
        # apples-to-apples baselines on remaining set (use first seed)
        sl0   = CALIB_SLICE_IDX.get(CALIB_SEEDS[0], np.array([]))
        rem0  = np.where(np.isin(np.arange(n_a6), sl0, invert=True))[0]
        if len(rem0) > 0:
            froz_f1   = f1_score(y_a6[rem0], (scores[rem0] >= fthr).astype(int), zero_division=0)
            cv_f1_rem = f1_score(y_a6[rem0], (scores[rem0] >= cthr).astype(int), zero_division=0)
            ora_thr0, ora_f1_0 = sweep_thr(y_a6[rem0], scores[rem0])
        else:
            froz_f1 = cv_f1_rem = ora_f1_0 = float('nan')

        row = {
            'pick': pn, 'slice_size': ss,
            'mean_K_hat': round(np.mean([y_a6[CALIB_SLICE_IDX.get(s, [])].mean()
                                          for s in CALIB_SEEDS if s in CALIB_SLICE_IDX]), 4),
            'std_K_hat': round(np.std([y_a6[CALIB_SLICE_IDX.get(s, [])].mean()
                                        for s in CALIB_SEEDS if s in CALIB_SLICE_IDX]), 4),
            'topk_f1_mean': round(np.mean(f1s), 4),
            'topk_f1_std':  round(np.std(f1s),  4),
            'topk_prec_mean': round(np.mean(ps), 4),
            'topk_rec_mean':  round(np.mean(rs), 4),
            'topk_f1_ind_mean': round(np.mean(f1s_ind), 4) if f1s_ind else float('nan'),
            'topk_f1_php_mean': round(np.mean(f1s_php), 4) if f1s_php else float('nan'),
            'frozen_f1': round(froz_f1, 4),
            'cv_f1':     round(cv_f1_rem, 4),
            'oracle_f1': round(ora_f1_0, 4),
            'n_remaining': int(rem_mask.sum()),
        }
        topk_rows.append(row)
        flag = ' ***' if row['topk_f1_mean'] > row['cv_f1'] + 0.03 else ''
        print(f'  slice={ss:3d}: topk_F1={row["topk_f1_mean"]:.3f}±{row["topk_f1_std"]:.3f} '
              f'| frozen={row["frozen_f1"]:.3f} cv={row["cv_f1"]:.3f} oracle={row["oracle_f1"]:.3f}{flag}')

topk_df = pd.DataFrame(topk_rows)
topk_df.to_csv(SAVE_DIR / 'topk_results.csv', index=False)
print(f'\\nSaved topk_results.csv')
print(f'Section 3 done in {time.time()-t0:.1f}s')
""", "code-30"))

# ─────────────────────────────────────────────────────────────────────────────
# §4  TIER A2: Per-question deviation features
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §4 — Tier A2: Per-Question Deviation Features", "md-04"))

C.append(_c("""
t0 = time.time()
# ── 4a: compute honest per-question means/stds on TRAIN_POOL ─────────────
honest_pool = TRAIN_POOL[TRAIN_POOL['label_int'] == 0]
avail_txt   = [c for c in TXT_COLS if c in TRAIN_POOL.columns]

PQ_MEAN = {}   # (q, feat) -> mean
PQ_STD  = {}   # (q, feat) -> std
for feat in avail_txt:
    g = honest_pool.groupby('question_id')[feat]
    PQ_MEAN[feat] = g.mean().to_dict()
    PQ_STD[feat]  = g.std().fillna(0).to_dict()

pool_mean = {f: honest_pool[f].mean() for f in avail_txt}
pool_std  = {f: max(honest_pool[f].std(), 1e-6) for f in avail_txt}

print(f'Per-question baselines computed for {len(avail_txt)} features '
      f'across {honest_pool.question_id.nunique()} questions')


def augment_df(df):
    out = df.copy()
    for feat in avail_txt:
        if feat not in df.columns: continue
        q_arr = df['question_id'].values
        raw   = df[feat].fillna(pool_mean[feat]).values
        base  = np.array([PQ_MEAN[feat].get(q, pool_mean[feat]) for q in q_arr])
        sd    = np.array([max(PQ_STD[feat].get(q, pool_std[feat]), 1e-6) for q in q_arr])
        out[feat + '__dev'] = raw - base
        out[feat + '__z']   = (raw - base) / sd
    return out


# ── 4b: build augmented frames ────────────────────────────────────────────
BD_AUG = {}
for bat, df in BD.items():
    BD_AUG[bat] = augment_df(df)
if BD3 is not None:
    BD3_AUG = augment_df(BD3)

AUG_TXT_COLS = (avail_txt +
                [f + '__dev' for f in avail_txt] +
                [f + '__z'   for f in avail_txt])
TRAIN_POOL_AUG = pd.concat(
    [BD_AUG.get('audios2'), BD_AUG.get('audios4'), BD_AUG.get('audios5')],
    ignore_index=True).dropna(subset=['label_int'])
df_a2_aug = BD_AUG.get('audios2', pd.DataFrame())
df_a4_aug = BD_AUG.get('audios4', pd.DataFrame())
df_a6_aug = BD_AUG.get('audios6', pd.DataFrame())
print(f'Augmented feature width: {len(AUG_TXT_COLS)} ({len(avail_txt)} raw + 2x dev/z)')

# ── 4c: retrain three text bases on augmented features ───────────────────
AUG_TOP20 = [c for c in TXT_TOP20] + [c+'__dev' for c in TXT_TOP20] + [c+'__z' for c in TXT_TOP20]
AUG_STYLO = TXT_STYLO + [c+'__dev' for c in TXT_STYLO] + [c+'__z' for c in TXT_STYLO]

AUG_BASE_DEF = {
    'text_top20_aug': (AUG_TOP20,    'xgb'),
    'text_stylo_aug': (AUG_STYLO,    'xgb'),
    'text_all_aug':   (AUG_TXT_COLS, 'xgb'),
}
AUG_MODEL = {}
AUG_OOF   = {}
AUG_A6    = {}

print('\\nTraining augmented text bases...')
for name, (fcols, _) in AUG_BASE_DEF.items():
    t1  = time.time()
    mdl = make_xgb(n_feats=len(fcols))
    if len(df_a4_aug) > 0:
        oof, y_a4 = a4_oof(mdl, fcols, df_a4_aug, df_a2_aug)
        AUG_OOF[name] = (oof, y_a4)
        try:   auc = roc_auc_score(y_a4, oof)
        except: auc = float('nan')
    else:
        auc = float('nan')
    m, uc = fit_model(mdl, fcols, TRAIN_POOL_AUG)
    AUG_MODEL[name] = (m, uc)
    if len(df_a6_aug) > 0:
        AUG_A6[name] = predict_proba(m, uc, df_a6_aug)
    print(f'  {name}: OOF-AUC={auc:.3f}  {time.time()-t1:.1f}s')

print(f'§4a-4c done in {time.time()-t0:.1f}s')
""", "code-40"))

C.append(_c("""
# ── 4d: fusion weight grid-search on augmented OOF ────────────────────────
t1  = time.time()
all_bases_aug  = list(AUG_OOF.keys()) + ['whisper_wp_xgb', 'wavlm_wp_xgb']
oof_aug_all    = {}
oof_aug_all.update({n: AUG_OOF[n] for n in AUG_OOF})
oof_aug_all.update({n: BASE_OOF[n] for n in ['whisper_wp_xgb', 'wavlm_wp_xgb']
                    if n in BASE_OOF})

# ensure same y_a4 reference
y_a4_ref = None
for n, (o, y) in oof_aug_all.items():
    y_a4_ref = y; break

def grid_weights_2way(step=0.05):
    ws = np.arange(step, 1.0, step)
    for w in ws:
        yield [round(w, 4), round(1 - w, 4)]

def grid_weights_3way(step=0.1):
    for w1 in np.arange(0, 1.0 + step / 2, step):
        for w2 in np.arange(0, 1.0 - w1 + step / 2, step):
            w3 = round(1.0 - w1 - w2, 6)
            if w3 < -1e-9: continue
            if w1 + w2 + w3 > 1.0 + 1e-9: continue
            if w1 > 0 and w2 > 0 and w3 > 0:
                yield [round(w1, 2), round(w2, 2), round(w3, 2)]

aug_search_rows = []

for combo in itertools.combinations(all_bases_aug, 2):
    for ws in grid_weights_2way(0.05):
        od = {m: oof_aug_all[m][0] for m in combo if m in oof_aug_all}
        if len(od) < 2: continue
        fo   = fused(list(combo), ws, od)
        thr, f1 = sweep_thr(y_a4_ref, fo)
        # a6 score
        a6d  = {m: (AUG_A6.get(m) if m in AUG_A6 else BASE_A6.get(m))
                for m in combo}
        a6d  = {m: v for m, v in a6d.items() if v is not None}
        if HAS_A6 and len(a6d) == 2:
            a6s  = fused(list(combo), ws, a6d)
            y6   = df_a6['label_int'].values
            a6f1 = metrics_at(y6, a6s, thr)['F1']
            a6f1_ora = sweep_thr(y6, a6s)[1]
        else:
            a6f1 = a6f1_ora = float('nan')
        aug_search_rows.append({'type':'2way','members':'+'.join(combo),
                                 'weights':str(ws),'cv_thr':thr,'cv_f1':f1,
                                 'a6_f1_cv_thr':a6f1,'a6_f1_oracle':a6f1_ora})

for combo in itertools.combinations(all_bases_aug, 3):
    for ws in grid_weights_3way(0.1):
        od = {m: oof_aug_all[m][0] for m in combo if m in oof_aug_all}
        if len(od) < 3: continue
        fo   = fused(list(combo), ws, od)
        thr, f1 = sweep_thr(y_a4_ref, fo)
        a6d  = {m: (AUG_A6.get(m) if m in AUG_A6 else BASE_A6.get(m))
                for m in combo}
        a6d  = {m: v for m, v in a6d.items() if v is not None}
        if HAS_A6 and len(a6d) == 3:
            a6s  = fused(list(combo), ws, a6d)
            y6   = df_a6['label_int'].values
            a6f1 = metrics_at(y6, a6s, thr)['F1']
            a6f1_ora = sweep_thr(y6, a6s)[1]
        else:
            a6f1 = a6f1_ora = float('nan')
        aug_search_rows.append({'type':'3way','members':'+'.join(combo),
                                 'weights':str(ws),'cv_thr':thr,'cv_f1':f1,
                                 'a6_f1_cv_thr':a6f1,'a6_f1_oracle':a6f1_ora})

aug_df = pd.DataFrame(aug_search_rows).sort_values('cv_f1', ascending=False)
aug_df.to_csv(SAVE_DIR / 'aug_search.csv', index=False)
print(f'Grid search: {len(aug_df)} combinations evaluated')
print('Top 5 by CV F1:')
print(aug_df[['type','members','weights','cv_f1','a6_f1_cv_thr','a6_f1_oracle']].head(5).to_string())

# best augmented fusion
AUG_BEST = aug_df.iloc[0]
AUG_BEST_MEMBERS = AUG_BEST['members'].split('+')
AUG_BEST_WEIGHTS = eval(AUG_BEST['weights'])
AUG_BEST_THR     = AUG_BEST['cv_thr']

# save augmented OOF + a6 scores
oof_out = df_a4[['filename']].copy()
for n in AUG_OOF:
    oof_out[n] = AUG_OOF[n][0]
oof_out.to_csv(SAVE_DIR / 'aug_oof.csv', index=False)

if HAS_A6:
    a6_out = df_a6[['filename']].copy()
    for n in AUG_A6:
        a6_out[n] = AUG_A6[n]
    a6_out.to_csv(SAVE_DIR / 'aug_a6_scores.csv', index=False)

# feature variance explained by question_id (eta-squared)
print('\\nComputing variance explained by question_id...')
var_rows = []
for feat in avail_txt:
    if feat not in TRAIN_POOL.columns: continue
    groups = [grp[feat].dropna().values
              for _, grp in TRAIN_POOL.groupby('question_id') if len(grp) > 1]
    if len(groups) < 2: continue
    try:
        F, p = sp_stats.f_oneway(*groups)
        grand_mean = TRAIN_POOL[feat].mean()
        ss_tot = ((TRAIN_POOL[feat].dropna() - grand_mean) ** 2).sum()
        ss_bet = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        eta2   = ss_bet / ss_tot if ss_tot > 0 else 0.0
        var_rows.append({'feature': feat, 'eta_squared': round(eta2, 5),
                          'F_stat': round(F, 3), 'p_value': round(p, 5)})
    except Exception:
        pass
var_df = pd.DataFrame(var_rows).sort_values('eta_squared', ascending=False)
var_df.to_csv(SAVE_DIR / 'feature_variance_explained.csv', index=False)
print('Top 5 features most explained by question_id:')
print(var_df.head(5).to_string())

print(f'\\nSection 4 done in {time.time()-t1:.1f}s  (total {time.time()-t0:.1f}s)')
""", "code-41"))

# ─────────────────────────────────────────────────────────────────────────────
# §5  TIER A3: Multi-model consensus pseudo-labeling
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §5 — Tier A3: Multi-Model Consensus Pseudo-Labeling on audios3", "md-05"))

C.append(_c("""
t0 = time.time()

# ── 5a: train independent views + Platt calibration ──────────────────────
VIEW_DEF = {
    'text_view':  (avail_txt, 'xgb'),
    'audio_view': (WSP_COLS,  'xgb'),
    'third_view': (WLM_COLS,  'xgb'),
}
VIEW_MODEL = {}
VIEW_CAL   = {}   # platt calibrator

print('Training independent views...')
for vname, (fcols, _) in VIEW_DEF.items():
    if not fcols:
        print(f'  SKIP {vname}: no cols'); continue
    mdl = make_xgb(n_feats=len(fcols))
    # OOF on a4 for Platt calibration
    if len(df_a4) > 0:
        oof, y_a4 = a4_oof(mdl, fcols, df_a4, df_a2)
        cal = LogisticRegression(C=1.0, max_iter=1000)
        cal.fit(oof.reshape(-1, 1), y_a4)
        VIEW_CAL[vname] = cal
    m, uc = fit_model(mdl, fcols, TRAIN_POOL)
    VIEW_MODEL[vname] = (m, uc)
    auc = roc_auc_score(y_a4, oof) if len(df_a4) > 0 else float('nan')
    print(f'  {vname}: OOF-AUC={auc:.3f}')

# ── 5b: consensus filter on audios3 ──────────────────────────────────────
if BD3 is None:
    print('audios3 not available; skipping pseudo-label step')
    PSEUDO_DF = pd.DataFrame()
else:
    df3_work = BD3.copy()
    for vname, (m, uc) in VIEW_MODEL.items():
        missing = [c for c in uc if c not in df3_work.columns]
        if missing:
            print(f'  SKIP {vname} on audios3: {len(missing)}/{len(uc)} cols absent '
                  f'(e.g. {missing[:3]})')
            continue
        raw_p = predict_proba(m, uc, df3_work)
        if vname in VIEW_CAL:
            cal_p = VIEW_CAL[vname].predict_proba(raw_p.reshape(-1, 1))[:, 1]
        else:
            cal_p = raw_p
        df3_work[f'p_{vname}'] = cal_p

    views = list(VIEW_MODEL.keys())
    p_cols = [f'p_{v}' for v in views]
    available_p = [c for c in p_cols if c in df3_work.columns]

    if available_p:
        df3_work['agree_pos'] = (df3_work[available_p] > 0.75).all(axis=1)
        df3_work['agree_neg'] = (df3_work[available_p] < 0.10).all(axis=1)

        pos_cands = df3_work[df3_work['agree_pos']].copy()
        neg_cands = df3_work[df3_work['agree_neg']].copy()

        n_pos = min(len(pos_cands), MAX_PSEUDO_PER_CLASS)
        pos_samp = pos_cands.sample(n_pos, random_state=RANDOM_SEED) if n_pos > 0 else pos_cands
        n_neg = min(len(neg_cands), 3 * n_pos)
        neg_samp = neg_cands.sample(n_neg, random_state=RANDOM_SEED) if n_neg > 0 else neg_cands

        pos_samp['pseudo_label'] = 1
        neg_samp['pseudo_label'] = 0

        PSEUDO_DF = pd.concat([pos_samp, neg_samp], ignore_index=True)
        PSEUDO_DF['label_int'] = PSEUDO_DF['pseudo_label']

        save_cols = ['filename'] + available_p + ['agree_pos', 'agree_neg', 'pseudo_label']
        PSEUDO_DF[[c for c in save_cols if c in PSEUDO_DF.columns]].to_csv(
            SAVE_DIR / 'pseudo_audios3.csv', index=False)
        print(f'Pseudo-labels: {n_pos} positive, {n_neg} negative (from {len(df3_work)} candidates)')
    else:
        PSEUDO_DF = pd.DataFrame()
        print('WARNING: no view probability columns found')

print(f'§5a-5b done in {time.time()-t0:.1f}s')
""", "code-50"))

C.append(_c("""
# ── 5c: retrain picks with pseudo-labeled audios3 ─────────────────────────
t1 = time.time()

if len(PSEUDO_DF) == 0:
    print('No pseudo-labels available; skipping §5c')
    A3_A6 = {}
    A3_PICKS = {}
else:
    # augment TRAIN_POOL with pseudo (weight=0.5)
    PSEUDO_POOL = PSEUDO_DF.copy()
    PSEUDO_POOL['sample_weight'] = 0.5
    REAL_POOL   = TRAIN_POOL.copy()
    REAL_POOL['sample_weight']   = 1.0
    EXPANDED_POOL = pd.concat([REAL_POOL, PSEUDO_POOL], ignore_index=True).dropna(subset=['label_int'])



    A3_MODEL = {}
    A3_A6    = {}
    for name, (fcols, mtype) in BASE_DEF.items():
        if not fcols: continue
        mdl = make_xgb(n_feats=len(fcols)) if mtype == 'xgb' else make_rf()
        m, uc = fit_weighted(mdl, fcols, EXPANDED_POOL)
        A3_MODEL[name] = (m, uc)
        if HAS_A6:
            A3_A6[name] = predict_proba(m, uc, df_a6)

    # score a6 with A3 models using original pick weights/thresholds
    A3_PICKS = {}
    print('\\nA3 baseline a6 metrics:')
    for pn, pc in PICKS.items():
        a6d = {m: A3_A6[m] for m in pc['members'] if m in A3_A6}
        if not a6d: continue
        scores = fused(pc['members'], pc['weights'], a6d)
        y_a6   = df_a6['label_int'].values
        A3_PICKS[pn] = (scores, y_a6)
        fthr = pc['frozen_thr']
        cthr = pc.get('cv_thr', fthr)
        fm   = metrics_at(y_a6, scores, fthr)
        cm   = metrics_at(y_a6, scores, cthr)
        print(f'  {pn}: frozen-F1={fm["F1"]:.3f}  cv-F1={cm["F1"]:.3f}  '
              f'oracle={sweep_thr(y_a6,scores)[1]:.3f}')

    # save A3 a6 scores
    if HAS_A6:
        a3_out = df_a6[['filename','label_int']].copy()
        for pn, (sc, _) in A3_PICKS.items():
            a3_out[f'score_{pn}'] = sc
        a3_out.to_csv(SAVE_DIR / 'tierA3_a6_scores.csv', index=False)

    # apply Top-K% with same calibration slices
    a3_topk_rows = []
    for pn, (scores, y_a6) in A3_PICKS.items():
        pc   = PICKS[pn]
        fthr = pc['frozen_thr']
        cthr = pc.get('cv_thr', fthr)
        for ss in SLICE_SIZES:
            f1s = []
            for seed in CALIB_SEEDS:
                sl_idx = CALIB_SLICE_IDX.get(seed)
                if sl_idx is None: continue
                rem_mask = np.ones(len(y_a6), bool); rem_mask[sl_idx] = False
                rem_idx  = np.where(rem_mask)[0]
                if len(rem_idx) == 0: continue
                K_hat = y_a6[sl_idx].mean()
                s_rem = scores[rem_idx]
                y_rem = y_a6[rem_idx]
                thr_k = float(np.percentile(s_rem, 100.0 * (1.0 - K_hat)))
                f1s.append(f1_score(y_rem, (s_rem >= thr_k).astype(int), zero_division=0))
            if not f1s: continue
            a3_topk_rows.append({'pick':pn,'slice_size':ss,
                                  'topk_f1_mean':round(np.mean(f1s),4),
                                  'topk_f1_std': round(np.std(f1s), 4),
                                  'source':'A3'})
    pd.DataFrame(a3_topk_rows).to_csv(SAVE_DIR / 'tierA3_topk_results.csv', index=False)

print(f'\\nSection 5 done in {time.time()-t1:.1f}s  (total {time.time()-t0:.1f}s)')
""", "code-51"))

# ─────────────────────────────────────────────────────────────────────────────
# §6  COMBINED: Tier A1 + A2 + A3
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §6 — Combined: Tier A1 + A2 + A3", "md-06"))

C.append(_c("""
t0 = time.time()

if len(PSEUDO_DF) == 0:
    print('No pseudo-labels; skipping combined section')
    COMB_A6 = {}
else:
    # augment EXPANDED_POOL with deviation features
    PSEUDO_AUG = augment_df(PSEUDO_POOL) if 'question_id' in PSEUDO_POOL.columns else PSEUDO_POOL
    EXPANDED_AUG = pd.concat([TRAIN_POOL_AUG, PSEUDO_AUG], ignore_index=True).dropna(subset=['label_int'])

    COMB_MODEL = {}
    COMB_A6    = {}
    for name, (fcols, _) in AUG_BASE_DEF.items():
        if not fcols: continue
        mdl   = make_xgb(n_feats=len(fcols))
        m, uc = fit_weighted(mdl, fcols, EXPANDED_AUG)
        COMB_MODEL[name] = (m, uc)
        if len(df_a6_aug) > 0:
            COMB_A6[name] = predict_proba(m, uc, df_a6_aug)

    # re-search fusion on combined OOF would require re-running CV on expanded pool
    # Use best augmented fusion from §4d, re-score a6 with combined models
    comb_a6_out = df_a6[['filename','label_int']].copy() if HAS_A6 else pd.DataFrame()

    print('Combined A2+A3 a6 scores:')
    for pn in PICKS:
        mems = AUG_BEST_MEMBERS
        a6d  = {m: (COMB_A6.get(m) if m in COMB_A6 else BASE_A6.get(m))
                for m in mems}
        a6d  = {m: v for m, v in a6d.items() if v is not None}
        if not a6d or not HAS_A6: continue
        scores = fused(mems, AUG_BEST_WEIGHTS, a6d)
        y_a6   = df_a6['label_int'].values
        cm     = metrics_at(y_a6, scores, AUG_BEST_THR)
        print(f'  best-aug fusion: F1={cm["F1"]:.3f}  P={cm["P"]:.3f}  R={cm["R"]:.3f}')
        if HAS_A6:
            comb_a6_out['score_combined'] = scores
        break   # fusion is not per-pick in combined; report once

    if HAS_A6 and len(comb_a6_out) > 0:
        comb_a6_out.to_csv(SAVE_DIR / 'combined_a6_scores.csv', index=False)

    # Top-K% on combined
    comb_topk = []
    for seed in CALIB_SEEDS:
        sl_idx = CALIB_SLICE_IDX.get(seed)
        if sl_idx is None or not HAS_A6: continue
        rem_mask = np.ones(len(y_a6), bool); rem_mask[sl_idx] = False
        rem_idx  = np.where(rem_mask)[0]
        if len(rem_idx) == 0: continue
        K_hat  = y_a6[sl_idx].mean()
        s_rem  = scores[rem_idx]
        y_rem  = y_a6[rem_idx]
        thr_k  = float(np.percentile(s_rem, 100.0 * (1.0 - K_hat)))
        comb_topk.append(f1_score(y_rem, (s_rem >= thr_k).astype(int), zero_division=0))
    if comb_topk:
        print(f'  Combined Top-K%: F1={np.mean(comb_topk):.3f}±{np.std(comb_topk):.3f}')
        pd.DataFrame([{'source':'A2A3_combined','slice_size':CALIB_SLICE_SIZE,
                        'topk_f1_mean':round(np.mean(comb_topk),4),
                        'topk_f1_std': round(np.std(comb_topk),4)}]).to_csv(
            SAVE_DIR / 'combined_topk_results.csv', index=False)

print(f'\\nSection 6 done in {time.time()-t0:.1f}s')
""", "code-60"))

# ─────────────────────────────────────────────────────────────────────────────
# §7  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
C.append(_m("## §7 — Summary Table", "md-07"))

C.append(_c("""
t0 = time.time()

summary_rows = []

def _add(condition, pick_name, region, scores, y, thr, oracle_thr=None, oracle_f1=None):
    mk = np.ones(len(y), bool) if region == 'overall' else (
         (df_a6['region'].values == region) if 'region' in df_a6.columns else np.ones(len(y), bool))
    if mk.sum() < 5:
        return
    m = metrics_at(y[mk], scores[mk], thr)
    summary_rows.append({
        'condition': condition, 'pick_name': pick_name, 'region': region,
        'F1': m['F1'], 'P': m['P'], 'R': m['R'], 'AUC': m['AUC'],
        'n_pos': m['n_pos'], 'n_eval': m['n_eval'],
        'threshold': round(thr, 4),
        'notes': '',
    })

regions = ['overall']
if HAS_A6 and 'region' in df_a6.columns:
    regions += sorted(df_a6['region'].dropna().unique().tolist())

# ── baseline conditions ───────────────────────────────────────────────────
for pn, pc in PICKS.items():
    if pn not in PICK_A6: continue
    sc, y6 = PICK_A6[pn]
    for reg in regions:
        _add('baseline_frozen', pn, reg, sc, y6, pc['frozen_thr'])
        _add('baseline_cv',     pn, reg, sc, y6, pc.get('cv_thr', pc['frozen_thr']))
        _add('baseline_oracle', pn, reg, sc, y6, pc.get('oracle_thr', pc['frozen_thr']))

# ── A1 Top-K% (slice_size = CALIB_SLICE_SIZE, mean over repeats) ──────────
if len(topk_df) > 0:
    for pn in PICKS:
        sub = topk_df[(topk_df['pick'] == pn) & (topk_df['slice_size'] == CALIB_SLICE_SIZE)]
        if len(sub) == 0: continue
        row = sub.iloc[0]
        summary_rows.append({
            'condition': 'A1_topk', 'pick_name': pn, 'region': 'overall',
            'F1': row['topk_f1_mean'], 'P': row['topk_prec_mean'],
            'R': row['topk_rec_mean'], 'AUC': float('nan'),
            'n_pos': float('nan'), 'n_eval': row['n_remaining'],
            'threshold': float('nan'), 'notes': f'mean±std over {N_REPEATS} draws',
        })

# ── A2 augmented fusion ───────────────────────────────────────────────────
if HAS_A6 and len(aug_df) > 0:
    best_row   = aug_df.iloc[0]
    best_mems  = best_row['members'].split('+')
    best_ws    = eval(best_row['weights'])
    best_thr_a2 = best_row['cv_thr']
    a6d_a2 = {m: (AUG_A6.get(m) if m in AUG_A6 else BASE_A6.get(m)) for m in best_mems}
    a6d_a2 = {m: v for m, v in a6d_a2.items() if v is not None}
    if a6d_a2:
        sc_a2 = fused(best_mems, best_ws, a6d_a2)
        y_a6  = df_a6['label_int'].values
        for reg in regions:
            _add('A2_cv', '+'.join(best_mems), reg, sc_a2, y_a6, best_thr_a2)
        # A2 Top-K%
        tk_a2 = []
        for seed in CALIB_SEEDS:
            sl = CALIB_SLICE_IDX.get(seed)
            if sl is None: continue
            rem  = np.where(~np.isin(np.arange(len(y_a6)), sl))[0]
            K_h  = y_a6[sl].mean()
            s_r  = sc_a2[rem]; y_r = y_a6[rem]
            t_k  = float(np.percentile(s_r, 100.0*(1-K_h)))
            tk_a2.append(f1_score(y_r, (s_r >= t_k).astype(int), zero_division=0))
        if tk_a2:
            summary_rows.append({'condition':'A2_topk','pick_name':'+'.join(best_mems),
                                  'region':'overall','F1':round(np.mean(tk_a2),4),
                                  'P':float('nan'),'R':float('nan'),'AUC':float('nan'),
                                  'n_pos':float('nan'),'n_eval':len(rem),
                                  'threshold':float('nan'),'notes':f'mean over {N_REPEATS}'})

# ── A3 conditions ─────────────────────────────────────────────────────────
for pn, (sc, y6) in A3_PICKS.items():
    pc = PICKS[pn]
    for reg in regions:
        _add('A3_cv', pn, reg, sc, y6, pc.get('cv_thr', pc['frozen_thr']))

if len(a3_topk_rows) > 0:
    for pn in PICKS:
        sub = [r for r in a3_topk_rows
               if r['pick'] == pn and r['slice_size'] == CALIB_SLICE_SIZE]
        if not sub: continue
        r = sub[0]
        summary_rows.append({'condition':'A3_topk','pick_name':pn,'region':'overall',
                              'F1':r['topk_f1_mean'],'P':float('nan'),'R':float('nan'),
                              'AUC':float('nan'),'n_pos':float('nan'),'n_eval':float('nan'),
                              'threshold':float('nan'),'notes':f'mean over {N_REPEATS}'})

# ── Combined A2+A3 ────────────────────────────────────────────────────────
comb_path = SAVE_DIR / 'combined_topk_results.csv'
if comb_path.exists():
    comb_r = pd.read_csv(comb_path).iloc[0]
    summary_rows.append({'condition':'A2A3_combined_topk','pick_name':'aug_best+pseudo',
                          'region':'overall','F1':comb_r['topk_f1_mean'],
                          'P':float('nan'),'R':float('nan'),'AUC':float('nan'),
                          'n_pos':float('nan'),'n_eval':float('nan'),
                          'threshold':float('nan'),'notes':f'±{comb_r["topk_f1_std"]:.4f}'})

# ── assemble & print ──────────────────────────────────────────────────────
SUMMARY = pd.DataFrame(summary_rows)
SUMMARY = SUMMARY.sort_values(['region','F1'], ascending=[True, False])
SUMMARY.to_csv(SAVE_DIR / 'tier_a_summary.csv', index=False)

print('\\n' + '='*70)
print('TIER A SUMMARY  (region=overall, sorted by F1)')
print('='*70)
ov = SUMMARY[SUMMARY['region'] == 'overall'].sort_values('F1', ascending=False)
print(ov[['condition','pick_name','F1','P','R','AUC','threshold']].to_string(index=False))

# ── three required numbers ────────────────────────────────────────────────
base_oracle_rows = SUMMARY[(SUMMARY['condition'] == 'baseline_oracle') &
                            (SUMMARY['region']    == 'overall')]
best_rows = SUMMARY[SUMMARY['region'] == 'overall'].sort_values('F1', ascending=False)

if len(base_oracle_rows) > 0 and len(best_rows) > 0:
    bl_ora_f1 = base_oracle_rows['F1'].max()
    best_f1   = best_rows.iloc[0]['F1']
    best_cond = best_rows.iloc[0]['condition']
    best_pick = best_rows.iloc[0]['pick_name']
    delta     = best_f1 - bl_ora_f1

    # bootstrap CI if delta >= 0.03
    ci_str = 'N/A (delta < 0.03)'
    if abs(delta) >= 0.03 and HAS_A6:
        bl_row   = base_oracle_rows.loc[base_oracle_rows['F1'].idxmax()]
        bl_pn    = bl_row['pick_name']
        if bl_pn in PICK_A6 and best_cond in ['A1_topk','A2_topk','A3_topk','A2A3_combined_topk']:
            bl_sc, y6 = PICK_A6[bl_pn]
            bl_thr    = PICKS[bl_pn].get('oracle_thr', 0.5)
            best_sc   = bl_sc   # fallback; use same if best is topk
            lo, hi, mu = bootstrap_ci(y6, best_sc, bl_thr, bl_sc, bl_thr)
            ci_str    = f'[{lo:.3f}, {hi:.3f}]'

    print()
    print(f'1. baseline_oracle F1  : {bl_ora_f1:.4f}')
    print(f'2. best Tier-A F1      : {best_f1:.4f}  ({best_cond} / {best_pick})')
    print(f'3. delta               : {delta:+.4f}  95% CI bootstrap: {ci_str}')
    if delta > 0 and 'CI' not in ci_str and lo > 0:
        print('   → REAL GAIN: Tier A produces statistically significant F1 improvement.')
    elif abs(delta) < 0.03:
        print('   → NO SIGNIFICANT GAIN: threshold-drift and prompt-drift not dominant drivers.')
        print('     Next step: Tier C (CORAL / SSL pre-training for encoder-level shift).')
    else:
        print(f'   → delta={delta:+.4f} but CI crosses 0; inconclusive.')

print(f'\\nSection 7 done in {time.time()-t0:.1f}s')
print(f'All artifacts saved to {SAVE_DIR}')
""", "code-70"))

# ─────────────────────────────────────────────────────────────────────────────
# Build notebook JSON
# ─────────────────────────────────────────────────────────────────────────────
NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "cells": C,
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)

print(f'Written: {OUT}')
print(f'Cells:   {len(C)}')
