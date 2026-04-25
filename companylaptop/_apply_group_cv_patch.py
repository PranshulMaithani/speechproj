"""One-shot patch script — converts fusion_text_wavlm.ipynb to use:
  - candidate-grouped CV (StratifiedGroupKFold) everywhere a fold split happens
  - fine-tuned WavLM (wavlm_whole_ft) loaded alongside pretrained
After running once, this script can be deleted.
"""
import json, sys
from pathlib import Path

NB = Path(__file__).with_name('fusion_text_wavlm.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

def src(i): return ''.join(nb['cells'][i]['source'])
def setsrc(i, new): nb['cells'][i]['source'] = new.splitlines(keepends=True)

# ---------- Cell 1: update _active list ----------
c1 = src(1)
c1 = c1.replace(
    "_active = ['wavlm_wp', 'whisper_wp_xgb', 'text_stylo', 'text_top10', 'text_top15']",
    "_active = ['wavlm_wp', 'wavlm_whole_ft', 'whisper_wp_xgb', 'text_stylo', 'text_top10', 'text_top15']"
)
setsrc(1, c1)

# ---------- Cell 2: add re, StratifiedGroupKFold ----------
c2 = src(2)
c2 = c2.replace(
    "import json, itertools, warnings\n",
    "import json, itertools, warnings, re\n"
)
c2 = c2.replace(
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold\n"
)
# add USE_GROUP_CV flag near the top after np.random.seed
c2 = c2.replace(
    "np.random.seed(RANDOM_SEED)\n",
    "np.random.seed(RANDOM_SEED)\n\nUSE_GROUP_CV = True   # candidate-aware CV; set False to revert to leaky StratifiedKFold\n"
)
setsrc(2, c2)

# ---------- Cell 4: candidate_id parsing + load fine-tuned WavLM + wft_cols + X_wft ----------
c4 = src(4)

# Insert parse helper at the top of the cell
parse_helper = (
    "# Filenames are <candidate_id>_<question>.<ext>; group folds by candidate_id so a\n"
    "# candidate's Q25/Q26/Q27 never end up on both sides of a fold.\n"
    "_CANDIDATE_RE = re.compile(r'^(.+)_(\\d{1,3})\\.[a-zA-Z0-9]+$')\n"
    "def _parse_candidate_id(fn):\n"
    "    m = _CANDIDATE_RE.match(str(fn))\n"
    "    return m.group(1) if m else None\n\n"
)
assert c4.startswith("def load_gt(name):"), 'unexpected cell 4 layout'
c4 = parse_helper + c4

# Add finetuned-wavlm path helper next to _wavlm_wp_path
c4 = c4.replace(
    "def _wavlm_wp_path(n):\n"
    "    p = NB_DIR / f'{n}_whole_pretrained.csv'\n"
    "    if p.exists(): return p\n"
    "    alt = NB_DIR / f'{n}_wavlm_whole.csv'\n"
    "    if alt.exists(): return alt\n"
    "    raise FileNotFoundError(f'{p} or {alt}')\n",
    "def _wavlm_wp_path(n):\n"
    "    p = NB_DIR / f'{n}_whole_pretrained.csv'\n"
    "    if p.exists(): return p\n"
    "    alt = NB_DIR / f'{n}_wavlm_whole.csv'\n"
    "    if alt.exists(): return alt\n"
    "    raise FileNotFoundError(f'{p} or {alt}')\n\n"
    "def _wavlm_wft_path(n):\n"
    "    p = NB_DIR / f'{n}_whole_finetuned.csv'\n"
    "    return p if p.exists() else None\n"
)

# Patch load_folder body to also load wft + tag candidate_id
old_loader = (
    "def load_folder(name):\n"
    "    gt      = load_gt(name)\n"
    "    text    = pd.read_csv(NB_DIR / f'{name}_features.csv')\n"
    "    wavlm   = pd.read_csv(_wavlm_wp_path(name))\n"
    "    whisper = pd.read_csv(NB_DIR / f'{name}_whisper_whole.csv')\n"
    "    df = (gt.merge(text,    on='filename', how='inner')\n"
    "            .merge(wavlm,   on='filename', how='inner', suffixes=('','_wp'))\n"
    "            .merge(whisper, on='filename', how='inner', suffixes=('','_wh')))\n"
    "    df['batch'] = name\n"
    "    return df\n"
)
new_loader = (
    "def load_folder(name):\n"
    "    gt      = load_gt(name)\n"
    "    text    = pd.read_csv(NB_DIR / f'{name}_features.csv')\n"
    "    wavlm   = pd.read_csv(_wavlm_wp_path(name))\n"
    "    whisper = pd.read_csv(NB_DIR / f'{name}_whisper_whole.csv')\n"
    "    df = (gt.merge(text,    on='filename', how='inner')\n"
    "            .merge(wavlm,   on='filename', how='inner', suffixes=('','_wp'))\n"
    "            .merge(whisper, on='filename', how='inner', suffixes=('','_wh')))\n"
    "    # Fine-tuned WavLM (whole) — suffixed _wft to avoid colliding with pretrained\n"
    "    wft_path = _wavlm_wft_path(name)\n"
    "    if wft_path is not None:\n"
    "        wft = pd.read_csv(wft_path)\n"
    "        wft = wft.rename(columns={c: (c if c == 'filename' else f'{c}_wft') for c in wft.columns})\n"
    "        df = df.merge(wft, on='filename', how='inner')\n"
    "    df['candidate_id'] = df['filename'].astype(str).map(_parse_candidate_id)\n"
    "    df['batch'] = name\n"
    "    df.attrs['has_wft'] = wft_path is not None\n"
    "    return df\n"
)
assert old_loader in c4, 'load_folder body did not match — abort'
c4 = c4.replace(old_loader, new_loader)

# Add wft_cols / X_wft right after X_stylo
c4 = c4.replace(
    "def X_stylo(df): return df[stylo_cols].fillna(0).values\n",
    "def X_stylo(df): return df[stylo_cols].fillna(0).values\n\n"
    "wft_cols = [c for c in _first.columns if c.endswith('_wft')]\n"
    "def X_wft(df): return df[wft_cols].fillna(0).values\n"
)

# Update the print line to also report wft and candidate count
c4 = c4.replace(
    "print(f'Text feats: {len(text_cols)}  |  Stylo: {len(stylo_cols)}  |  WP: {len(wp_cols)}  |  Whisper: {len(wh_cols)}')\n",
    "print(f'Text feats: {len(text_cols)}  |  Stylo: {len(stylo_cols)}  |  WP: {len(wp_cols)}  |  WFT: {len(wft_cols)}  |  Whisper: {len(wh_cols)}')\n"
)
c4 = c4.replace(
    "for name, df in batches.items():\n"
    "    y = df['label_int'].values\n"
    "    print(f'  {name}: n={len(df):4d}  cheat={int((y==1).sum()):3d}  honest={int((y==0).sum()):3d}')\n",
    "for name, df in batches.items():\n"
    "    y = df['label_int'].values\n"
    "    n_cand = df['candidate_id'].nunique()\n"
    "    has_wft = df.attrs.get('has_wft', False)\n"
    "    print(f'  {name}: n={len(df):4d}  cheat={int((y==1).sum()):3d}  honest={int((y==0).sum()):3d}  candidates={n_cand}  wft={\"y\" if has_wft else \"n\"}')\n",
)
setsrc(4, c4)

# ---------- Cell 10: registry adds wavlm_whole_ft ----------
c10 = src(10)
c10 = c10.replace(
    "    return {\n"
    "        'wavlm_wp':       (X_wp,    lambda: make_xgb(len(wp_cols))),\n"
    "        'whisper_wp_xgb': (X_wh,    lambda: make_xgb(len(wh_cols))),\n",
    "    return {\n"
    "        'wavlm_wp':       (X_wp,    lambda: make_xgb(len(wp_cols))),\n"
    "        'wavlm_whole_ft': (X_wft,   lambda: make_xgb(len(wft_cols))),\n"
    "        'whisper_wp_xgb': (X_wh,    lambda: make_xgb(len(wh_cols))),\n"
)
# If wft is missing in the data, drop the head to avoid runtime error
c10 = c10.replace(
    "    return {\n",
    "    reg = {\n"
).replace(
    "        'text_top15':     (X_top15, lambda: make_xgb(15)),\n"
    "    }\n",
    "        'text_top15':     (X_top15, lambda: make_xgb(15)),\n"
    "    }\n"
    "    if not wft_cols:\n"
    "        reg.pop('wavlm_whole_ft', None)\n"
    "    return reg\n"
)
setsrc(10, c10)

# ---------- Cell 12: StratifiedGroupKFold + expose cv_groups ----------
c12 = src(12)
old12 = (
    "skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)\n"
    "for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df_cv_target, y_target)):\n"
)
new12 = (
    "cv_groups = df_cv_target['candidate_id'].values\n"
    "if USE_GROUP_CV and pd.notna(cv_groups).all():\n"
    "    skf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)\n"
    "    split_iter = skf.split(df_cv_target, y_target, groups=cv_groups)\n"
    "    print(f'CV protocol: StratifiedGroupKFold over {pd.Series(cv_groups).nunique()} candidates')\n"
    "else:\n"
    "    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)\n"
    "    split_iter = skf.split(df_cv_target, y_target)\n"
    "    print('CV protocol: StratifiedKFold (LEAKY — candidate_id missing or USE_GROUP_CV=False)')\n"
    "for fold_idx, (tr_idx, va_idx) in enumerate(split_iter):\n"
)
assert old12 in c12
c12 = c12.replace(old12, new12)
setsrc(12, c12)

# ---------- Cell 16: stacker uses GroupKFold ----------
c16 = src(16)
old16 = (
    "X_meta_cv = np.column_stack([cv_scores[m] for m in model_names])\n"
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)\n"
    "p_meta_oof = np.zeros(len(cv_y))\n"
    "for tr, va in skf.split(X_meta_cv, cv_y):\n"
)
new16 = (
    "X_meta_cv = np.column_stack([cv_scores[m] for m in model_names])\n"
    "if USE_GROUP_CV and 'cv_groups' in dir() and pd.notna(cv_groups).all():\n"
    "    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)\n"
    "    meta_iter = skf.split(X_meta_cv, cv_y, groups=cv_groups)\n"
    "else:\n"
    "    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)\n"
    "    meta_iter = skf.split(X_meta_cv, cv_y)\n"
    "p_meta_oof = np.zeros(len(cv_y))\n"
    "for tr, va in meta_iter:\n"
)
assert old16 in c16
c16 = c16.replace(old16, new16)
setsrc(16, c16)

# ---------- Cell 20: run_pipeline_ex uses GroupKFold + fine-tuned WavLM head ----------
c20 = src(20)

# Add wft column list inside run_pipeline_ex (right after wp_cols_l)
c20 = c20.replace(
    "    wp_cols_l    = [c for c in first.columns if c.startswith('wavlm_')\n"
    "                    and not c.startswith('wavlm_mean_') and not c.startswith('wavlm_std_')]\n"
    "    wh_cols_l    = [c for c in first.columns if c.startswith('whisper_')]\n",
    "    wp_cols_l    = [c for c in first.columns if c.startswith('wavlm_')\n"
    "                    and not c.startswith('wavlm_mean_') and not c.startswith('wavlm_std_')\n"
    "                    and not c.endswith('_wft')]\n"
    "    wft_cols_l   = [c for c in first.columns if c.endswith('_wft')]\n"
    "    wh_cols_l    = [c for c in first.columns if c.startswith('whisper_')]\n"
)

# Add wavlm_whole_ft to local registry
c20 = c20.replace(
    "    registry = {\n"
    "        'wavlm_wp':       (mk_X(wp_cols_l),    lambda: make_xgb(len(wp_cols_l))),\n"
    "        'whisper_wp_xgb': (mk_X(wh_cols_l),    lambda: make_xgb(len(wh_cols_l))),\n",
    "    registry = {\n"
    "        'wavlm_wp':       (mk_X(wp_cols_l),    lambda: make_xgb(len(wp_cols_l))),\n"
    "        'whisper_wp_xgb': (mk_X(wh_cols_l),    lambda: make_xgb(len(wh_cols_l))),\n"
)
# Insert wavlm_whole_ft conditionally just before the closing brace of the registry dict
c20 = c20.replace(
    "        'text_top15':     (mk_X(top15_l),      lambda: make_xgb(15)),\n"
    "    }\n",
    "        'text_top15':     (mk_X(top15_l),      lambda: make_xgb(15)),\n"
    "    }\n"
    "    if wft_cols_l:\n"
    "        registry['wavlm_whole_ft'] = (mk_X(wft_cols_l), lambda: make_xgb(len(wft_cols_l)))\n"
)

# Switch internal CV to GroupKFold
old20 = (
    "    cv_sc = {m: np.full(len(df_cvt), np.nan) for m in registry}\n"
    "    skf_l = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)\n"
    "    for tr_idx, va_idx in skf_l.split(df_cvt, y_cv_l):\n"
)
new20 = (
    "    cv_sc = {m: np.full(len(df_cvt), np.nan) for m in registry}\n"
    "    groups_l = df_cvt['candidate_id'].values\n"
    "    if USE_GROUP_CV and pd.notna(groups_l).all():\n"
    "        skf_l = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)\n"
    "        split_l = skf_l.split(df_cvt, y_cv_l, groups=groups_l)\n"
    "    else:\n"
    "        skf_l = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)\n"
    "        split_l = skf_l.split(df_cvt, y_cv_l)\n"
    "    for tr_idx, va_idx in split_l:\n"
)
assert old20 in c20
c20 = c20.replace(old20, new20)
setsrc(20, c20)

# ---------- Write back ----------
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('PATCHED:', NB)
