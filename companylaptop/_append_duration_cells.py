"""Append §13 (speaking-time computation + per-batch plot) and
§14 (min-duration filter + rotation re-run) to honest_eval_and_improve.ipynb.
One-shot — safe to delete after running.
"""
import json, ast
from pathlib import Path

NB = Path(__file__).with_name('honest_eval_and_improve.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

# =====================================================================
# §13 — Speaking-time per audio + per-batch histograms
# =====================================================================
md13 = """## 13. Speaking-time per audio + per-batch distribution

The §11 audit surfaced a systematic confound: candidates whose audios are very short tend to either repeat the question or speak only a sentence, which acoustically looks scripted (low pause variance, fluent delivery) without the candidate actually cheating. Both whisper and wavlm_ft fire on these.

This cell computes per-audio:
- `total_duration_s` — file length, from `soundfile.info` (cheap, no audio loaded).
- `speaking_time_s` — sum of whisper segment durations if a `{batch}_transcripts.json` file is present; otherwise falls back to `n_words / words_per_sec` from `{batch}_features.csv` (already cached). Both are good approximations of "voiced time"; the transcript-based one is more accurate.
- `speech_ratio = speaking_time_s / total_duration_s`.

Caches to `checkpoints_honest_eval/{batch}_durations.csv` so the audio scan only happens once. Plots the distribution of `speaking_time_s` per batch, overlaid by class, with vertical lines at common thresholds (15 s / 25 s / 40 s)."""

lines13 = [
    "import soundfile as _sf",
    "import json as _json",
    "",
    "def _audio_dur(path):",
    "    info = _sf.info(str(path))",
    "    return info.frames / info.samplerate",
    "",
    "def _load_transcripts(batch):",
    "    p = NB_DIR / f'{batch}_transcripts.json'",
    "    if not p.exists():",
    "        return {}",
    "    with open(p, 'r', encoding='utf-8') as f:",
    "        t = _json.load(f)",
    "    if isinstance(t, dict):",
    "        return t",
    "    if isinstance(t, list):",
    "        out = {}",
    "        for x in t:",
    "            for k in ('filename', 'file', 'name'):",
    "                if k in x:",
    "                    out[x[k]] = x",
    "                    break",
    "        return out",
    "    return {}",
    "",
    "def _speech_from_segments(entry):",
    "    segs = entry.get('segments') if isinstance(entry, dict) else None",
    "    if not segs:",
    "        return None",
    "    try:",
    "        return float(sum((s['end'] - s['start']) for s in segs))",
    "    except (KeyError, TypeError):",
    "        return None",
    "",
    "def compute_durations(batch, force=False):",
    "    cache = SAVE_DIR / f'{batch}_durations.csv'",
    "    if cache.exists() and not force:",
    "        return pd.read_csv(cache)",
    "    df = batches[batch][['filename','candidate_id','label_int']].copy()",
    "    transcripts = _load_transcripts(batch)",
    "    feats = pd.read_csv(NB_DIR / f'{batch}_features.csv').set_index('filename')",
    "    durs, speeches, sources = [], [], []",
    "    for fn, cid in zip(df['filename'], df['candidate_id']):",
    "        path = NB_DIR / batch / str(cid) / fn",
    "        try:",
    "            dur = _audio_dur(path)",
    "        except Exception:",
    "            dur = float('nan')",
    "        durs.append(dur)",
    "        # Speaking-time source preference: transcripts > derived from features",
    "        sp, src = None, 'none'",
    "        entry = transcripts.get(fn) or transcripts.get(str(fn))",
    "        if entry is not None:",
    "            sp = _speech_from_segments(entry)",
    "            if sp is not None:",
    "                src = 'transcript'",
    "        if sp is None and fn in feats.index:",
    "            n_w = feats.at[fn, 'n_words']    if 'n_words'       in feats.columns else None",
    "            wps = feats.at[fn, 'words_per_sec'] if 'words_per_sec' in feats.columns else None",
    "            if pd.notna(n_w) and pd.notna(wps) and wps > 0:",
    "                sp, src = float(n_w) / float(wps), 'derived_words_per_sec'",
    "        speeches.append(sp)",
    "        sources.append(src)",
    "    df['total_duration_s'] = durs",
    "    df['speaking_time_s']  = speeches",
    "    df['speech_ratio']     = df['speaking_time_s'] / df['total_duration_s']",
    "    df['source']           = sources",
    "    df.to_csv(cache, index=False)",
    "    return df",
    "",
    "print('Computing per-audio durations (cached on disk after first run)...')",
    "durations = {b: compute_durations(b) for b in BATCHES}",
    "for b, d in durations.items():",
    "    src_counts = d['source'].value_counts().to_dict()",
    "    miss_dur   = int(d['total_duration_s'].isna().sum())",
    "    miss_sp    = int(d['speaking_time_s'].isna().sum())",
    "    print(f'  {b}: n={len(d)}  source={src_counts}  missing_dur={miss_dur}  missing_speech={miss_sp}')",
    "",
    "# Compact summary table",
    "summary_rows = []",
    "for b, d in durations.items():",
    "    for cls, label in [(0, 'honest'), (1, 'cheat')]:",
    "        sub = d[d['label_int']==cls]['speaking_time_s'].dropna()",
    "        summary_rows.append({",
    "            'batch':   b,",
    "            'class':   label,",
    "            'n':       len(sub),",
    "            'min':     round(sub.min(), 1)  if len(sub) else None,",
    "            'p10':     round(sub.quantile(0.10), 1) if len(sub) else None,",
    "            'p25':     round(sub.quantile(0.25), 1) if len(sub) else None,",
    "            'median':  round(sub.median(), 1)      if len(sub) else None,",
    "            'p75':     round(sub.quantile(0.75), 1) if len(sub) else None,",
    "            'max':     round(sub.max(), 1)         if len(sub) else None,",
    "            'pct_under_25s': round((sub < 25).mean() * 100, 1) if len(sub) else None,",
    "            'pct_under_15s': round((sub < 15).mean() * 100, 1) if len(sub) else None,",
    "        })",
    "summary_df = pd.DataFrame(summary_rows)",
    "print()",
    "print('Speaking-time distribution per batch x class (seconds):')",
    "with pd.option_context('display.max_columns', None, 'display.width', 160):",
    "    print(summary_df.to_string(index=False))",
    "",
    "# Per-batch histograms",
    "if HAS_MPL:",
    "    fig, axes = plt.subplots(1, len(BATCHES), figsize=(5*len(BATCHES), 4), sharey=True)",
    "    if len(BATCHES) == 1:",
    "        axes = [axes]",
    "    bins = np.arange(0, 121, 5)",
    "    for ax, b in zip(axes, BATCHES):",
    "        d = durations[b]",
    "        h = d[d['label_int']==0]['speaking_time_s'].dropna()",
    "        c = d[d['label_int']==1]['speaking_time_s'].dropna()",
    "        ax.hist(h, bins=bins, alpha=0.55, label=f'honest n={len(h)}', color='C0')",
    "        ax.hist(c, bins=bins, alpha=0.55, label=f'cheat n={len(c)}',  color='C3')",
    "        for thr, style in [(15, ':'), (25, '--'), (40, '-.')]:",
    "            ax.axvline(thr, color='k', linestyle=style, linewidth=0.8, alpha=0.6)",
    "        ax.set_title(b)",
    "        ax.set_xlabel('speaking_time_s')",
    "        ax.legend(fontsize=8, loc='upper right')",
    "    axes[0].set_ylabel('# audios')",
    "    fig.suptitle('Speaking-time distribution by batch and class  (dotted=15s, dashed=25s, dashdot=40s)')",
    "    fig.tight_layout()",
    "    plt.show()",
    "else:",
    "    print('matplotlib not available; skipping histograms')",
]

# =====================================================================
# §14 — Min-speaking-time filter + filtered-training rotation re-run
# =====================================================================
md14 = """## 14. Min-speaking-time filter — does dropping short audios from training close the Rot B gap?

Set `MIN_SPEAKING_S` below. The cell:
1. Filters audios with `speaking_time_s < MIN_SPEAKING_S` out of the **training-eligible** batches only (i.e. always-train batch + CV target). Test batch is left intact — at deployment we still have to handle short audios; we only want to stop the model *learning* from acoustically-confounded ones.
2. Re-runs both rotations on the filtered batches.
3. Compares F1 / gap to the §3 baseline rotations.

Hypothesis being tested: the same short audios that confounded §11 (low-cognitive-load monologues that look scripted) are pulling the negative-class boundary in the wrong direction. If we filter them out of training, Rot B's whisper gap should shrink without losing much CV F1.

Things to watch:
- **Headline:** whisper_wp_xgb Rot B gap, baseline vs filtered.
- **Side effect:** training-set size reduction. If we drop >25% of audios2, the always-train batch shrinks materially and CV F1 may drop on its own. Keep an eye on `n_train_used`.
- **Sanity:** test-set F1 should not improve dramatically — if it jumps by more than +0.05, we have leakage (test audios accidentally getting filtered)."""

lines14 = [
    "MIN_SPEAKING_S = 25.0   # tweak this — re-run cell to see the effect",
    "",
    "if 'durations' not in dir():",
    "    raise RuntimeError('Run section 13 first (defines `durations`).')",
    "if 'run_rotation_base' not in dir():",
    "    raise RuntimeError('Run section 3 first (defines `run_rotation_base`).')",
    "",
    "# --- 1. Build filtered copy of `batches`. NaN speaking_time -> KEEP (don't punish unknowns) ---",
    "def _filter_batch(name, min_s):",
    "    d = durations[name][['filename','speaking_time_s']]",
    "    keep_files = set(d[(d['speaking_time_s'] >= min_s) | (d['speaking_time_s'].isna())]['filename'])",
    "    sub = batches[name][batches[name]['filename'].isin(keep_files)].reset_index(drop=True)",
    "    return sub",
    "",
    "filt_stats = []",
    "for b in BATCHES:",
    "    n_before = len(batches[b])",
    "    n_after  = len(_filter_batch(b, MIN_SPEAKING_S))",
    "    pos_b = int((batches[b]['label_int']==1).sum())",
    "    pos_a = int((_filter_batch(b, MIN_SPEAKING_S)['label_int']==1).sum())",
    "    filt_stats.append({",
    "        'batch':           b,",
    "        'n_before':        n_before,",
    "        'n_after':         n_after,",
    "        'dropped':         n_before - n_after,",
    "        'pct_dropped':     round((n_before - n_after) / max(n_before,1) * 100, 1),",
    "        'pos_before':      pos_b,",
    "        'pos_after':       pos_a,",
    "        'neg_dropped':     (n_before - pos_b) - (n_after - pos_a),",
    "        'pos_dropped':     pos_b - pos_a,",
    "    })",
    "print(f'Min speaking-time filter: keep audios with speaking_time_s >= {MIN_SPEAKING_S}')",
    "print(pd.DataFrame(filt_stats).to_string(index=False))",
    "",
    "# --- 2. Run rotation with filtered TRAINING + CV but UNFILTERED test ---",
    "def _run_rot_filtered(train_folders, cv_target, test_folder, min_s):",
    "    batches_orig = {b: batches[b] for b in batches}",
    "    try:",
    "        for b in train_folders:",
    "            batches[b] = _filter_batch(b, min_s)",
    "        # test_folder stays unfiltered",
    "        df_out, _ = run_rotation_base(train_folders, cv_target, test_folder)",
    "    finally:",
    "        for b in train_folders:",
    "            batches[b] = batches_orig[b]",
    "    return df_out",
    "",
    "print()",
    "print('Running Rotation A filtered (train=[a2,a4] filtered, test=a5 unfiltered) ...')",
    "df_A_f = _run_rot_filtered(['audios2','audios4'], 'audios4', 'audios5', MIN_SPEAKING_S)",
    "print('Running Rotation B filtered (train=[a2,a5] filtered, test=a4 unfiltered) ...')",
    "df_B_f = _run_rot_filtered(['audios2','audios5'], 'audios5', 'audios4', MIN_SPEAKING_S)",
    "df_A_f.to_csv(SAVE_DIR / f'rotation_A_minS{int(MIN_SPEAKING_S)}.csv', index=False)",
    "df_B_f.to_csv(SAVE_DIR / f'rotation_B_minS{int(MIN_SPEAKING_S)}.csv', index=False)",
    "",
    "# --- 3. Compare to baseline ---",
    "BASELINE_A = SAVE_DIR / 'rotation_A_baseline.csv' if (SAVE_DIR / 'rotation_A_baseline.csv').exists() else SAVE_DIR / 'rotation_A.csv'",
    "BASELINE_B = SAVE_DIR / 'rotation_B_baseline.csv' if (SAVE_DIR / 'rotation_B_baseline.csv').exists() else SAVE_DIR / 'rotation_B.csv'",
    "",
    "def _cmp(label, base_csv, filt_df):",
    "    base = pd.read_csv(base_csv)",
    "    base_f1 = base[base['strategy']=='F1'][['model','cv','te','gap']].rename(columns={'cv':'cv_base','te':'te_base','gap':'gap_base'})",
    "    filt_f1 = filt_df[filt_df['strategy']=='F1'][['model','cv','te','gap']].rename(columns={'cv':'cv_filt','te':'te_filt','gap':'gap_filt'})",
    "    m = base_f1.merge(filt_f1, on='model')",
    "    m['cv_delta']  = (m['cv_filt']  - m['cv_base']).round(3)",
    "    m['te_delta']  = (m['te_filt']  - m['te_base']).round(3)",
    "    m['gap_delta'] = (m['gap_filt'] - m['gap_base']).round(3)",
    "    print()",
    "    print('=' * 100)",
    "    print(f' {label}  (F1 strategy, baseline vs filtered)')",
    "    print('=' * 100)",
    "    with pd.option_context('display.max_columns', None, 'display.width', 220):",
    "        print(m.to_string(index=False))",
    "    return m",
    "",
    "A_cmp = _cmp(f'Rotation A  CV=a4 test=a5  min_speaking_s={MIN_SPEAKING_S}', BASELINE_A, df_A_f)",
    "B_cmp = _cmp(f'Rotation B  CV=a5 test=a4  min_speaking_s={MIN_SPEAKING_S}', BASELINE_B, df_B_f)",
    "",
    "# --- 4. Headline ---",
    "print()",
    "print('=' * 100)",
    "print(' HEADLINE: did Rot B whisper gap shrink under the speaking-time filter?')",
    "print('=' * 100)",
    "for model_name in ['whisper_wp_xgb','wavlm_whole_ft']:",
    "    if model_name not in B_cmp['model'].values: continue",
    "    r = B_cmp[B_cmp['model']==model_name].iloc[0]",
    "    print(f'  {model_name:18s}  Rot B gap   baseline {r[\"gap_base\"]:+.3f}  -> filtered {r[\"gap_filt\"]:+.3f}   delta {r[\"gap_delta\"]:+.3f}')",
    "    print(f'  {model_name:18s}  Rot B test  baseline {r[\"te_base\"]:+.3f}  -> filtered {r[\"te_filt\"]:+.3f}   delta {r[\"te_delta\"]:+.3f}')",
    "",
    "print()",
    "print('Reading guide:')",
    "print('  - gap_delta strongly negative (e.g. -0.05 or more) on whisper Rot B -> short-audio confound was real.')",
    "print('  - gap_delta close to 0 -> short audios were not the dominant noise source; reset MIN_SPEAKING_S=20 or 30.')",
    "print('  - cv_delta strongly negative -> filter removed too much; training set too small.')",
    "print('  - te_delta strongly positive on Rot A is benign (test=a5 is unfiltered; nothing changed there mechanically).')",
]

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md13.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": [ln + '\n' for ln in '\n'.join(lines13).split('\n')]},
    {"cell_type": "markdown", "metadata": {}, "source": md14.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": [ln + '\n' for ln in '\n'.join(lines14).split('\n')]},
]
nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')

# AST-validate the two new code cells
ast.parse(''.join(nb['cells'][-3]['source']))
ast.parse(''.join(nb['cells'][-1]['source']))
print('APPENDED §13 + §14 — total cells:', len(nb['cells']), '— both code cells AST-valid')
