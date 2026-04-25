"""Append §10 'Data-asymmetry diagnostic' to honest_eval_and_improve.ipynb.
One-shot — safe to delete after running.
"""
import json
from pathlib import Path

NB = Path(__file__).with_name('honest_eval_and_improve.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

md_src = """## 10. Data-asymmetry diagnostic — audios2-only training

Goal: discriminate between three hypotheses for why Rotation B (CV=audios5, test=audios4) shows a +0.13 to +0.19 F1 gap on whisper-driven fusions, while Rotation A (CV=audios4, test=audios5) shows ~0 gap.

The three hypotheses:
1. **a5 contaminates training** (label noise or batch-shortcut memorisation): whisper learns from a5 a rule that doesn't fit a4.
2. **a4 cheaters are intrinsically subtler**: whisper trained without ever seeing a4 still cannot find them.
3. **a5 is a narrower slice of cheating styles**: whisper trained on a2+a5 only sees one mode, but a4 has both modes.

Method: train whisper (and fine-tuned WavLM if available) on **audios2 alone** (147 positives / 72 negatives, ~67% positive — the inverted prior is real and noted), then score audios4 and audios5 separately. Use prior-invariant metrics (AUC, average precision, best-F1-over-threshold-sweep) so the audios2 67% training prior does not confound the comparison.

Reading the result:
- If `AUC(a4) ≈ AUC(a5)` and both are reasonable (>0.75), a5 is fine on its own — the Rot B gap comes from *training with a5*, pointing to label noise or batch-shortcut memorisation. → hypothesis 1.
- If `AUC(a4) << AUC(a5)`, a4 cheaters are harder to find from a2-only features. → hypothesis 2 or 3 (the balanced re-run distinguishes them from a prior issue).
- The score-distribution table below shows per-batch class separation. If `separation(a4) << separation(a5)`, whisper's features carry less signal on a4."""

code_src = """from sklearn.metrics import roc_auc_score, average_precision_score

if 'batches' not in dir() or 'BASE_REGISTRY' not in dir():
    raise RuntimeError(\"Run sections 1-3 first so 'batches' and 'BASE_REGISTRY' are in scope.\")

# Sanity print the actual audios2 distribution (user flagged 147+/~70-).
_y2 = batches['audios2']['label_int'].values
print(f'audios2 training prior: {int((_y2==1).sum())}+ / {int((_y2==0).sum())}- '
      f'({(_y2==1).mean()*100:.1f}% positive)  — note: deployment prior is ~17% positive')

def _best_f1_sweep(p, y, grid=np.arange(0.05, 0.96, 0.01)):
    bf, bt = -1.0, 0.5
    for thr in grid:
        f = f1_score(y, (p >= thr).astype(int), zero_division=0)
        if f > bf: bf, bt = f, float(thr)
    return bt, bf

def _train_on_audios2(name, X_fn, factory, balanced=False, seed=42):
    df_tr = batches['audios2']
    if balanced:
        n_neg = int((df_tr['label_int']==0).sum())
        n_pos_target = n_neg  # match the smaller class so 1:1
        pos = df_tr[df_tr['label_int']==1].sample(n=n_pos_target, random_state=seed)
        neg = df_tr[df_tr['label_int']==0]
        df_tr = pd.concat([pos, neg], ignore_index=True)
    Xtr = X_fn(df_tr); ytr = df_tr['label_int'].values
    sc = StandardScaler().fit(Xtr)
    clf = factory(seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf, sc, ytr

def diagnose(name, X_fn, factory, balanced=False):
    clf, sc, ytr = _train_on_audios2(name, X_fn, factory, balanced=balanced)
    tag = 'BALANCED audios2' if balanced else 'audios2 (full, SPW_DEPLOY)'
    print(f'\\n{\"=\"*82}')
    print(f' {name}   trained on {tag}  ({int((ytr==1).sum())}+/{int((ytr==0).sum())}-)')
    print('='*82)
    rows = []
    for tgt in ['audios2','audios4','audios5']:
        df_e = batches[tgt]
        Xe   = X_fn(df_e); ye = df_e['label_int'].values
        p    = clf.predict_proba(sc.transform(Xe))[:,1]
        auc  = roc_auc_score(ye, p) if len(set(ye.tolist())) > 1 else float('nan')
        ap   = average_precision_score(ye, p) if len(set(ye.tolist())) > 1 else float('nan')
        thr, f1 = _best_f1_sweep(p, ye)
        rows.append({
            'eval_batch':    tgt,
            'n':             len(ye),
            'pos':           int((ye==1).sum()),
            'pos_rate':      round((ye==1).mean(), 3),
            'AUC':           round(auc, 3),
            'AP':            round(ap, 3),
            'best_thr':      round(thr, 2),
            'best_F1':       round(f1, 3),
            'pos_score_med': round(np.median(p[ye==1]), 3) if (ye==1).sum() else None,
            'neg_score_med': round(np.median(p[ye==0]), 3) if (ye==0).sum() else None,
        })
    df = pd.DataFrame(rows)
    df['separation'] = (df['pos_score_med'] - df['neg_score_med']).round(3)
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(df.to_string(index=False))
    return df

# ---- Run for whisper (always present) and fine-tuned WavLM (if loaded) ----
print('\\n' + '#'*82)
print('# PASS 1: train on FULL audios2 (147+/72-, the 67% positive training prior)')
print('# Uses SPW_DEPLOY=4.88 to calibrate to deployment 17% prior.')
print('#'*82)
res_w_full = diagnose('whisper_wp_xgb',
                       BASE_REGISTRY['whisper_wp_xgb'][0],
                       BASE_REGISTRY['whisper_wp_xgb'][1],
                       balanced=False)
if 'wavlm_whole_ft' in BASE_REGISTRY:
    res_v_full = diagnose('wavlm_whole_ft',
                          BASE_REGISTRY['wavlm_whole_ft'][0],
                          BASE_REGISTRY['wavlm_whole_ft'][1],
                          balanced=False)

print('\\n' + '#'*82)
print('# PASS 2: BALANCED audios2 (down-sample positives to 72 to remove the 67% prior)')
print('# Removes any prior-shift confound from the audios2-only training.')
print('#'*82)
res_w_bal  = diagnose('whisper_wp_xgb',
                       BASE_REGISTRY['whisper_wp_xgb'][0],
                       BASE_REGISTRY['whisper_wp_xgb'][1],
                       balanced=True)
if 'wavlm_whole_ft' in BASE_REGISTRY:
    res_v_bal = diagnose('wavlm_whole_ft',
                         BASE_REGISTRY['wavlm_whole_ft'][0],
                         BASE_REGISTRY['wavlm_whole_ft'][1],
                         balanced=True)

# ---- Compact summary that maps directly onto the three hypotheses ----
print('\\n' + '='*82)
print(' SUMMARY — does whisper-trained-on-a2-alone find a4 cheaters as well as a5?')
print('='*82)
def _row(label, df):
    a4 = df[df['eval_batch']=='audios4'].iloc[0]
    a5 = df[df['eval_batch']=='audios5'].iloc[0]
    return {
        'pass':            label,
        'AUC_a4':          a4['AUC'],
        'AUC_a5':          a5['AUC'],
        'dAUC(a4-a5)':     round(a4['AUC'] - a5['AUC'], 3),
        'AP_a4':           a4['AP'],
        'AP_a5':           a5['AP'],
        'sep_a4':          a4['separation'],
        'sep_a5':          a5['separation'],
        'bestF1_a4':       a4['best_F1'],
        'bestF1_a5':       a5['best_F1'],
    }
summary_rows = [_row('whisper · full a2',     res_w_full),
                _row('whisper · balanced a2', res_w_bal)]
if 'wavlm_whole_ft' in BASE_REGISTRY:
    summary_rows += [_row('wavlm_ft · full a2',     res_v_full),
                     _row('wavlm_ft · balanced a2', res_v_bal)]
summary_df = pd.DataFrame(summary_rows)
with pd.option_context('display.max_columns', None, 'display.width', 160):
    print(summary_df.to_string(index=False))

print('\\nDecision rules:')
print('  (1) dAUC(a4-a5) ≈ 0  AND both AUCs > 0.75   ->  a5 is fine on its own.')
print('      The Rot B gap comes from TRAINING with a5 -> label noise or batch')
print('      memorisation.  Action: spot-check a5 positives for label correctness.')
print('  (2) dAUC(a4-a5) << 0  (a4 much harder)      ->  a4 cheating is subtler.')
print('      Action: a4 must always be in training; never CV with a5 alone.')
print('  (3) AUC similar but BOTH low (< 0.65)       ->  audios2 features do not')
print('      generalise off-batch at all -> per-batch standardisation or domain')
print('      adaptation is the next move, not relabelling.')
"""

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md_src.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": code_src.splitlines(keepends=True)},
]
nb['cells'].extend(new_cells)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('APPENDED:', NB, '— total cells now:', len(nb['cells']))
