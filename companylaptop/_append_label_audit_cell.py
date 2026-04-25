"""Append §11 'a5 label-suspicion audit' to honest_eval_and_improve.ipynb.
Requires §10 to have been run (uses _train_on_audios2)."""
import json
from pathlib import Path

NB = Path(__file__).with_name('honest_eval_and_improve.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))

md_src = """## 11. audios5 label-suspicion audit

§10 confirmed: training on audios2 alone gives AUC 0.92 on a4 and 0.94 on a5. Adding a5 to training is what breaks a4 transfer. This cell ranks individual a5 audios by how strongly an a2-only-trained model disagrees with their label.

The top of each disagreement list (positives the model thinks are honest, negatives the model thinks are cheating) is the shortlist worth a 30-second re-listen. Two acoustic models trained independently on a2 (whisper and fine-tuned WavLM) — agreement between their "this label looks wrong" signal is much stronger evidence than either alone.

Output: per-class shortlists (top-15 each) and a combined "both models flag this" list. Filenames are printed so you can pull the audio file directly."""

code_src = """if 'batches' not in dir() or 'BASE_REGISTRY' not in dir():
    raise RuntimeError(\"Run sections 1-3 first.\")
if '_train_on_audios2' not in dir():
    raise RuntimeError(\"Run section 10 first (defines _train_on_audios2).\")

def _score_audios5(name):
    X_fn, factory = BASE_REGISTRY[name]
    clf, sc, _ = _train_on_audios2(name, X_fn, factory, balanced=False)
    df = batches['audios5'].copy()
    p = clf.predict_proba(sc.transform(X_fn(df)))[:,1]
    df = df[['filename','candidate_id','label_int']].copy()
    df['proba'] = p
    return df

w_df = _score_audios5('whisper_wp_xgb')
print(f'audios5 scored by whisper trained on audios2 only — n={len(w_df)}')

if 'wavlm_whole_ft' in BASE_REGISTRY:
    v_df = _score_audios5('wavlm_whole_ft')
    print(f'audios5 scored by wavlm_whole_ft trained on audios2 only — n={len(v_df)}')
else:
    v_df = None

# ---- per-model shortlists ----
def _shortlist(df, model_name, k=15):
    pos = df[df['label_int']==1].sort_values('proba', ascending=True).head(k)   # labeled +, scored low
    neg = df[df['label_int']==0].sort_values('proba', ascending=False).head(k)  # labeled -, scored high
    print(f'\\n--- {model_name}: top-{k} POSITIVES the model thinks are HONEST ---')
    print(pos[['filename','candidate_id','proba']].to_string(index=False))
    print(f'\\n--- {model_name}: top-{k} NEGATIVES the model thinks are CHEATING ---')
    print(neg[['filename','candidate_id','proba']].to_string(index=False))

_shortlist(w_df, 'whisper_wp_xgb')
if v_df is not None:
    _shortlist(v_df, 'wavlm_whole_ft')

# ---- joint shortlist: BOTH models flag the same audio ----
if v_df is not None:
    # Suspicious-positive score = how confident BOTH models are that label=1 audio is actually negative
    pos_a = w_df[w_df['label_int']==1].set_index('filename')['proba'].rename('whisper')
    pos_b = v_df[v_df['label_int']==1].set_index('filename')['proba'].rename('wavlm_ft')
    pos_join = pd.concat([pos_a, pos_b], axis=1).dropna()
    pos_join['avg_prob'] = (pos_join['whisper'] + pos_join['wavlm_ft']) / 2
    pos_join['min_prob'] = pos_join[['whisper','wavlm_ft']].min(axis=1)
    pos_join = pos_join.merge(batches['audios5'][['filename','candidate_id']],
                               left_index=True, right_on='filename')
    print('\\n' + '='*82)
    print(' JOINT: a5 audios LABELED CHEATING but BOTH models score low (likely label noise)')
    print(' Sorted by min(prob) — strongest joint disagreement first')
    print('='*82)
    print(pos_join.sort_values('min_prob').head(20)[
        ['filename','candidate_id','whisper','wavlm_ft','avg_prob']
    ].to_string(index=False))

    neg_a = w_df[w_df['label_int']==0].set_index('filename')['proba'].rename('whisper')
    neg_b = v_df[v_df['label_int']==0].set_index('filename')['proba'].rename('wavlm_ft')
    neg_join = pd.concat([neg_a, neg_b], axis=1).dropna()
    neg_join['avg_prob'] = (neg_join['whisper'] + neg_join['wavlm_ft']) / 2
    neg_join['max_prob'] = neg_join[['whisper','wavlm_ft']].max(axis=1)
    neg_join = neg_join.merge(batches['audios5'][['filename','candidate_id']],
                               left_index=True, right_on='filename')
    print('\\n' + '='*82)
    print(' JOINT: a5 audios LABELED HONEST but BOTH models score high (likely label noise)')
    print(' Sorted by max(prob) descending — strongest joint disagreement first')
    print('='*82)
    print(neg_join.sort_values('max_prob', ascending=False).head(20)[
        ['filename','candidate_id','whisper','wavlm_ft','avg_prob']
    ].to_string(index=False))

    # ---- per-candidate consistency ----
    # If a candidate's three audios disagree wildly with their labels, they're worth special review
    print('\\n' + '='*82)
    print(' Per-candidate joint-disagreement (mean over their 3 audios)')
    print('='*82)
    df_all = w_df.merge(v_df[['filename','proba']].rename(columns={'proba':'proba_wavlm_ft'}),
                        on='filename')
    df_all = df_all.rename(columns={'proba':'proba_whisper'})
    df_all['expected'] = df_all['label_int']  # 1 means model SHOULD score high
    df_all['disagree_w'] = (df_all['expected'] - df_all['proba_whisper']).abs()
    df_all['disagree_v'] = (df_all['expected'] - df_all['proba_wavlm_ft']).abs()
    df_all['disagree_joint'] = (df_all['disagree_w'] + df_all['disagree_v']) / 2
    cand = df_all.groupby('candidate_id').agg(
        n=('filename','size'),
        n_pos=('label_int','sum'),
        mean_disagree=('disagree_joint','mean'),
        max_disagree =('disagree_joint','max'),
    ).sort_values('mean_disagree', ascending=False).head(15)
    print(cand.to_string())

print('\\nReading guide:')
print('  - JOINT lists are the strongest signal: both models trained without ever')
print('    seeing audios5 disagree with the label.  Shortlist for re-listen.')
print('  - If 5+ of the 43 a5 positives appear in the joint POSITIVE-suspect list')
print('    with both probs <0.3, that is enough label noise to fully explain Rot B.')
print('  - Per-candidate disagreement: a candidate whose 3 audios all disagree')
print('    consistently is more likely a true label issue than a one-off outlier.')
"""

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": md_src.splitlines(keepends=True)},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
     "source": code_src.splitlines(keepends=True)},
]
nb['cells'].extend(new_cells)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('APPENDED:', NB, '— total cells now:', len(nb['cells']))
