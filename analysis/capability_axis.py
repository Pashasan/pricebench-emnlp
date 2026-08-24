"""
External capability axis vs decisiveness.

Merges capability_scores.csv (externally sourced capability scores:
Artificial Analysis Intelligence Index v4.1, LMArena text Elo, model-card MMLU)
with outputs/data/personality.csv and outputs/data/triage.csv, then for each
capability column with >= 10 matched models reports:

  (a) ranks of the 5 position-locked models on the axis (1 = lowest score)
  (b) Pearson + Spearman of decisiveness (r2_log) and |b_logp| vs score,
      pooled (open + proprietary) and open-only, among engaged matched models
  (c) WTP (b_review / |b_logp| x $250) vs score for the 20 engaged models
      with both coefficients p < 0.001

Outputs:
  outputs/robustness/capability_axis_results.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_CSV = os.path.join(BASE, 'capability_scores.csv')
PERS_CSV = os.path.join(BASE, 'outputs', 'data', 'personality.csv')
TRIAGE_CSV = os.path.join(BASE, 'outputs', 'data', 'triage.csv')
OUT_DIR = os.path.join(BASE, 'outputs', 'robustness')
OUT_CSV = os.path.join(OUT_DIR, 'capability_axis_results.csv')

REF_PRICE = 250.0

PROPRIETARY = {'GPT-4.1 Nano', 'GPT-4.1 Mini', 'GPT-5.4 Nano', 'GPT-5.4 Mini',
               'GPT-5.4', 'Claude Haiku 4.5'}

LOCKED = ['Llama3.2 1B', 'Mistral 7B', 'Qwen3 0.6B', 'Llama3 8B', 'Llama3.1 8B']

AXES = [('score_aa', 'AA Intelligence Index v4.1'),
        ('score_arena', 'LMArena text Elo'),
        ('score_mmlu', 'Model-card MMLU')]


def corr(x, y):
    """Pearson and Spearman with p-values; returns 4-tuple of floats."""
    if len(x) < 3:
        return (np.nan,) * 4
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return pr, pp, sr, sp


scores = pd.read_csv(SCORES_CSV)
pers = pd.read_csv(PERS_CSV)
triage = pd.read_csv(TRIAGE_CSV)[['model', 'verdict']]

df = scores.merge(pers.drop(columns=['verdict'], errors='ignore'),
                  on='model', how='left')
df = df.merge(triage, on='model', how='left')
df = df[df['model'] != 'Qwen3 1.7B'].copy()

df['open'] = ~df['model'].isin(PROPRIETARY)
df['abs_blogp'] = df['b_logp'].abs()
df['wtp250'] = df['b_d_review_score'] / df['abs_blogp'] * REF_PRICE
df['engaged'] = df['verdict'] == 'engaged'
df['sig_price'] = df['b_logp_p'] < 0.05
df['wtp_ok'] = (df['engaged'] & (df['b_logp_p'] < 0.001)
                & (df['b_d_review_score_p'] < 0.001))

print('n models          :', len(df))
print('n engaged         :', int(df['engaged'].sum()))
print('n locked          :', len(LOCKED))
print('n WTP subset      :', int(df['wtp_ok'].sum()))
print()

results = []


def record(axis, analysis, subset, n, pr=np.nan, pp=np.nan, sr=np.nan,
           sp=np.nan, note=''):
    results.append(dict(axis=axis, analysis=analysis, subset=subset, n=n,
                        pearson_r=pr, pearson_p=pp,
                        spearman_rho=sr, spearman_p=sp, note=note))


for col, label in AXES:
    have = df[df[col].notna()].copy()
    n_matched = len(have)
    missing = df.loc[df[col].isna(), 'model'].tolist()
    print('=' * 78)
    print(f'{label}  ({col}): {n_matched}/28 models matched')
    print('  missing:', ', '.join(missing) if missing else 'none')
    record(col, 'coverage', 'all', n_matched,
           note='missing: ' + ('; '.join(missing) if missing else 'none'))

    # (a) locked-model ranks (1 = lowest capability among matched)
    have['rank'] = have[col].rank(method='min', ascending=True)
    print(f'  locked-model ranks (1 = lowest of {n_matched}):')
    for m in LOCKED:
        r = have.loc[have['model'] == m]
        if len(r):
            rk = int(r['rank'].iloc[0])
            sc = r[col].iloc[0]
            print(f'    {m:<14s} score={sc:>8.2f}  rank={rk}/{n_matched}')
            record(col, 'locked_rank', m, n_matched, pr=sc, sr=rk,
                   note=f'rank {rk} of {n_matched} (1 = lowest)')
        else:
            print(f'    {m:<14s} not matched on this axis')
            record(col, 'locked_rank', m, n_matched, note='not matched')

    if n_matched < 10:
        print('  < 10 matched models: correlations skipped for this axis')
        continue

    # (b) capability vs decisiveness among engaged matched models
    for sub_label, mask in [
            ('engaged pooled', have['engaged']),
            ('engaged open-only', have['engaged'] & have['open']),
            ('engaged pooled, sig price', have['engaged'] & have['sig_price']),
            ('engaged open-only, sig price',
             have['engaged'] & have['open'] & have['sig_price'])]:
        sub = have[mask]
        for yvar, yname in [('r2_log', 'r2_log'), ('abs_blogp', '|b_logp|')]:
            pr, pp, sr, sp = corr(sub[col].values, sub[yvar].values)
            print(f'  {yname:<9s} vs score, {sub_label:<28s} n={len(sub):>2d} '
                  f'Pearson r={pr:+.2f} (p={pp:.3f})  '
                  f'Spearman rho={sr:+.2f} (p={sp:.3f})')
            record(col, f'{yname} vs score', sub_label, len(sub), pr, pp,
                   sr, sp)

    # proprietary placement
    prop = have[have['engaged'] & ~have['open']].sort_values(col)
    if len(prop):
        print('  proprietary placement (score, r2_log, |b_logp|):')
        for _, r in prop.iterrows():
            print(f'    {r["model"]:<18s} {r[col]:>7.1f}  '
                  f'r2={r["r2_log"]:.3f}  |b|={r["abs_blogp"]:.2f}')

    # (c) WTP vs score
    for sub_label, mask in [
            ('WTP subset pooled', have['wtp_ok']),
            ('WTP subset open-only', have['wtp_ok'] & have['open'])]:
        sub = have[mask]
        pr, pp, sr, sp = corr(sub[col].values, sub['wtp250'].values)
        print(f'  WTP($250) vs score, {sub_label:<26s} n={len(sub):>2d} '
              f'Pearson r={pr:+.2f} (p={pp:.3f})  '
              f'Spearman rho={sr:+.2f} (p={sp:.3f})')
        record(col, 'WTP vs score', sub_label, len(sub), pr, pp, sr, sp)

    print()

# Sensitivity: AA axis excluding the GPT-5.4 family, whose AA scores are for
# the xhigh reasoning config while the conjoint used 16-token completions.
col = 'score_aa'
have = df[df[col].notna() & ~df['model'].isin(
    ['GPT-5.4', 'GPT-5.4 Mini', 'GPT-5.4 Nano'])].copy()
have['abs_blogp'] = have['b_logp'].abs()
print('=' * 78)
print('Sensitivity: AA axis excluding GPT-5.4 family (xhigh config mismatch)')
for yvar, yname in [('r2_log', 'r2_log'), ('abs_blogp', '|b_logp|')]:
    sub = have[have['engaged']]
    pr, pp, sr, sp = corr(sub[col].values, sub[yvar].values)
    print(f'  {yname:<9s} vs score, engaged pooled excl GPT-5.4  n={len(sub)} '
          f'Pearson r={pr:+.2f} (p={pp:.3f})  '
          f'Spearman rho={sr:+.2f} (p={sp:.3f})')
    record(col, f'{yname} vs score', 'engaged pooled, excl GPT-5.4 family',
           len(sub), pr, pp, sr, sp,
           note='sensitivity: AA scores for GPT-5.4 family are xhigh config')

out = pd.DataFrame(results)
os.makedirs(OUT_DIR, exist_ok=True)
out.to_csv(OUT_CSV, index=False, encoding='utf-8')
print(f'\nSaved {OUT_CSV}')
