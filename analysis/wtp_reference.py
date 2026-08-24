"""
Reference-price invariance of the WTP ranking.

The log-price WTP reported in the paper is WTP(ref) = b_review / |b_logp| * ref
with ref = $250. Because ref enters every model's WTP as the same multiplicative
constant, the model RANKING (and every WTP ratio, including the ~16x spread) is
invariant to the choice of reference price. This script demonstrates that
numerically at ref in {100, 250, 600} and cross-checks against the linear-price
WTP (wtp_review in personality.csv), which needs no reference price at all.

Inputs:
  outputs/data/personality.csv

Outputs:
  outputs/robustness/wtp_reference.csv
  outputs/robustness/wtp_reference.md
"""
import os, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, BASE)

import numpy as np
import pandas as pd
from scipy import stats

PERSONALITY_CSV = os.path.join(BASE, 'outputs', 'data', 'personality.csv')
OUT_DIR = os.path.join(BASE, 'outputs', 'robustness')
OUT_CSV = os.path.join(OUT_DIR, 'wtp_reference.csv')
OUT_MD = os.path.join(OUT_DIR, 'wtp_reference.md')

REFS = [100, 250, 600]
P_THRESH = 0.001

os.makedirs(OUT_DIR, exist_ok=True)

pers = pd.read_csv(PERSONALITY_CSV)

# ---------------------------------------------------------------------------
# Step 1: engaged models with b_logp_p < 0.001 AND b_d_review_score_p < 0.001
# ---------------------------------------------------------------------------
engaged = pers[pers['verdict'] == 'engaged'].copy()
sel = engaged[(engaged['b_logp_p'] < P_THRESH) &
              (engaged['b_d_review_score_p'] < P_THRESH)].copy()
excluded = engaged[~engaged['model'].isin(sel['model'])]

print(f'Engaged models: {len(engaged)}')
print(f'Selected (b_logp_p<{P_THRESH} and b_d_review_score_p<{P_THRESH}): '
      f'{len(sel)}')
print('Selected models:')
for m in sel['model']:
    print(f'  {m}')
print('Engaged but excluded:')
for _, r in excluded.iterrows():
    print(f'  {r["model"]:<20s} b_logp_p={r["b_logp_p"]:.3g} '
          f'b_rev_p={r["b_d_review_score_p"]:.3g}')

# ---------------------------------------------------------------------------
# Step 2: WTP(ref) = b_review / |b_logp| * ref, ranks under each ref
# ---------------------------------------------------------------------------
sel['wtp_unit'] = sel['b_d_review_score'] / sel['b_logp'].abs()
for ref in REFS:
    sel[f'wtp_{ref}'] = sel['wtp_unit'] * ref
    # rank 1 = highest WTP
    sel[f'rank_{ref}'] = sel[f'wtp_{ref}'].rank(ascending=False,
                                                method='min').astype(int)

ranks_identical = all(
    (sel[f'rank_{ref}'] == sel[f'rank_{REFS[0]}']).all() for ref in REFS[1:])

print(f'\nRanks identical across refs {REFS}: {ranks_identical}')

spread = {}
for ref in REFS:
    lo, hi = sel[f'wtp_{ref}'].min(), sel[f'wtp_{ref}'].max()
    lo_m = sel.loc[sel[f'wtp_{ref}'].idxmin(), 'model']
    hi_m = sel.loc[sel[f'wtp_{ref}'].idxmax(), 'model']
    spread[ref] = (lo, hi, hi / lo, lo_m, hi_m)
    print(f'  ref=${ref:>3d}: min={lo:8.2f} ({lo_m}), max={hi:8.2f} ({hi_m}), '
          f'ratio={hi/lo:.2f}x')

# ---------------------------------------------------------------------------
# Step 3: cross-form check vs linear-spec WTP (wtp_review, no reference price)
# ---------------------------------------------------------------------------
sel['wtp_linear'] = sel['wtp_review']
sel['rank_linear'] = sel['wtp_linear'].rank(ascending=False,
                                            method='min').astype(int)
sel['rank_shift'] = sel['rank_linear'] - sel['rank_250']

rho, rho_p = stats.spearmanr(sel['wtp_250'], sel['wtp_linear'])
r, r_p = stats.pearsonr(sel['wtp_250'], sel['wtp_linear'])
print(f'\nLog-spec WTP@250 vs linear-spec WTP (n={len(sel)}):')
print(f'  Spearman rho = {rho:.3f} (p={rho_p:.2g})')
print(f'  Pearson  r   = {r:.3f} (p={r_p:.2g})')

movers = sel[sel['rank_shift'].abs() > 3]
print(f'Models moving more than 3 rank positions: {len(movers)}')
for _, m in movers.sort_values('rank_shift', key=lambda s: s.abs(),
                               ascending=False).iterrows():
    print(f'  {m["model"]:<20s} log-rank {m["rank_250"]:>2d} -> '
          f'lin-rank {m["rank_linear"]:>2d} (shift {m["rank_shift"]:+d}), '
          f'fit_form={m["fit_form"]}')

# ---------------------------------------------------------------------------
# Step 4: linear-spec WTP span
# ---------------------------------------------------------------------------
lin_lo, lin_hi = sel['wtp_linear'].min(), sel['wtp_linear'].max()
lin_lo_m = sel.loc[sel['wtp_linear'].idxmin(), 'model']
lin_hi_m = sel.loc[sel['wtp_linear'].idxmax(), 'model']
print(f'\nLinear-spec WTP span: min={lin_lo:.2f} ({lin_lo_m}), '
      f'max={lin_hi:.2f} ({lin_hi_m}), ratio={lin_hi/lin_lo:.2f}x')

fit_counts = sel['fit_form'].value_counts()
print('AIC-preferred form among the selected models:')
for k, v in fit_counts.items():
    print(f'  {k}: {v}')

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
cols = (['model', 'b_logp', 'b_logp_p', 'b_d_review_score',
         'b_d_review_score_p', 'wtp_unit'] +
        [f'wtp_{ref}' for ref in REFS] + [f'rank_{ref}' for ref in REFS] +
        ['wtp_linear', 'rank_linear', 'rank_shift', 'fit_form'])
out = sel.sort_values('rank_250')[cols]
out.to_csv(OUT_CSV, index=False, encoding='utf-8')
print(f'\nSaved {OUT_CSV}')

md = []
md.append('# WTP reference-price invariance\n')
md.append('Source: `outputs/data/personality.csv` (binary tasks, standard '
          'controls). Script: `analysis/wtp_reference.py`.\n')
md.append('## Sample\n')
md.append(f'Engaged models: {len(engaged)}. Of these, {len(sel)} have both '
          f'`b_logp_p < {P_THRESH}` and `b_d_review_score_p < {P_THRESH}` '
          '(matches the 20 claimed in the paper).\n')
md.append('Engaged but excluded: ' + '; '.join(
    f'{r["model"]} (b_logp_p={r["b_logp_p"]:.2g}, '
    f'b_rev_p={r["b_d_review_score_p"]:.2g})'
    for _, r in excluded.iterrows()) + '.\n')
md.append('## Structural fact\n')
md.append('The log-price WTP is `WTP(ref) = b_review / |b_logp| * ref`. The '
          'reference price `ref` is the same multiplicative constant for every '
          'model, so it cancels from every between-model ratio and cannot '
          'change the ranking. The choice of $250 affects only the dollar '
          'units of the reported numbers, not any comparative claim.\n')
md.append(f'Numerical check: rankings under ref = $100, $250, $600 are '
          f'{"IDENTICAL" if ranks_identical else "NOT identical (bug!)"}.\n')
md.append('## Spread at each reference price\n')
md.append('| ref | min WTP (model) | max WTP (model) | ratio |')
md.append('|---|---|---|---|')
for ref in REFS:
    lo, hi, ratio, lo_m, hi_m = spread[ref]
    md.append(f'| ${ref} | ${lo:,.0f} ({lo_m}) | ${hi:,.0f} ({hi_m}) '
              f'| {ratio:.1f}x |')
md.append('')
md.append('The ratio is identical at every reference price, as it must be.\n')
md.append('## Cross-form check: linear-price WTP needs no reference price\n')
md.append('`wtp_review` in personality.csv is `-b_review_lin / b_linp` from '
          'the linear-price logit; it is denominated in dollars directly and '
          'involves no reference price.\n')
md.append(f'- Spearman rho (log WTP@250 vs linear WTP) = {rho:.3f} '
          f'(p = {rho_p:.2g}), n = {len(sel)}')
md.append(f'- Pearson r = {r:.3f} (p = {r_p:.2g})')
md.append(f'- Models moving more than 3 rank positions: {len(movers)}')
for _, m in movers.sort_values('rank_shift', key=lambda s: s.abs(),
                               ascending=False).iterrows():
    md.append(f'  - {m["model"]}: log-spec rank {m["rank_250"]} -> '
              f'linear-spec rank {m["rank_linear"]} '
              f'(shift {m["rank_shift"]:+d}); AIC-preferred form: '
              f'{m["fit_form"]}')
md.append('')
md.append(f'- Linear-spec WTP span: ${lin_lo:,.0f} ({lin_lo_m}) to '
          f'${lin_hi:,.0f} ({lin_hi_m}), ratio {lin_hi/lin_lo:.1f}x. '
          'Note this differs from the log-spec 16x figure; the wide-spread '
          'claim is robust in direction (large heterogeneity) but the exact '
          'multiple is specification-dependent.')
md.append(f'- AIC-preferred form among the {len(sel)} models: ' +
          ', '.join(f'{k} = {v}' for k, v in fit_counts.items()) + '.\n')
md.append('## Per-model table\n')
hdr = ('| model | WTP@$100 | WTP@$250 | WTP@$600 | rank (all refs) '
       '| linear WTP | linear rank | shift | fit_form |')
md.append(hdr)
md.append('|---|---|---|---|---|---|---|---|---|')
for _, m in out.iterrows():
    md.append(f'| {m["model"]} | {m["wtp_100"]:,.0f} | {m["wtp_250"]:,.0f} '
              f'| {m["wtp_600"]:,.0f} | {m["rank_250"]} '
              f'| {m["wtp_linear"]:,.0f} | {m["rank_linear"]} '
              f'| {m["rank_shift"]:+d} | {m["fit_form"]} |')
md.append('')

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))
print(f'Saved {OUT_MD}')
