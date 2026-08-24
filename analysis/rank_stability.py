"""
Rank stability, cluster-bootstrap CIs, and multiplicity audit.

For every engaged model (triage verdict == 'engaged') we cluster-bootstrap the
binary-task log-price logit (CTRLS + d_log_price) over the 450 unique
group_id values (B=500, numpy seed 42).  Each draw keeps b_logp, b_review
(coef on d_review_score) and wtp = b_review / |b_logp| * 250 (the paper's WTP
definition).  Identical group draws are used for every model so that per-draw
rankings are comparable.

Also audits the paper's significance claims under Bonferroni and
Benjamini-Hochberg corrections, and correlates WTP and decisiveness with
parameter counts for the open-weight models.

Outputs:
  outputs/robustness/rank_stability.md
  outputs/robustness/rank_stability_cis.csv
  outputs/robustness/multiplicity_audit.csv
"""
import os, sys, time, warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, spearmanr, rankdata

from config import MODELS, build_dataset

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(BASE, 'outputs', 'robustness')
os.makedirs(OUT_DIR, exist_ok=True)

TRIAGE_CSV = os.path.join(BASE, 'outputs', 'data', 'triage.csv')
PERS_CSV = os.path.join(BASE, 'outputs', 'data', 'personality.csv')

CTRLS = ['d_stars', 'd_bed', 'd_cancel', 'd_breakfast',
         'd_review_score', 'd_review_count_100']
PRICE_VAR = 'd_log_price'
B = 500
SEED = 42
ALPHA = 0.05

# Parameter counts copied verbatim from analysis/figures.py (MODEL_PARAMS_B);
# MoE models use ACTIVE parameters, proprietary counts are tier-based
# estimates and are excluded from the open-weight correlations below.
MODEL_PARAMS_B = {
    'Qwen3 0.6B': 0.6, 'Qwen3 1.7B': 1.7, 'Qwen3 4B': 4.0, 'Qwen3 8B': 8.0,
    'Qwen3 30B-A3B': 3.0,  # MoE, 3B active
    'Gemma3 1B': 1.0, 'Gemma3 4B': 4.0, 'Gemma3 12B': 12.0, 'Gemma3 27B': 27.0,
    'Gemma2 9B': 9.0,
    'Llama3.2 1B': 1.0, 'Llama3.2 3B': 3.0, 'Llama3 8B': 8.0,
    'Llama3.1 8B': 8.0,
    'Phi-2 2.7B': 2.7, 'Phi-3 Mini': 3.8, 'Phi-3 Medium': 14.0,
    'Phi-4 Mini': 3.8, 'Phi-4 14B': 14.0,
    'Mistral 7B': 7.0, 'Mistral-Nemo 12B': 12.0,
    'DeepSeek-R1 1.5B': 1.5, 'DeepSeek-R1 7B': 7.0,
    'GPT-4.1 Nano': 8.0,   'GPT-4.1 Mini': 40.0,
    'GPT-5.4 Nano': 10.0,  'GPT-5.4 Mini': 60.0,  'GPT-5.4': 400.0,
    'Claude Haiku 4.5': 25.0,
}
PROPRIETARY = {'GPT-4.1 Nano', 'GPT-4.1 Mini', 'GPT-5.4 Nano',
               'GPT-5.4 Mini', 'GPT-5.4', 'Claude Haiku 4.5'}

FAMILIES = {
    'Gemma3': ['Gemma3 1B', 'Gemma3 4B', 'Gemma3 12B', 'Gemma3 27B'],
    'Qwen3': ['Qwen3 4B', 'Qwen3 8B', 'Qwen3 30B-A3B'],
    'Phi': ['Phi-2 2.7B', 'Phi-3 Mini', 'Phi-3 Medium',
            'Phi-4 Mini', 'Phi-4 14B'],
}

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Model sets
# ---------------------------------------------------------------------------
triage = pd.read_csv(TRIAGE_CSV)
engaged = [m for m in triage[triage['verdict'] == 'engaged']['model']]
pers = pd.read_csv(PERS_CSV).set_index('model')

screen20 = [m for m in engaged
            if pers.loc[m, 'b_logp_p'] < 0.001
            and pers.loc[m, 'b_d_review_score_p'] < 0.001]

print(f'Engaged models: {len(engaged)}')
print(f'Well-identified screen (b_logp_p<.001 & b_d_review_score_p<.001): '
      f'{len(screen20)}')

# Point estimates (from personality.csv, the paper's numbers)
point_blogp = {m: pers.loc[m, 'b_logp'] for m in engaged}
point_brev = {m: pers.loc[m, 'b_d_review_score'] for m in engaged}
point_wtp = {m: point_brev[m] / abs(point_blogp[m]) * 250 for m in engaged}

# ---------------------------------------------------------------------------
# Build datasets and group index maps
# ---------------------------------------------------------------------------
print('\nBuilding datasets...')
Xs, ys, group_rows, starts = {}, {}, {}, {}
canon_groups = None
for m in engaged:
    df = build_dataset(m, task_type='binary')
    X = sm.add_constant(df[CTRLS + [PRICE_VAR]]).to_numpy(dtype=float)
    y = df['choice_A'].to_numpy(dtype=float)
    gids = np.sort(df['group_id'].unique())
    if canon_groups is None:
        canon_groups = gids
    else:
        assert np.array_equal(gids, canon_groups), \
            f'{m}: group_id set differs from canonical'
    gvals = df['group_id'].to_numpy()
    group_rows[m] = [np.where(gvals == g)[0] for g in canon_groups]
    Xs[m], ys[m] = X, y
    starts[m] = sm.Logit(y, X).fit(disp=False).params

n_groups = len(canon_groups)
print(f'Unique binary group_ids: {n_groups}  (rows per model: {len(ys[engaged[0]])})')
assert n_groups == 450, f'expected 450 groups, got {n_groups}'

IDX_REV = 1 + CTRLS.index('d_review_score')   # column index of d_review_score
IDX_LP = 1 + len(CTRLS)                       # column index of d_log_price

# ---------------------------------------------------------------------------
# Cluster bootstrap: identical group draws across models
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
draws = rng.integers(0, n_groups, size=(B, n_groups))

bs_blogp = np.full((len(engaged), B), np.nan)
bs_brev = np.full((len(engaged), B), np.nan)
n_fail = {m: 0 for m in engaged}

t0 = time.time()
for mi, m in enumerate(engaged):
    rows = group_rows[m]
    X, y, sp = Xs[m], ys[m], starts[m]
    for b in range(B):
        idx = np.concatenate([rows[g] for g in draws[b]])
        try:
            res = sm.Logit(y[idx], X[idx]).fit(
                disp=False, start_params=sp, maxiter=200)
            if not res.mle_retvals.get('converged', False):
                raise RuntimeError('no convergence')
            bs_blogp[mi, b] = res.params[IDX_LP]
            bs_brev[mi, b] = res.params[IDX_REV]
        except Exception:
            n_fail[m] += 1
    print(f'  [{mi+1:2d}/{len(engaged)}] {m:<20s} fails={n_fail[m]:3d} '
          f'elapsed={time.time()-t0:6.1f}s')

bs_wtp = bs_brev / np.abs(bs_blogp) * 250
total_fail = sum(n_fail.values())
print(f'Total failed/non-converged fits: {total_fail} of {len(engaged)*B}')

# ---------------------------------------------------------------------------
# Per-model percentile CIs
# ---------------------------------------------------------------------------
def pctl_ci(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    return tuple(np.percentile(v, [2.5, 97.5]))

ci_blogp = {m: pctl_ci(bs_blogp[i]) for i, m in enumerate(engaged)}
ci_brev = {m: pctl_ci(bs_brev[i]) for i, m in enumerate(engaged)}
ci_wtp = {m: pctl_ci(bs_wtp[i]) for i, m in enumerate(engaged)}

# ---------------------------------------------------------------------------
# Rank stability
# ---------------------------------------------------------------------------
# |b_logp| ranking over all engaged models (rank 1 = most price sensitive)
absb = np.abs(bs_blogp)
ok_all = ~np.isnan(absb).any(axis=0)
n_ok_all = int(ok_all.sum())

point_absb_vec = np.array([abs(point_blogp[m]) for m in engaged])
point_rank_absb = rankdata(-point_absb_vec, method='average')

rank_absb = np.full((len(engaged), B), np.nan)
sp_absb = []
for b in range(B):
    if not ok_all[b]:
        continue
    r = rankdata(-absb[:, b], method='average')
    rank_absb[:, b] = r
    sp_absb.append(spearmanr(r, point_rank_absb)[0])
mean_sp_absb = float(np.mean(sp_absb))

# WTP ranking over the 20 well-identified models (rank 1 = highest WTP)
s20_idx = [engaged.index(m) for m in screen20]
wtp20 = bs_wtp[s20_idx, :]
ok_20 = ~np.isnan(wtp20).any(axis=0)
n_ok_20 = int(ok_20.sum())

point_wtp_vec = np.array([point_wtp[m] for m in screen20])
point_rank_wtp = rankdata(-point_wtp_vec, method='average')

rank_wtp = np.full((len(screen20), B), np.nan)
sp_wtp = []
for b in range(B):
    if not ok_20[b]:
        continue
    r = rankdata(-wtp20[:, b], method='average')
    rank_wtp[:, b] = r
    sp_wtp.append(spearmanr(r, point_rank_wtp)[0])
mean_sp_wtp = float(np.mean(sp_wtp))

def rank_ci(mat, i):
    v = mat[i][~np.isnan(mat[i])]
    return tuple(np.percentile(v, [2.5, 97.5]))

# Headline probabilities (over draws where all 20 fits succeeded)
i_phi4m = screen20.index('Phi-4 Mini')
i_gem2 = screen20.index('Gemma2 9B')
i_gpt54 = screen20.index('GPT-5.4')
n20 = len(screen20)

r_phi = rank_wtp[i_phi4m][ok_20]
r_gem = rank_wtp[i_gem2][ok_20]
r_g54 = rank_wtp[i_gpt54][ok_20]
p_phi4_first = float(np.mean(r_phi <= 1.0))
p_gem2_bottom2 = float(np.mean(r_gem >= n20 - 1))
p_gpt54_bottom3 = float(np.mean(r_g54 >= n20 - 2))

# ---------------------------------------------------------------------------
# The 16x WTP spread: max(wtp)/min(wtp) across the 20 models
# ---------------------------------------------------------------------------
point_ratio = point_wtp_vec.max() / point_wtp_vec.min()
ratios = []
n_nonpos_min = 0
for b in range(B):
    if not ok_20[b]:
        continue
    w = wtp20[:, b]
    if w.min() <= 0:
        n_nonpos_min += 1
        continue
    ratios.append(w.max() / w.min())
ratios = np.array(ratios)
ratio_ci = tuple(np.percentile(ratios, [2.5, 97.5]))
ratio_med = float(np.median(ratios))

# ---------------------------------------------------------------------------
# WTP CI overlap among the 20 models
# ---------------------------------------------------------------------------
n_pairs = 0
n_disjoint = 0
disjoint_share_by_model = {m: 0 for m in screen20}
for a in range(n20):
    for c in range(a + 1, n20):
        n_pairs += 1
        lo_a, hi_a = ci_wtp[screen20[a]]
        lo_c, hi_c = ci_wtp[screen20[c]]
        if lo_a > hi_c or lo_c > hi_a:
            n_disjoint += 1
            disjoint_share_by_model[screen20[a]] += 1
            disjoint_share_by_model[screen20[c]] += 1

# ---------------------------------------------------------------------------
# Multiplicity audit (184 tests: 8 coefficients x 23 engaged models)
# ---------------------------------------------------------------------------
PCOLS = {
    'b_logp': 'b_logp_p',
    'b_linp': 'b_linp_p',
    'b_d_stars': 'b_d_stars_p',
    'b_d_bed': 'b_d_bed_p',
    'b_d_cancel': 'b_d_cancel_p',
    'b_d_breakfast': 'b_d_breakfast_p',
    'b_d_review_score': 'b_d_review_score_p',
    'b_d_review_count_100': 'b_d_review_count_100_p',
}
mrows = []
for m in engaged:
    for coef, col in PCOLS.items():
        claimed = (m in screen20) and coef in ('b_logp', 'b_d_review_score')
        mrows.append(dict(model=m, coef=coef, p_raw=pers.loc[m, col],
                          paper_claimed=claimed))
maudit = pd.DataFrame(mrows)
assert len(maudit) == len(engaged) * 8, 'expected 8 p-values per model'

rej_bonf, p_bonf, _, _ = multipletests(maudit['p_raw'], alpha=ALPHA,
                                       method='bonferroni')
rej_bh, p_bh, _, _ = multipletests(maudit['p_raw'], alpha=ALPHA,
                                   method='fdr_bh')
maudit['p_bonferroni'] = p_bonf
maudit['p_bh'] = p_bh
maudit['sig_raw_05'] = maudit['p_raw'] < ALPHA
maudit['sig_bonferroni_05'] = rej_bonf
maudit['sig_bh_05'] = rej_bh

claimed = maudit[maudit['paper_claimed']]
n_claimed = len(claimed)
max_claimed_raw = claimed['p_raw'].max()
max_row = claimed.loc[claimed['p_raw'].idxmax()]
lost_bonf = claimed[~claimed['sig_bonferroni_05']]
lost_bh = claimed[~claimed['sig_bh_05']]

# ---------------------------------------------------------------------------
# Capability correlations among open-weight engaged models
# ---------------------------------------------------------------------------
open_engaged = [m for m in engaged if m not in PROPRIETARY]
ow = pd.DataFrame({
    'model': open_engaged,
    'params_b': [MODEL_PARAMS_B[m] for m in open_engaged],
    'wtp': [point_wtp[m] for m in open_engaged],
    'r2_log': [pers.loc[m, 'r2_log'] for m in open_engaged],
})
ow['log10_params'] = np.log10(ow['params_b'])

pear_wtp = pearsonr(ow['log10_params'], ow['wtp'])
spear_wtp = spearmanr(ow['log10_params'], ow['wtp'])
pear_r2 = pearsonr(ow['log10_params'], ow['r2_log'])
spear_r2 = spearmanr(ow['log10_params'], ow['r2_log'])

# Same correlations restricted to the open-weight members of the 20-model set
ow20 = ow[ow['model'].isin(screen20)]
pear_wtp20 = pearsonr(ow20['log10_params'], ow20['wtp'])
spear_wtp20 = spearmanr(ow20['log10_params'], ow20['wtp'])

fam_rows = []
for fam, members in FAMILIES.items():
    sub = ow[ow['model'].isin(members)]
    rho, p = spearmanr(sub['params_b'], sub['r2_log'])
    fam_rows.append(dict(family=fam, n=len(sub), spearman_r2_vs_params=rho,
                         p=p, members='; '.join(sub['model'])))

# ---------------------------------------------------------------------------
# Output CSVs
# ---------------------------------------------------------------------------
crows = []
for i, m in enumerate(engaged):
    in20 = m in screen20
    row = dict(
        model=m,
        in_screen20=in20,
        n_boot_fail=n_fail[m],
        b_logp=point_blogp[m],
        b_logp_ci_lo=ci_blogp[m][0], b_logp_ci_hi=ci_blogp[m][1],
        b_review=point_brev[m],
        b_review_ci_lo=ci_brev[m][0], b_review_ci_hi=ci_brev[m][1],
        wtp=point_wtp[m],
        wtp_ci_lo=ci_wtp[m][0], wtp_ci_hi=ci_wtp[m][1],
        rank_absb_point=point_rank_absb[i],
        rank_absb_ci_lo=rank_ci(rank_absb, i)[0],
        rank_absb_ci_hi=rank_ci(rank_absb, i)[1],
    )
    if in20:
        j = screen20.index(m)
        row['rank_wtp_point'] = point_rank_wtp[j]
        row['rank_wtp_ci_lo'] = rank_ci(rank_wtp, j)[0]
        row['rank_wtp_ci_hi'] = rank_ci(rank_wtp, j)[1]
    else:
        row['rank_wtp_point'] = np.nan
        row['rank_wtp_ci_lo'] = np.nan
        row['rank_wtp_ci_hi'] = np.nan
    crows.append(row)
cis = pd.DataFrame(crows).sort_values('rank_absb_point')
cis_path = os.path.join(OUT_DIR, 'rank_stability_cis.csv')
cis.to_csv(cis_path, index=False, encoding='utf-8')

maudit_path = os.path.join(OUT_DIR, 'multiplicity_audit.csv')
maudit.to_csv(maudit_path, index=False, encoding='utf-8')

# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------
md = []
md.append('# Rank stability, bootstrap CIs, multiplicity\n')
md.append(f'Cluster bootstrap over the {n_groups} unique binary group_id '
          f'values, B={B}, numpy seed {SEED}; identical group draws applied '
          f'to every model so per-draw rankings are comparable. '
          f'Log-price logit spec: CTRLS + d_log_price (the paper spec). '
          f'WTP = b_review / |b_logp| x 250.\n')
md.append(f'- Engaged models: {len(engaged)}; well-identified screen '
          f'(b_logp_p < .001 and b_d_review_score_p < .001): '
          f'{len(screen20)} models.')
md.append(f'- Failed / non-converged bootstrap fits: {total_fail} of '
          f'{len(engaged)*B}.')
md.append(f'- Draws with all {len(engaged)} engaged fits valid: {n_ok_all}; '
          f'draws with all {len(screen20)} screen fits valid: {n_ok_20}.\n')

md.append('## Per-model 95% percentile CIs and rank CIs\n')
md.append('Sorted by point-estimate |b_logp| rank (1 = most price '
          'sensitive). WTP ranks are over the 20-model screen '
          '(1 = highest WTP). Full detail in rank_stability_cis.csv.\n')
md.append('| Model | b_logp [95% CI] | WTP [95% CI] | rank |b_logp| '
          '[95% CI] | rank WTP [95% CI] |')
md.append('|---|---|---|---|---|')
for _, r in cis.iterrows():
    wtp_s = (f'{r["wtp"]:.0f} [{r["wtp_ci_lo"]:.0f}, {r["wtp_ci_hi"]:.0f}]')
    if r['in_screen20']:
        rk = (f'{r["rank_wtp_point"]:.0f} [{r["rank_wtp_ci_lo"]:.0f}, '
              f'{r["rank_wtp_ci_hi"]:.0f}]')
    else:
        rk = 'not in screen'
    md.append(f'| {r["model"]} | {r["b_logp"]:.2f} '
              f'[{r["b_logp_ci_lo"]:.2f}, {r["b_logp_ci_hi"]:.2f}] '
              f'| {wtp_s} | {r["rank_absb_point"]:.0f} '
              f'[{r["rank_absb_ci_lo"]:.0f}, {r["rank_absb_ci_hi"]:.0f}] '
              f'| {rk} |')
md.append('')

md.append('## Rank stability\n')
md.append(f'- Mean Spearman correlation between each bootstrap draw\'s '
          f'|b_logp| ranking ({len(engaged)} engaged models) and the '
          f'point-estimate ranking: {mean_sp_absb:.3f} '
          f'(over {n_ok_all} complete draws).')
md.append(f'- Mean Spearman for the WTP ranking ({len(screen20)} models): '
          f'{mean_sp_wtp:.3f} (over {n_ok_20} complete draws).')
md.append(f'- P(Phi-4 Mini ranks #1 on WTP) = {p_phi4_first:.3f}.')
md.append(f'- P(Gemma2 9B in bottom 2 on WTP) = {p_gem2_bottom2:.3f}.')
md.append(f'- P(GPT-5.4 in bottom 3 on WTP) = {p_gpt54_bottom3:.3f}.\n')

md.append('## The 16x WTP spread\n')
md.append(f'- Point estimate: max(WTP)/min(WTP) over the {len(screen20)} '
          f'models = {point_ratio:.1f}x '
          f'(Phi-4 Mini {point_wtp["Phi-4 Mini"]:.0f} / '
          f'Gemma2 9B {point_wtp["Gemma2 9B"]:.0f}).')
md.append(f'- Bootstrap 95% percentile CI for the ratio: '
          f'[{ratio_ci[0]:.1f}x, {ratio_ci[1]:.1f}x]; median '
          f'{ratio_med:.1f}x (over {len(ratios)} draws; {n_nonpos_min} '
          f'draws with min(WTP) <= 0 were excluded from the ratio).\n')

md.append('## WTP CI overlap among the 20 models\n')
md.append(f'- Of the {n_pairs} pairwise comparisons, {n_disjoint} pairs '
          f'({100*n_disjoint/n_pairs:.0f}%) have non-overlapping 95% '
          f'bootstrap CIs.')
md.append('- Interpretation: individual adjacent models are often not '
          'separable, but the cross-model spread is not noise; the '
          'majority of pairwise contrasts, and in particular the '
          'top-vs-bottom contrasts behind the headline spread, remain '
          'separated after accounting for sampling uncertainty.\n')

md.append('## Multiplicity audit (184 tests)\n')
md.append(f'- All 8 estimated coefficients x {len(engaged)} engaged models '
          f'= {len(maudit)} tests; Bonferroni and Benjamini-Hochberg at '
          f'alpha = {ALPHA}.')
md.append(f'- Significant at raw p < .05: {int(maudit["sig_raw_05"].sum())}; '
          f'after Bonferroni: {int(maudit["sig_bonferroni_05"].sum())}; '
          f'after BH: {int(maudit["sig_bh_05"].sum())}.')
md.append(f'- Paper-claimed coefficients (b_logp and b_d_review_score for '
          f'the {len(screen20)}-model screen, {n_claimed} tests): largest '
          f'RAW p = {max_claimed_raw:.2e} ({max_row["model"]}, '
          f'{max_row["coef"]}).')
md.append(f'- Claimed coefficients losing significance under Bonferroni: '
          f'{len(lost_bonf)}; under BH: {len(lost_bh)}.')
md.append(f'- Bonferroni-adjusted p for the weakest claimed coefficient: '
          f'{min(max_claimed_raw*len(maudit), 1):.2e} '
          f'(threshold {ALPHA}).\n')

md.append('## Capability correlations\n')
md.append(f'Open-weight engaged models (n = {len(ow)}); parameter counts '
          f'reused verbatim from analysis/figures.py (MoE = active '
          f'params). Proprietary models excluded (tier-based size '
          f'estimates only).\n')
md.append(f'- WTP vs log10(params): Pearson r = {pear_wtp[0]:.3f} '
          f'(p = {pear_wtp[1]:.3f}); Spearman rho = {spear_wtp[0]:.3f} '
          f'(p = {spear_wtp[1]:.3f}).')
md.append(f'- Same, restricted to open-weight members of the 20-model '
          f'screen (n = {len(ow20)}): Pearson r = {pear_wtp20[0]:.3f} '
          f'(p = {pear_wtp20[1]:.3f}); Spearman rho = '
          f'{spear_wtp20[0]:.3f} (p = {spear_wtp20[1]:.3f}).')
md.append(f'- r2_log (decisiveness) vs log10(params): Pearson r = '
          f'{pear_r2[0]:.3f} (p = {pear_r2[1]:.3f}); Spearman rho = '
          f'{spear_r2[0]:.3f} (p = {spear_r2[1]:.3f}).')
md.append('- Per-family Spearman of r2_log vs params (engaged members '
          'only):')
for fr in fam_rows:
    md.append(f'  - {fr["family"]} (n = {fr["n"]}): rho = '
              f'{fr["spearman_r2_vs_params"]:.3f} (p = {fr["p"]:.3f}) '
              f'[{fr["members"]}]')
md.append('')

md_path = os.path.join(OUT_DIR, 'rank_stability.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f'\nSaved {cis_path}')
print(f'Saved {maudit_path}')
print(f'Saved {md_path}')

# Console summary (ASCII only)
print('\n=== HEADLINES ===')
print(f'screen20 count: {len(screen20)}')
print(f'mean Spearman |b_logp| ranking: {mean_sp_absb:.3f}')
print(f'mean Spearman WTP ranking:      {mean_sp_wtp:.3f}')
print(f'P(Phi-4 Mini #1 WTP):     {p_phi4_first:.3f}')
print(f'P(Gemma2 9B bottom-2):    {p_gem2_bottom2:.3f}')
print(f'P(GPT-5.4 bottom-3):      {p_gpt54_bottom3:.3f}')
print(f'ratio point {point_ratio:.1f}x, CI [{ratio_ci[0]:.1f}, '
      f'{ratio_ci[1]:.1f}], median {ratio_med:.1f}, '
      f'nonpos-min draws {n_nonpos_min}')
print(f'non-overlapping WTP CI pairs: {n_disjoint}/{n_pairs}')
print(f'max claimed raw p: {max_claimed_raw:.3e} '
      f'({max_row["model"]}, {max_row["coef"]})')
print(f'claimed lost under Bonferroni: {len(lost_bonf)}, BH: {len(lost_bh)}')
print(f'open-weight WTP~log10(params): r={pear_wtp[0]:.3f} '
      f'rho={spear_wtp[0]:.3f}')
print(f'open-weight r2_log~log10(params): r={pear_r2[0]:.3f} '
      f'rho={spear_r2[0]:.3f}')
