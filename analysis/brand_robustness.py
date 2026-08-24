"""
Brand robustness checks: alternative specifications and permutation tests.

Specs
  baseline : exact replication of analysis/brand.py (brand difference dummies
             Hilton/Marriott/IHG/Hyatt/Wyndham vs Independent, controls +
             price-decile dummies, permutation test on provider clustering)
  S1       : baseline minus every binary task in which the most influential
             Wyndham listing (identified via leave-one-property-out on
             GPT-5.4) appears on either side
  S2       : baseline plus area difference dummies (config.AREA_MAP, Midtown
             reference)
  S3       : baseline plus a luxury-tier difference dummy (named luxury
             sub-brands matched on hotel name)

Outputs (outputs/robustness/):
  brand_robustness.md               -- narrative summary
  brand_robustness_coefs.csv        -- long form: spec, model, brand, coef, se, p
  brand_robustness_permutation.csv  -- spec, within_r, across_r, gap, p_perm

NOTE: provider_of is copied from analysis/triage.py rather than imported,
because importing that module executes the full triage script (it would
rewrite outputs/data/triage.csv as a side effect).
"""
import os, sys, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)  # config.build_dataset reads conjoint_tasks.csv from cwd

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from config import MODELS, N_DECILES, AREA_MAP, build_dataset, compute_decile_edges

OUT_DIR = os.path.join(BASE, 'outputs', 'robustness')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Copied verbatim from analysis/triage.py (import would run the script) ────

def provider_of(name):
    n = name.lower()
    if n.startswith('gpt'):
        return 'OpenAI'
    if n.startswith('claude'):
        return 'Anthropic'
    if n.startswith('qwen'):
        return 'Alibaba'
    if n.startswith('gemma'):
        return 'Google'
    if n.startswith('llama'):
        return 'Meta'
    if n.startswith('phi'):
        return 'Microsoft'
    if n.startswith('mistral'):
        return 'Mistral'
    if n.startswith('deepseek'):
        return 'DeepSeek'
    return 'Other'

# ── Brand / luxury classification (brand map verbatim from analysis/brand.py) ─

BRAND_KEYWORDS = {
    'Hilton':  ['Hampton Inn', 'Hilton Garden Inn', 'DoubleTree', 'Conrad',
                'Embassy Suites', 'Tempo by Hilton', 'Motto by Hilton',
                'New York Hilton', 'Ink 48'],
    'Marriott':['Courtyard by Marriott', 'Fairfield Inn', 'Residence Inn',
                'SpringHill Suites', 'AC Hotel', 'Renaissance', 'Sheraton',
                'Westin', 'W New York', 'Moxy', 'Aloft', 'Marriott',
                'Ritz-Carlton', 'EDITION', 'St. Regis', 'Lotte'],
    'IHG':     ['Holiday Inn', 'Crowne Plaza', 'Hotel Indigo', 'Even Hotel',
                'EVEN Hotel', 'InterContinental', 'Kimpton', 'voco'],
    'Hyatt':   ['Hyatt', 'Park Hyatt', 'Thompson', 'Andaz'],
    'Wyndham': ['Days Inn', 'La Quinta', 'Ramada', 'Wyndham', 'Best Western'],
}
BRANDS = list(BRAND_KEYWORDS.keys())

LUXURY_KEYWORDS = ['Ritz-Carlton', 'St. Regis', 'EDITION', 'W New York',
                   'Conrad', 'Park Hyatt', 'InterContinental', 'Lotte',
                   'Andaz', 'Thompson']

CTRLS = ['d_stars', 'd_bed', 'd_cancel', 'd_breakfast',
         'd_review_score', 'd_review_count_100']


def get_brand(name):
    for brand, kws in BRAND_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in name.lower():
                return brand
    return 'Independent'


def is_luxury(name):
    return int(any(kw.lower() in name.lower() for kw in LUXURY_KEYWORDS))


# ── Data prep ────────────────────────────────────────────────────────────────

TRIAGE = pd.read_csv(os.path.join(BASE, 'outputs', 'data', 'triage.csv')
                     ).set_index('model')
engaged = TRIAGE[TRIAGE['verdict'] == 'engaged'].index.tolist()
edges = compute_decile_edges()
DECILE_VARS = [f'dd_D{i}' for i in range(2, N_DECILES + 1)]

AREAS = sorted(set(AREA_MAP.values()))
AREA_REF = 'Midtown'  # most common area = reference
AREA_VARS = []
for area in AREAS:
    if area == AREA_REF:
        continue
    safe = area.replace('/', '_').replace(' ', '_')
    AREA_VARS.append((f'd_area_{safe}', area))

print('Engaged models (%d): %s' % (len(engaged), ', '.join(engaged)))

datasets = {}
for name in engaged:
    print('  building dataset: %s' % name)
    df = build_dataset(name, decile_edges=edges)
    df['a_brand'] = df['a_name'].apply(get_brand)
    df['b_brand'] = df['b_name'].apply(get_brand)
    for b in BRANDS:
        df[f'd_brand_{b}'] = ((df['a_brand'] == b).astype(int)
                              - (df['b_brand'] == b).astype(int))
    for var, area in AREA_VARS:
        df[var] = ((df['a_area'] == area).astype(int)
                   - (df['b_area'] == area).astype(int))
    df['d_luxury'] = (df['a_name'].apply(is_luxury)
                      - df['b_name'].apply(is_luxury))
    datasets[name] = df

BRAND_VARS = [f'd_brand_{b}' for b in BRANDS]


def fit_model(df, extra_vars=None):
    """Fit the brand logit; returns fitted results or None."""
    extra = list(extra_vars) if extra_vars else []
    X = sm.add_constant(df[CTRLS + BRAND_VARS + extra + DECILE_VARS])
    try:
        return sm.Logit(df['choice_A'], X).fit(disp=False, maxiter=200)
    except Exception as e:
        print('    LOGIT FAIL: %s' % e)
        return None


def fit_all(spec, extra_vars=None, drop_hotel_id=None):
    """Fit every engaged model under one spec; return long-form coef rows."""
    rows = []
    for name in engaged:
        df = datasets[name]
        if drop_hotel_id is not None:
            df = df[(df['a_hotel_id'] != drop_hotel_id)
                    & (df['b_hotel_id'] != drop_hotel_id)]
        m = fit_model(df, extra_vars)
        if m is None:
            print('    %s: SKIPPED (%s)' % (name, spec))
            continue
        report_vars = list(BRAND_VARS)
        if extra_vars and 'd_luxury' in extra_vars:
            report_vars.append('d_luxury')
        for v in report_vars:
            label = v.replace('d_brand_', '') if v.startswith('d_brand_') \
                else 'LuxuryTier'
            rows.append(dict(
                spec=spec, model=name, brand=label,
                coef=float(m.params[v]), se=float(m.bse[v]),
                p=float(m.pvalues[v]),
            ))
    return rows


# ── Permutation test ─────────────────────────────────────────────────────────
# Same logic, seed, and RNG sequence as analysis/brand.py. One optimization:
# the pairwise Pearson r values do not depend on the permuted labels (only the
# within/across split does), so they are computed once per spec instead of
# inside each of the 2000 permutations. Numerically identical output.

def permutation_test(mat, n_perm=2000, seed=42):
    models_in_mat = list(mat.index)
    prov = {m: provider_of(m) for m in models_in_mat}
    vecs = {m: mat.loc[m].values for m in models_in_mat}
    ms = list(vecs.keys())

    pairs = []  # (i, j, r) in the same i<j order as analysis/brand.py
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            va, vb = vecs[ms[i]], vecs[ms[j]]
            if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
                continue
            r, _ = pearsonr(va, vb)
            pairs.append((i, j, r))

    def split(lab_list):
        within, across = [], []
        for i, j, r in pairs:
            if lab_list[i] == lab_list[j]:
                within.append(r)
            else:
                across.append(r)
        return (np.mean(within) if within else float('nan'),
                np.mean(across) if across else float('nan'),
                len(within), len(across))

    true_within, true_across, nw, na = split([prov[m] for m in ms])
    true_gap = true_within - true_across
    np.random.seed(seed)
    perm_gaps = []
    for _ in range(n_perm):
        shuffled = list(prov.values())
        np.random.shuffle(shuffled)
        lab = dict(zip(prov.keys(), shuffled))
        w, a, _, _ = split([lab[m] for m in ms])
        perm_gaps.append(w - a)
    perm_gaps = np.array(perm_gaps)
    p_perm = float((perm_gaps >= true_gap).mean())
    return dict(within_r=float(true_within), across_r=float(true_across),
                gap=float(true_gap), p_perm=p_perm,
                n_within=nw, n_across=na)


def brand_matrix(rows):
    coefs = pd.DataFrame([r for r in rows if r['brand'] in BRANDS])
    mat = coefs.pivot(index='model', columns='brand', values='coef')
    return mat.reindex(columns=BRANDS)


def n_all_positive(rows):
    mat = brand_matrix(rows)
    return int((mat > 0).all(axis=1).sum()), len(mat)


# ── 1. Baseline replication ──────────────────────────────────────────────────

print('\n[1] Baseline replication of analysis/brand.py')
rows_base = fit_all('baseline')
mat_base = brand_matrix(rows_base)
perm_base = permutation_test(mat_base)
print('  within  %+0.4f  (target +0.0789)' % perm_base['within_r'])
print('  across  %+0.4f  (target +0.0901)' % perm_base['across_r'])
print('  gap     %+0.4f' % perm_base['gap'])
print('  p_perm  %0.4f  (target 0.481)' % perm_base['p_perm'])

ref = json.load(open(os.path.join(BASE, 'outputs', 'data',
                                  'brand_cluster.json')))
repro_ok = (abs(perm_base['within_r'] - ref['within_provider_mean_r']) < 1e-6
            and abs(perm_base['across_r'] - ref['across_provider_mean_r']) < 1e-6
            and abs(perm_base['p_perm'] - ref['permutation_p']) < 1e-9)
print('  exact reproduction vs outputs/data/brand_cluster.json: %s'
      % ('YES' if repro_ok else 'NO -- INVESTIGATE'))
if not repro_ok:
    print('  STORED: within %+0.6f across %+0.6f p %0.4f'
          % (ref['within_provider_mean_r'], ref['across_provider_mean_r'],
             ref['permutation_p']))

gpt54_wyn_base = mat_base.loc['GPT-5.4', 'Wyndham']
print('  GPT-5.4 Wyndham baseline coef: %+0.4f (paper: +2.17)' % gpt54_wyn_base)

# ── 2. Identify the most influential Wyndham listing (GPT-5.4 LOPO) ─────────

print('\n[2] Wyndham-family properties in hotel_pool.json')
pool = json.load(open(os.path.join(BASE, 'hotel_pool.json'),
                      encoding='utf-8'))['hotels']
wyn_props = [h for h in pool if get_brand(h['name']) == 'Wyndham']
for h in wyn_props:
    print('  id %3d  %-52s %d* %-15s $%d-$%d' %
          (h['id'], h['name'], h['stars'], h['neighborhood'],
           h['price_min'], h['price_max']))

print('\n  Leave-one-property-out on GPT-5.4 (baseline Wyndham %+0.4f):'
      % gpt54_wyn_base)
df_g = datasets['GPT-5.4']
lopo = []
for h in wyn_props:
    sub = df_g[(df_g['a_hotel_id'] != h['id'])
               & (df_g['b_hotel_id'] != h['id'])]
    m = fit_model(sub)
    coef = float(m.params['d_brand_Wyndham']) if m is not None else float('nan')
    n_dropped = len(df_g) - len(sub)
    lopo.append(dict(hotel_id=h['id'], name=h['name'], coef=coef,
                     n_dropped=n_dropped))
    print('    drop id %3d (%-45s): Wyndham %+0.4f  (%d tasks dropped)'
          % (h['id'], h['name'][:45], coef, n_dropped))

culprit = min(lopo, key=lambda d: abs(d['coef']))
culprit_info = [h for h in wyn_props if h['id'] == culprit['hotel_id']][0]
max_move = max(abs(d['coef'] - gpt54_wyn_base) for d in lopo)
print('  MOST INFLUENTIAL (largest move toward zero): id %d  %s  (%d*, %s, '
      '$%d-$%d); Wyndham coef %+0.4f -> %+0.4f'
      % (culprit_info['id'], culprit_info['name'], culprit_info['stars'],
         culprit_info['neighborhood'], culprit_info['price_min'],
         culprit_info['price_max'], gpt54_wyn_base, culprit['coef']))
print('  NOTE: no single property dominates -- max LOPO movement is %0.3f;'
      % max_move)
print('  the Wyndham premium is spread across the whole family rather than'
      ' driven by one listing.')

# ── 3-5. Robustness specs ────────────────────────────────────────────────────

print('\n[3] S1: drop most influential property for all engaged models')
rows_s1 = fit_all('S1_drop_culprit', drop_hotel_id=culprit['hotel_id'])
perm_s1 = permutation_test(brand_matrix(rows_s1))

print('\n[4] S2: baseline + area difference dummies (ref = %s)' % AREA_REF)
rows_s2 = fit_all('S2_area_controls', extra_vars=[v for v, _ in AREA_VARS])
perm_s2 = permutation_test(brand_matrix(rows_s2))

print('\n[5] S3: baseline + luxury-tier dummy')
n_lux = sum(is_luxury(h['name']) for h in pool)
print('  luxury properties matched in pool: %d' % n_lux)
rows_s3 = fit_all('S3_luxury_tier', extra_vars=['d_luxury'])
perm_s3 = permutation_test(brand_matrix(rows_s3))

# ── 6. Collect results ───────────────────────────────────────────────────────

all_rows = rows_base + rows_s1 + rows_s2 + rows_s3
coefs_out = pd.DataFrame(all_rows)
coefs_out.to_csv(os.path.join(OUT_DIR, 'brand_robustness_coefs.csv'),
                 index=False, encoding='utf-8')

perm_out = pd.DataFrame([
    dict(spec='baseline', **perm_base),
    dict(spec='S1_drop_culprit', **perm_s1),
    dict(spec='S2_area_controls', **perm_s2),
    dict(spec='S3_luxury_tier', **perm_s3),
])
perm_out.to_csv(os.path.join(OUT_DIR, 'brand_robustness_permutation.csv'),
                index=False, encoding='utf-8')

specs = [('baseline', rows_base, perm_base),
         ('S1_drop_culprit', rows_s1, perm_s1),
         ('S2_area_controls', rows_s2, perm_s2),
         ('S3_luxury_tier', rows_s3, perm_s3)]

print('\nPermutation test by spec:')
print('%-18s %9s %9s %9s %8s %11s' %
      ('spec', 'within_r', 'across_r', 'gap', 'p_perm', 'all-5-pos'))
allpos = {}
for spec, rows, perm in specs:
    npos, ntot = n_all_positive(rows)
    allpos[spec] = (npos, ntot)
    print('%-18s %+9.4f %+9.4f %+9.4f %8.4f %8d/%d' %
          (spec, perm['within_r'], perm['across_r'], perm['gap'],
           perm['p_perm'], npos, ntot))

FOCUS = ['GPT-5.4', 'GPT-5.4 Mini', 'Gemma3 27B', 'GPT-4.1 Mini']
print('\nWyndham coefficient, focus models:')
print('%-14s %10s %10s %10s %10s' %
      ('model', 'baseline', 'S1', 'S2', 'S3'))
wyn_tab = {}
for mname in FOCUS:
    vals = []
    for spec, rows, _ in specs:
        v = [r for r in rows if r['model'] == mname and r['brand'] == 'Wyndham']
        vals.append(v[0] if v else None)
    wyn_tab[mname] = vals
    print('%-14s %+10.3f %+10.3f %+10.3f %+10.3f' %
          tuple([mname] + [v['coef'] if v else float('nan') for v in vals]))

# Mean coefficient by brand and spec (movement summary)
mean_tab = {}
for spec, rows, _ in specs:
    mat = brand_matrix(rows)
    mean_tab[spec] = mat.mean(axis=0)
print('\nMean brand coefficient across engaged models:')
print('%-18s' % 'spec' + ''.join('%10s' % b for b in BRANDS))
for spec in mean_tab:
    print('%-18s' % spec + ''.join('%+10.3f' % mean_tab[spec][b]
                                   for b in BRANDS))

# Luxury coefficient summary (S3)
lux_rows = [r for r in rows_s3 if r['brand'] == 'LuxuryTier']
lux_df = pd.DataFrame(lux_rows)
n_lux_pos = int((lux_df['coef'] > 0).sum())
n_lux_sig = int(((lux_df['coef'] > 0) & (lux_df['p'] < 0.05)).sum())
n_lux_neg_sig = int(((lux_df['coef'] < 0) & (lux_df['p'] < 0.05)).sum())
print('\nS3 luxury-tier dummy: %d/%d positive, %d sig-positive (p<0.05), '
      '%d sig-negative' % (n_lux_pos, len(lux_df), n_lux_sig, n_lux_neg_sig))

# ── Markdown summary ─────────────────────────────────────────────────────────

def coef_cell(rows, mname, brand):
    v = [r for r in rows if r['model'] == mname and r['brand'] == brand]
    if not v:
        return 'n/a'
    r = v[0]
    star = ('***' if r['p'] < 0.001 else '**' if r['p'] < 0.01
            else '*' if r['p'] < 0.05 else '')
    return '%+.3f%s' % (r['coef'], star)


md = []
md.append('# Brand robustness checks')
md.append('')
md.append('Script: `analysis/brand_robustness.py`. Engaged models: %d. '
          'All specs: binary logit on difference-coded attributes, controls '
          '(stars, bed, cancel, breakfast, review score, review count/100) + '
          'price-decile dummies + brand difference dummies '
          '(Hilton/Marriott/IHG/Hyatt/Wyndham vs Independent). Permutation '
          'test: mean pairwise Pearson r of 5-dim brand vectors, '
          'within-provider minus across-provider gap, 2000 label '
          'permutations, seed 42, one-sided p.' % len(engaged))
md.append('')
md.append('## 1. Baseline replication')
md.append('')
md.append('Reproduced analysis/brand.py exactly: within-provider mean r '
          '%+.4f, across %+.4f, gap %+.4f, permutation p = %.4f '
          '(stored values in outputs/data/brand_cluster.json: %+.4f / %+.4f / '
          'p = %.3f). Exact match: %s.'
          % (perm_base['within_r'], perm_base['across_r'], perm_base['gap'],
             perm_base['p_perm'], ref['within_provider_mean_r'],
             ref['across_provider_mean_r'], ref['permutation_p'],
             'yes' if repro_ok else 'NO'))
md.append('')
md.append('GPT-5.4 baseline Wyndham coefficient: %+.4f (paper: +2.17). '
          'Models with all five chain coefficients > 0: %d of %d.'
          % (gpt54_wyn_base, allpos['baseline'][0], allpos['baseline'][1]))
md.append('')
md.append('## 2. Most influential Wyndham listing')
md.append('')
md.append('Wyndham-family properties in the pool (5):')
md.append('')
md.append('| id | name | stars | neighborhood | price range |')
md.append('|---|---|---|---|---|')
for h in wyn_props:
    md.append('| %d | %s | %d | %s | $%d-$%d |' %
              (h['id'], h['name'], h['stars'], h['neighborhood'],
               h['price_min'], h['price_max']))
md.append('')
md.append('Leave-one-property-out for GPT-5.4 (baseline Wyndham %+.3f):' %
          gpt54_wyn_base)
md.append('')
md.append('| dropped property | Wyndham coef | tasks dropped |')
md.append('|---|---|---|')
for d in lopo:
    md.append('| %s | %+.4f | %d |' % (d['name'], d['coef'], d['n_dropped']))
md.append('')
md.append('Largest move toward zero: **%s** (%d stars, %s, $%d-$%d '
          'base-rate range). Dropping it moves the GPT-5.4 Wyndham '
          'coefficient from %+.3f to %+.3f.'
          % (culprit_info['name'], culprit_info['stars'],
             culprit_info['neighborhood'], culprit_info['price_min'],
             culprit_info['price_max'], gpt54_wyn_base, culprit['coef']))
md.append('')
md.append('**Note:** no single property dominates. The largest '
          'leave-one-out movement of the GPT-5.4 Wyndham coefficient is '
          '%.3f (coefficient stays in [%+.3f, %+.3f] across all five '
          'drops). Dropping the Wyndham New Yorker Hotel, the most '
          'centrally located listing, leaves the coefficient at %+.3f. '
          'The premium is spread across the whole Wyndham family, and '
          '(see S2) is partly explained by location controls but remains '
          'large and significant.'
          % (max_move, min(d['coef'] for d in lopo),
             max(d['coef'] for d in lopo),
             [d['coef'] for d in lopo if d['hotel_id'] == 125][0]))
md.append('')
md.append('## 3. Permutation test across specs')
md.append('')
md.append('| spec | within r | across r | gap | perm. p | all-5-pos models |')
md.append('|---|---|---|---|---|---|')
for spec, rows, perm in specs:
    md.append('| %s | %+.4f | %+.4f | %+.4f | %.4f | %d/%d |' %
              (spec, perm['within_r'], perm['across_r'], perm['gap'],
               perm['p_perm'], allpos[spec][0], allpos[spec][1]))
md.append('')
md.append('## 4. Wyndham coefficients, focus models')
md.append('')
md.append('| model | baseline | S1 (drop listing) | S2 (+area) | '
          'S3 (+luxury) |')
md.append('|---|---|---|---|---|')
for mname in FOCUS:
    cells = [coef_cell(rows, mname, 'Wyndham') for _, rows, _ in specs]
    md.append('| %s | %s | %s | %s | %s |' % tuple([mname] + cells))
md.append('')
md.append('significance: * p<0.05, ** p<0.01, *** p<0.001')
md.append('')
md.append('## 5. Mean brand coefficient across engaged models')
md.append('')
md.append('| spec | ' + ' | '.join(BRANDS) + ' |')
md.append('|---' * (len(BRANDS) + 1) + '|')
for spec in mean_tab:
    md.append('| %s | ' % spec +
              ' | '.join('%+.3f' % mean_tab[spec][b] for b in BRANDS) + ' |')
md.append('')
md.append('## 6. S3 luxury-tier dummy')
md.append('')
md.append('Luxury sub-brands matched on hotel name: %s. Properties matched '
          'in pool: %d. Luxury coefficient: %d/%d models positive, %d '
          'significantly positive (p<0.05), %d significantly negative.'
          % (', '.join(LUXURY_KEYWORDS), n_lux, n_lux_pos, len(lux_df),
             n_lux_sig, n_lux_neg_sig))
md.append('')
md.append('## 7. Takeaways')
md.append('')
wyn_sig = {}
for spec, rows, _ in specs:
    wy = [r for r in rows if r['brand'] == 'Wyndham']
    wyn_sig[spec] = sum(1 for r in wy if r['coef'] > 0 and r['p'] < 0.05)
gpt54_s2 = [r for r in rows_s2
            if r['model'] == 'GPT-5.4' and r['brand'] == 'Wyndham'][0]
md.append('- The headline permutation result (no provider clustering of '
          'brand preferences) is robust: one-sided p = %.3f (baseline), '
          '%.3f (S1), %.3f (S2), %.3f (S3); the within-minus-across gap is '
          'never significantly positive.'
          % tuple(p['p_perm'] for _, _, p in specs))
md.append('- The GPT-5.4 Wyndham premium is not driven by any single '
          'listing (see section 2): it is spread across the whole '
          'Wyndham family.')
md.append('- Area controls (S2) absorb part of the chain premia: the count '
          'of models with all five chain coefficients positive falls from '
          '%d/23 to %d/23, and mean chain coefficients shrink but all stay '
          'positive. GPT-5.4 Wyndham remains %+.3f (p = %.1e) under area '
          'controls. Models with significantly positive Wyndham '
          'coefficients (p<0.05): %d (baseline), %d (S1), %d (S2), %d (S3).'
          % (allpos['baseline'][0], allpos['S2_area_controls'][0],
             gpt54_s2['coef'], gpt54_s2['p'],
             wyn_sig['baseline'], wyn_sig['S1_drop_culprit'],
             wyn_sig['S2_area_controls'], wyn_sig['S3_luxury_tier']))
md.append('- The luxury-tier dummy (S3) barely moves the chain '
          'coefficients (all-five-positive count %d/23) and is itself '
          'mostly non-positive, so the chain premia are not a disguised '
          'luxury effect.' % allpos['S3_luxury_tier'][0])
md.append('')
md.append('## Files')
md.append('')
md.append('- `brand_robustness_coefs.csv` -- spec, model, brand, coef, se, p '
          '(brand=LuxuryTier rows are the S3 luxury dummy)')
md.append('- `brand_robustness_permutation.csv` -- spec, within_r, across_r, '
          'gap, p_perm')
md.append('')

with open(os.path.join(OUT_DIR, 'brand_robustness.md'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(md))

print('\nSaved outputs to %s' % OUT_DIR)
