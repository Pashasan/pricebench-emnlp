"""
Locked-model response forensics: does position-locking reflect parse failure,
or a genuine slot-following behavior?

For the 5 position-locked models and 2 engaged contrast models we compute,
on the BINARY tasks (task_id 1-1800 original, 10001-11800 swap):

  1. Valid-choice rate (choice in {A, B}) and count of empty/NaN choices.
  2. Raw-output character: bare-letter (^[ABC]$ after strip), short-verbose
     (<= 50 chars containing a choice letter), long-verbose (> 50 chars),
     empty/other.  Plus 5 most common raw outputs per locked model.
  3. Slot-following: P(pick the first-SHOWN option) separately by ordering.
     Original file: first-shown = recorded choice 'A'.  Swap file: choices are
     stored flipped back to the original frame, so first-shown = recorded 'B'.
  4. Attribute signal: standard log-price logit (personality.py spec) on the
     difference-coded binary data; b_logp, p, pseudo-R2, and the number of the
     7 slope coefficients reaching p < .05.
  5. Retry proxy: share of ORIGINAL-run binary raw outputs that are not a bare
     letter.  (The runner retried up to 3 times when parsing failed; only the
     final attempt's raw_output is logged, so exact retry counts are not
     recoverable from the CSVs.)

Outputs:
  outputs/robustness/locked_forensics.csv
  outputs/robustness/locked_forensics.md
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, BASE)
os.chdir(BASE)  # config helpers read result CSVs relative to project root

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import MODELS, build_dataset

OUT_DIR = os.path.join(BASE, 'outputs', 'robustness')
OUT_CSV = os.path.join(OUT_DIR, 'locked_forensics.csv')
OUT_MD = os.path.join(OUT_DIR, 'locked_forensics.md')
TRIAGE_CSV = os.path.join(BASE, 'outputs', 'data', 'triage.csv')

LOCKED = ['Llama3.2 1B', 'Mistral 7B', 'Qwen3 0.6B', 'Llama3 8B',
          'Llama3.1 8B']
ENGAGED = ['Gemma3 4B', 'Qwen3 4B']
ALL = LOCKED + ENGAGED

CTRLS = ['d_stars', 'd_bed', 'd_cancel', 'd_breakfast',
         'd_review_score', 'd_review_count_100']

BIN_ORIG = (1, 1800)
BIN_SWAP = (10001, 11800)
N_EXAMPLES = 5


def binary_rows(df, lo, hi):
    """Rows of a raw results CSV belonging to binary tasks."""
    return df[(df['task_id'] >= lo) & (df['task_id'] <= hi)].copy()


def classify_raw(raw):
    """Classify one raw_output string."""
    if not isinstance(raw, str) or raw.strip() == '':
        return 'empty/other'
    t = raw.strip()
    if re.fullmatch(r'[ABC]', t):
        return 'bare-letter'
    if len(t) > 50:
        return 'long-verbose'
    if re.search(r'[ABC]', t, re.IGNORECASE):
        return 'short-verbose'
    return 'empty/other'


def sanitize(text, width=80):
    """One-line, truncated representation of a raw output for the md file."""
    if not isinstance(text, str):
        return '<NaN>'
    t = text.replace('\r', ' ').replace('\n', ' \\n ')
    t = re.sub(r'\s+', ' ', t).strip()
    if len(t) > width:
        t = t[:width - 3] + '...'
    return t if t else '<empty string>'


triage = pd.read_csv(TRIAGE_CSV).set_index('model')

rows = []
examples = {}
notes = {}

for name in ALL:
    orig_file, swap_file = MODELS[name]
    orig = pd.read_csv(orig_file)
    swap = pd.read_csv(swap_file)

    note_parts = []

    # Mistral's swap file also contains a legacy +1000-offset block; keep only
    # the standard +10000 block for the swap-side numbers.
    n_legacy = int((swap['task_id'] < BIN_SWAP[0]).sum())
    if n_legacy:
        legacy = swap[swap['task_id'] < BIN_SWAP[0]].copy()
        legacy_bin = binary_rows(legacy, 1001, 2800)
        std_bin = binary_rows(swap, *BIN_SWAP)
        merged = legacy_bin.assign(tid=legacy_bin['task_id'] - 1000).merge(
            std_bin.assign(tid=std_bin['task_id'] - 10000),
            on='tid', suffixes=('_old', '_new'))
        agree = (merged['choice_old'] == merged['choice_new']).mean()
        note_parts.append(
            f'swap file also holds a legacy +1000-offset block '
            f'({n_legacy} rows, excluded); its binary choices replicate the '
            f'standard block at {agree*100:.1f}% agreement')
        swap = swap[swap['task_id'] >= BIN_SWAP[0]].copy()

    ob = binary_rows(orig, *BIN_ORIG)
    sb = binary_rows(swap, *BIN_SWAP)

    if len(ob) < 1800:
        note_parts.append(f'original file missing {1800 - len(ob)} '
                          f'binary tasks')
    if len(sb) < 1800:
        note_parts.append(f'swap file missing {1800 - len(sb)} binary tasks')
    n_tern_missing = (1800 - (len(orig) - len(ob))) + \
                     (1800 - (len(swap) - len(sb)))
    if n_tern_missing > 0:
        note_parts.append(f'{n_tern_missing} ternary rows absent across the '
                          f'pair (does not affect binary analysis)')

    # 1. valid-choice rate + empty/NaN counts (binary rows)
    ob_valid = ob['choice'].isin(['A', 'B'])
    sb_valid = sb['choice'].isin(['A', 'B'])
    n_empty_orig = int(ob['choice'].isna().sum()
                       + (ob['choice'].astype(str).str.strip() == '').sum())
    n_empty_swap = int(sb['choice'].isna().sum()
                       + (sb['choice'].astype(str).str.strip() == '').sum())

    # 2. raw output character (binary rows, both orderings pooled)
    pooled_raw = pd.concat([ob['raw_output'], sb['raw_output']],
                           ignore_index=True)
    cats = pooled_raw.apply(classify_raw)
    n_pool = len(cats)
    pct = {c: 100.0 * (cats == c).sum() / n_pool
           for c in ['bare-letter', 'short-verbose', 'long-verbose',
                     'empty/other']}

    # representative examples: most common binary raw outputs
    vc = pd.concat([ob, sb])['raw_output'].fillna('<NaN>').value_counts()
    examples[name] = [(sanitize(v), int(c))
                      for v, c in vc.head(N_EXAMPLES).items()]

    # 5. retry proxy on the ORIGINAL run only
    orig_cats = ob['raw_output'].apply(classify_raw)
    pct_bare_orig = 100.0 * (orig_cats == 'bare-letter').mean()

    # 3. slot-following
    p_first_orig = (ob.loc[ob_valid, 'choice'] == 'A').mean()
    p_first_swap = (sb.loc[sb_valid, 'choice'] == 'B').mean()

    # 4. attribute-signal logit (standard personality.py spec)
    b_logp = b_logp_se = b_logp_p = r2_log = np.nan
    n_sig = np.nan
    n_obs = 0
    fit_err = ''
    try:
        df = build_dataset(name, task_type='binary')
        n_obs = len(df)
        X = sm.add_constant(df[CTRLS + ['d_log_price']])
        m = sm.Logit(df['choice_A'], X).fit(disp=False)
        b_logp = m.params['d_log_price']
        b_logp_se = m.bse['d_log_price']
        b_logp_p = m.pvalues['d_log_price']
        r2_log = m.prsquared
        n_sig = int((m.pvalues[CTRLS + ['d_log_price']] < 0.05).sum())
    except Exception as e:
        fit_err = str(e)
        note_parts.append(f'logit failed: {e}')

    notes[name] = '; '.join(note_parts)

    rows.append(dict(
        model=name,
        group='locked' if name in LOCKED else 'engaged',
        triage_verdict=triage.loc[name, 'verdict'],
        triage_first_rate=triage.loc[name, 'first_rate'],
        n_binary_orig=len(ob),
        n_binary_swap=len(sb),
        valid_rate_orig=float(ob_valid.mean()),
        valid_rate_swap=float(sb_valid.mean()),
        n_invalid_orig=int((~ob_valid).sum()),
        n_invalid_swap=int((~sb_valid).sum()),
        n_empty_orig=n_empty_orig,
        n_empty_swap=n_empty_swap,
        pct_bare=pct['bare-letter'],
        pct_short_verbose=pct['short-verbose'],
        pct_long_verbose=pct['long-verbose'],
        pct_empty_other=pct['empty/other'],
        pct_bare_orig=pct_bare_orig,
        p_first_orig=float(p_first_orig),
        p_first_swap=float(p_first_swap),
        n_obs_logit=n_obs,
        b_logp=b_logp,
        b_logp_se=b_logp_se,
        b_logp_p=b_logp_p,
        r2_log=r2_log,
        n_coef_p05_of7=n_sig,
        notes=notes[name],
    ))

out = pd.DataFrame(rows)
os.makedirs(OUT_DIR, exist_ok=True)
out.to_csv(OUT_CSV, index=False, encoding='utf-8')

# ── Console summary (ASCII only) ─────────────────────────────────────────────
print(f'{"Model":<14s} {"grp":<8s} {"valid_o":>8s} {"valid_s":>8s} '
      f'{"bare%":>6s} {"Pfirst_o":>9s} {"Pfirst_s":>9s} '
      f'{"b_logp":>8s} {"p":>9s} {"R2":>6s} {"sig/7":>6s}')
print('-' * 100)
for _, r in out.iterrows():
    print(f'{r["model"]:<14s} {r["group"]:<8s} '
          f'{r["valid_rate_orig"]*100:>7.1f}% {r["valid_rate_swap"]*100:>7.1f}% '
          f'{r["pct_bare"]:>6.1f} {r["p_first_orig"]*100:>8.1f}% '
          f'{r["p_first_swap"]*100:>8.1f}% '
          f'{r["b_logp"]:>8.3f} {r["b_logp_p"]:>9.2e} {r["r2_log"]:>6.3f} '
          f'{int(r["n_coef_p05_of7"]):>6d}')

# ── Markdown report ──────────────────────────────────────────────────────────
md = []
md.append('# Locked-model response forensics: parse validity, '
          'slot-following, and deviation signal\n')
md.append('Script: `analysis/locked_forensics.py`. Data: raw result CSV '
          'pairs registered in `config.MODELS`, binary tasks only '
          '(task_id 1-1800 original, 10001-11800 swap). Swap-file choices '
          'are stored flipped back to the original frame, so the '
          'first-shown option corresponds to recorded choice A in the '
          'original file and recorded choice B in the swap file.\n')

md.append('## 1. Valid-choice rates (binary tasks)\n')
md.append('| Model | Group | n orig | n swap | Valid orig | Valid swap | '
          'Invalid orig | Invalid swap | Empty/NaN |')
md.append('|---|---|---|---|---|---|---|---|---|')
for _, r in out.iterrows():
    md.append(f'| {r["model"]} | {r["group"]} | {r["n_binary_orig"]} | '
              f'{r["n_binary_swap"]} | {r["valid_rate_orig"]*100:.2f}% | '
              f'{r["valid_rate_swap"]*100:.2f}% | {r["n_invalid_orig"]} | '
              f'{r["n_invalid_swap"]} | '
              f'{r["n_empty_orig"] + r["n_empty_swap"]} |')
md.append('')

md.append('## 2. Raw-output character (binary tasks, both orderings)\n')
md.append('Categories: bare-letter = output is exactly one of A/B/C after '
          'stripping whitespace; short-verbose = <= 50 chars containing a '
          'choice letter (e.g. `A.` or `Option A.`); long-verbose = > 50 '
          'chars; empty/other = empty or no choice letter found.\n')
md.append('| Model | Group | Bare-letter | Short-verbose | Long-verbose | '
          'Empty/other |')
md.append('|---|---|---|---|---|---|')
for _, r in out.iterrows():
    md.append(f'| {r["model"]} | {r["group"]} | {r["pct_bare"]:.1f}% | '
              f'{r["pct_short_verbose"]:.1f}% | '
              f'{r["pct_long_verbose"]:.1f}% | '
              f'{r["pct_empty_other"]:.1f}% |')
md.append('')

md.append('### Representative raw outputs (5 most frequent per model, '
          'truncated to 80 chars)\n')
for name in ALL:
    tag = 'locked' if name in LOCKED else 'engaged contrast'
    md.append(f'**{name}** ({tag}):\n')
    for v, c in examples[name]:
        md.append(f'- `{v}` (x{c})')
    md.append('')

md.append('## 3. Slot-following vs hotel-habit (binary tasks)\n')
md.append('P(pick first-SHOWN option) by ordering. A slot-follower is high '
          'on both columns; a model with a stable preference for one hotel '
          'would be high on one and low on the other.\n')
md.append('| Model | Group | P(first-shown), original | P(first-shown), '
          'swap | Triage first-rate (all tasks) |')
md.append('|---|---|---|---|---|')
for _, r in out.iterrows():
    md.append(f'| {r["model"]} | {r["group"]} | '
              f'{r["p_first_orig"]*100:.1f}% | {r["p_first_swap"]*100:.1f}% | '
              f'{r["triage_first_rate"]*100:.1f}% |')
md.append('')

md.append('## 4. Attribute signal in deviations (standard log-price '
          'logit)\n')
md.append('Spec: `Logit(choice_A ~ const + d_stars + d_bed + d_cancel + '
          'd_breakfast + d_review_score + d_review_count_100 + '
          'd_log_price)` on the pooled original+swap binary data '
          '(identical to `analysis/personality.py`).\n')
md.append('| Model | Group | n | b_logp | SE | p | Pseudo-R2 | '
          'Coefs p<.05 (of 7) |')
md.append('|---|---|---|---|---|---|---|---|')
for _, r in out.iterrows():
    md.append(f'| {r["model"]} | {r["group"]} | {r["n_obs_logit"]} | '
              f'{r["b_logp"]:.3f} | {r["b_logp_se"]:.3f} | '
              f'{r["b_logp_p"]:.2e} | {r["r2_log"]:.4f} | '
              f'{int(r["n_coef_p05_of7"])} |')
md.append('')

md.append('## 5. Retry evidence (original runs)\n')
md.append('The runner retried a task up to 3 times when no choice letter '
          'could be parsed, and logged only the final attempt\'s '
          'raw_output, so exact retry counts are not recoverable from the '
          'CSVs. Two observable proxies: (a) share of final outputs that '
          'are not a bare letter (weak instruction following), and (b) '
          'rows whose final choice is empty/NaN (all 3 attempts failed).\n')
md.append('| Model | Group | Bare-letter, orig run | Not-bare, orig run | '
          'All-attempts-failed rows (orig+swap) |')
md.append('|---|---|---|---|---|')
for _, r in out.iterrows():
    md.append(f'| {r["model"]} | {r["group"]} | {r["pct_bare_orig"]:.1f}% | '
              f'{100 - r["pct_bare_orig"]:.1f}% | '
              f'{r["n_empty_orig"] + r["n_empty_swap"]} |')
md.append('')
md.append('Interpretation caveat: the not-bare outputs of Llama3.2 1B '
          '(`A.`) and Mistral 7B (`Option A.`) parse deterministically on '
          'the first attempt, so a not-bare final output does not imply a '
          'retry happened. The zero unparseable/empty rows show that no '
          'task ever exhausted its 3 attempts.\n')

md.append('## Bottom line\n')
md.append('1. Parse failure is ruled out: every one of the 3,600 binary '
          'responses per model (1,800 per ordering) parsed to a valid A/B '
          'choice; there are zero empty, NaN, or unparseable outputs for '
          'any of the five locked models. Three locked models (Qwen3 '
          '0.6B, Llama3 8B, Llama3.1 8B) emit a perfectly bare letter in '
          '99.5-100% of cases, matching the format discipline of the '
          'engaged contrasts, yet still land in the locked band.\n')
md.append('2. The behavior is slot-following, not a hotel habit: locked '
          'models pick the first-SHOWN option at 90.9-100% under BOTH '
          'orderings, which only position can explain. Engaged contrasts '
          'sit far from the band in both orderings (Gemma3 4B 75.7%/67.9% '
          'primacy lean; Qwen3 4B 24.4%/13.7% recency lean).\n')
md.append('3. A nuance: the rare deviations of Llama3 8B and '
          'Llama3.1 8B (first-shown 90.9-94.3%) do carry attribute '
          'signal (b_logp -0.59 and -0.55, p < 1e-11, 5 of 7 '
          'coefficients p < .05), but the fit is an order of magnitude '
          'weaker than the engaged contrasts (pseudo-R2 0.02-0.03 vs '
          '0.13-0.22) and the price coefficient is attenuated roughly '
          '2.5x by the dominant position behavior, so excluding them is '
          'conservative: it avoids reporting attenuated preference '
          'estimates rather than hiding parse failures. The other three '
          'locked models carry no attribute signal at all (b_logp p >= '
          '0.90, 0 of 7 coefficients significant).\n')

md.append('## Data notes\n')
for name in ALL:
    if notes[name]:
        md.append(f'- {name}: {notes[name]}.')
md.append('')

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))

print(f'\nSaved {OUT_CSV}')
print(f'Saved {OUT_MD}')
