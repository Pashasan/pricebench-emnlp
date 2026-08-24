"""
Temperature check: does position-locking persist at temperature 0.7?

Compares first-shown selection rates and attribute coefficients at T=0.7
(top_p .95, top_k 40, seed 42) against the paper's T=0 baseline for the five
position-locked models and two engaged controls. Tolerates missing files so
it can run on a partial set of result CSVs.

Run from repo root:
    python analysis/analyze_temp.py
Outputs:
    outputs/robustness/temp_check.csv
    outputs/robustness/temp_check.md
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_common import (ROOT, OUT_DIR, TAG2NAME, load_result_pair,
                               build_binary_dataset, fit_log_spec,
                               first_shown_rates)

TEMP_TAGS = ['llama3_1_8b', 'llama3_8b-instruct-fp16',
             'mistral_7b-instruct-fp16', 'qwen3_0_6b', 'llama3_2_1b',
             'gemma3_latest', 'qwen3_4b-instruct']
LOCKED = {'Llama3.1 8B', 'Llama3 8B', 'Mistral 7B', 'Qwen3 0.6B',
          'Llama3.2 1B'}


def main():
    triage = pd.read_csv(os.path.join(ROOT, 'outputs', 'data', 'triage.csv')
                         ).set_index('model')
    pers = pd.read_csv(os.path.join(ROOT, 'outputs', 'data', 'personality.csv')
                       ).set_index('model')

    rows = []
    for tag in TEMP_TAGS:
        name = TAG2NAME[tag]
        orig = os.path.join(ROOT, f'conjoint_results_{tag}_temp07.csv')
        swap = os.path.join(ROOT, f'conjoint_results_{tag}_temp07_swap.csv')
        results = load_result_pair(orig, swap)
        if results is None:
            print(f'  {name:<18s} no temp07 data yet, skipping')
            continue

        rates = first_shown_rates(results)
        row = dict(model=name,
                   role='locked' if name in LOCKED else 'engaged-control',
                   **{k: v for k, v in rates.items()})

        # T=0 baseline (triage.csv stores first_rate as a fraction)
        row['first_t0'] = float(triage.loc[name, 'first_rate']) \
            if name in triage.index else float('nan')

        # Verdict at T=.7 under the paper's [15%, 85%] band
        f = row['first_shown_pooled']
        row['verdict_t07'] = ('position-locked'
                              if (f > 0.85 or f < 0.15) else 'engaged')

        # Attribute coefficients at T=.7
        df = build_binary_dataset(results)
        fit = fit_log_spec(df)
        if fit:
            row.update({f't07_{k}': v for k, v in fit.items()})
        # T=0 coefficients for comparison
        if name in pers.index and pd.notna(pers.loc[name, 'b_logp']):
            row['t0_b_logp'] = float(pers.loc[name, 'b_logp'])
            row['t0_b_logp_se'] = float(pers.loc[name, 'b_logp_se'])
            row['t0_b_review'] = float(pers.loc[name, 'b_d_review_score'])
            row['t0_r2'] = float(pers.loc[name, 'r2_log'])
        rows.append(row)
        print(f'  {name:<18s} first(T0)={row["first_t0"]:.3f} -> '
              f'first(T.7)={f:.3f}  verdict={row["verdict_t07"]}  '
              f'n={rates["n_valid"]}')

    if not rows:
        print('No temp07 results found at all.')
        return
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, 'temp_check.csv'), index=False,
               encoding='utf-8')

    # Markdown summary
    lines = [
        '# Temperature check: position-locking at temperature 0.7',
        '',
        'Settings: temperature 0.7, top_p 0.95, top_k 40, seed 42; binary',
        'tasks, both orderings; first-shown rate pooled over orderings.',
        'T=0 baseline from outputs/data/triage.csv. Verdict band: locked if',
        'first-shown rate outside [15%, 85%].',
        '',
        '| Model | Role | First (T=0) | First (T=0.7) | Verdict T=0.7 |'
        ' b_lnp T=0.7 (SE) | b_lnp T=0 | R2 T=0.7 / T=0 | n |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for _, r in out.iterrows():
        blogp = (f"{r.get('t07_b_logp', float('nan')):+.2f} "
                 f"({r.get('t07_b_logp_se', float('nan')):.2f})")
        r2s = (f"{r.get('t07_r2', float('nan')):.3f} / "
               f"{r.get('t0_r2', float('nan')):.3f}")
        lines.append(
            f"| {r['model']} | {r['role']} | {r['first_t0']:.1%} | "
            f"{r['first_shown_pooled']:.1%} | {r['verdict_t07']} | {blogp} | "
            f"{r.get('t0_b_logp', float('nan')):+.2f} | {r2s} | "
            f"{int(r['n_valid'])} |")
    lines += ['', f'Coverage: {len(out)}/{len(TEMP_TAGS)} models scored.', '']
    with open(os.path.join(OUT_DIR, 'temp_check.md'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nSaved temp_check.csv / temp_check.md ({len(out)} models)')


if __name__ == '__main__':
    main()
