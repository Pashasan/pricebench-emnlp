"""
Prompt-format ablation: are recovered preferences stable across prompt formats?

For each ablation model x variant (json / reorder / persona / cot), fits the
paper's log-price logit and compares coefficients, WTP, and cross-model
orderings against the baseline prompt scored on the SAME task subset (so the
only difference is the prompt format). Tolerates missing files.

Run from the repo root:
    python analysis/analyze_variants.py
Outputs:
    outputs/robustness/variant_ablation.csv
    outputs/robustness/variant_ablation.md
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_common import (ROOT, OUT_DIR, TAG2NAME, load_result_pair,
                               build_binary_dataset, fit_log_spec)
sys.path.insert(0, ROOT)
from config import MODELS  # noqa: E402

# tag -> (display name, subset file)
ABLATION = {
    'qwen3_4b-instruct': ('Qwen3 4B', 'tasks_binary_r2.csv'),
    'gemma2_9b': ('Gemma2 9B', 'tasks_binary_r2.csv'),
    'phi4-mini': ('Phi-4 Mini', 'tasks_binary_r2.csv'),
    'gemma3_27b': ('Gemma3 27B', 'tasks_binary_r1.csv'),
    'openaigpt-5_4-nano': ('GPT-5.4 Nano', 'tasks_binary_r2.csv'),
    'openaigpt-5_4-mini': ('GPT-5.4 Mini', 'tasks_binary_r2.csv'),
}
VARIANTS = ['json', 'reorder', 'persona', 'cot']


def main():
    rows = []
    for tag, (name, subset_file) in ABLATION.items():
        ids = set(pd.read_csv(
            os.path.join(ROOT, subset_file))['task_id'])

        # Baseline: original paper scoring restricted to the same tasks
        orig_f, swap_f = MODELS[name]
        base_res = load_result_pair(os.path.join(ROOT, orig_f),
                                    os.path.join(ROOT, swap_f))
        base_fit = fit_log_spec(build_binary_dataset(
            base_res[base_res['task_id'].isin(ids)], task_ids=ids)) \
            if base_res is not None else None
        if base_fit:
            rows.append(dict(model=name, variant='baseline', **base_fit))

        for v in VARIANTS:
            vres = load_result_pair(
                os.path.join(ROOT, f'conjoint_results_{tag}_{v}.csv'),
                os.path.join(ROOT, f'conjoint_results_{tag}_{v}_swap.csv'))
            if vres is None:
                continue
            fit = fit_log_spec(build_binary_dataset(vres, task_ids=ids))
            if fit:
                # z for difference vs baseline (approximate: same tasks,
                # different prompts, treated as independent -> conservative
                # to the extent errors are positively correlated)
                if base_fit:
                    z = ((fit['b_logp'] - base_fit['b_logp'])
                         / np.sqrt(fit['b_logp_se'] ** 2
                                   + base_fit['b_logp_se'] ** 2))
                    fit['z_vs_base_logp'] = float(z)
                    zr = ((fit['b_review'] - base_fit['b_review'])
                          / np.sqrt(fit['b_review_se'] ** 2
                                    + base_fit['b_review_se'] ** 2))
                    fit['z_vs_base_review'] = float(zr)
                rows.append(dict(model=name, variant=v, **fit))

    if not rows:
        print('No variant results found yet.')
        return
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, 'variant_ablation.csv'), index=False,
               encoding='utf-8')

    # Cross-model ordering stability per variant (vs baseline ordering)
    base = out[out['variant'] == 'baseline'].set_index('model')
    stab_lines = []
    for v in VARIANTS:
        sub = out[out['variant'] == v].set_index('model')
        common = [m for m in sub.index if m in base.index]
        if len(common) >= 4:
            rho_p, _ = spearmanr(base.loc[common, 'b_logp'].abs(),
                                 sub.loc[common, 'b_logp'].abs())
            rho_w, _ = spearmanr(base.loc[common, 'wtp250'],
                                 sub.loc[common, 'wtp250'])
            stab_lines.append(
                f'| {v} | {len(common)} | {rho_p:+.2f} | {rho_w:+.2f} |')

    # Markdown
    lines = [
        '# Prompt-format ablation',
        '',
        'Log-price logit (paper spec) per model x prompt variant, baseline',
        're-estimated on the identical task subset. WTP at $250 reference.',
        '',
        '## Coefficients by model and variant',
        '',
        '| Model | Variant | b_lnp (SE) | b_review (SE) | WTP@250 (SE) |'
        ' R2 | n | z vs base (lnp) |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for _, r in out.iterrows():
        z = r.get('z_vs_base_logp', float('nan'))
        z_str = f'{z:+.2f}' if np.isfinite(z) else '-'
        lines.append(
            f"| {r['model']} | {r['variant']} | "
            f"{r['b_logp']:+.2f} ({r['b_logp_se']:.2f}) | "
            f"{r['b_review']:+.2f} ({r['b_review_se']:.2f}) | "
            f"{r['wtp250']:.0f} ({r.get('wtp250_se', float('nan')):.0f}) | "
            f"{r['r2']:.3f} | {r['n']} | {z_str} |")
    lines += [
        '',
        '## Cross-model ordering stability (Spearman vs baseline ordering)',
        '',
        '| Variant | Models | rho |b_lnp| | rho WTP |',
        '|---|---|---|---|',
        *stab_lines,
        '',
        'Coverage: ' + ', '.join(
            f"{m}: {sorted(out[out['model'] == m]['variant'])}"
            for m in out['model'].unique()),
        '',
    ]
    with open(os.path.join(OUT_DIR, 'variant_ablation.md'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved variant_ablation.csv / .md "
          f"({out['model'].nunique()} models, {len(out)} fits)")


if __name__ == '__main__':
    main()
