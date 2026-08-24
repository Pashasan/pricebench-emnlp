"""
Five-option analysis: does the binary-recovered policy predict five-option
choices?

For each model scored on the 300 five-option tasks (both orderings):
  1. Position use: distribution of the chosen option's SHOWN slot (1-5),
     per ordering and pooled (lock persistence in longer lists).
  2. Out-of-sample prediction: utilities from the paper's binary log-price
     logit (full binary block) applied to the five options; report hit rate
     (chance 20%), top-2 rate (chance 40%), mean predicted rank of the
     chosen option (chance 3.0).
  3. Within five-option conditional logit (log-price spec, groups = task x
     ordering) to compare b_lnp with the binary estimate.

Run from project root:
    python analysis/analyze_5opt.py
Outputs:
    outputs/robustness/fiveopt_results.csv
    outputs/robustness/fiveopt_results.md
"""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_common import (ROOT, OUT_DIR, TAG2NAME, load_result_pair,
                               fit_log_spec, CTRLS)
sys.path.insert(0, ROOT)
from config import MODELS, build_dataset, extract_bed_size  # noqa: E402

FIVE_TAGS = ['gemma2_9b', 'qwen3_4b-instruct', 'phi4-mini', 'mistral-nemo',
             'llama3_1_8b']
LETTERS = ['A', 'B', 'C', 'D', 'E']
PREFIXES = ['a', 'b', 'c', 'd', 'e']
CONTROLS_LONG = ['stars', 'bed_rank', 'cancel', 'breakfast',
                 'review_score', 'review_count_100']


def option_features(t, p):
    return dict(
        log_price=float(np.log(t[f'{p}_price'])),
        stars=float(t[f'{p}_stars']),
        bed_rank=float(extract_bed_size(t[f'{p}_room_type'])),
        cancel=float(t[f'{p}_cancellation_free']),
        breakfast=float(t[f'{p}_breakfast_included']),
        review_score=float(t[f'{p}_review_score']),
        review_count_100=float(t[f'{p}_review_count']) / 100,
    )


BETA_MAP = {  # binary difference-spec coef -> long-format feature
    'd_log_price': 'log_price', 'd_stars': 'stars', 'd_bed': 'bed_rank',
    'd_cancel': 'cancel', 'd_breakfast': 'breakfast',
    'd_review_score': 'review_score', 'd_review_count_100': 'review_count_100',
}


def binary_betas(name):
    """Refit the paper's binary log-price logit, return coef dict."""
    import statsmodels.api as sm
    df = build_dataset(name, task_type='binary')
    X = sm.add_constant(df[CTRLS + ['d_log_price']])
    m = sm.Logit(df['choice_A'], X).fit(disp=False, maxiter=200)
    return {BETA_MAP[k]: float(m.params[k]) for k in BETA_MAP}, \
        float(m.params['d_log_price'])


def main():
    tasks = pd.read_csv(os.path.join(ROOT, 'tasks_5opt.csv'))
    tasks = tasks.set_index('task_id')

    rows = []
    for tag in FIVE_TAGS:
        name = TAG2NAME[tag]
        res = load_result_pair(
            os.path.join(ROOT, f'conjoint_results_{tag}_5opt.csv'),
            os.path.join(ROOT, f'conjoint_results_{tag}_5opt_swap.csv'))
        if res is None:
            print(f'  {name:<18s} no 5opt data yet, skipping')
            continue
        res = res[res['choice'].isin(LETTERS)].copy()

        # 1. shown-slot distribution of the chosen option
        letter_idx = {l: i + 1 for i, l in enumerate(LETTERS)}   # orig frame
        res['orig_pos'] = res['choice'].map(letter_idx)
        res['shown_slot'] = np.where(res['ordering'] == 0, res['orig_pos'],
                                     6 - res['orig_pos'])
        slot_dist = (res['shown_slot'].value_counts(normalize=True)
                     .reindex(range(1, 6)).fillna(0))
        row = dict(model=name, n_valid=int(len(res)),
                   **{f'slot{i}': float(slot_dist[i]) for i in range(1, 6)})
        row['first_slot_rate'] = float(slot_dist[1])

        # 2. out-of-sample prediction from binary betas
        try:
            betas, b_logp_bin = binary_betas(name)
        except Exception as e:
            betas, b_logp_bin = None, float('nan')
            print(f'  {name}: binary refit failed: {e}')
        if betas:
            hits, top2, ranks = [], [], []
            for _, r in res.iterrows():
                t = tasks.loc[int(r['task_id'])]
                utils = {}
                for letter, p in zip(LETTERS, PREFIXES):
                    feats = option_features(t, p)
                    utils[letter] = sum(betas[k] * feats[k] for k in betas)
                order = sorted(utils, key=utils.get, reverse=True)
                rank = order.index(r['choice']) + 1
                ranks.append(rank)
                hits.append(rank == 1)
                top2.append(rank <= 2)
            row['hit_rate'] = float(np.mean(hits))
            row['top2_rate'] = float(np.mean(top2))
            row['mean_rank'] = float(np.mean(ranks))
            row['b_logp_binary'] = b_logp_bin

        # 3. within five-option conditional logit
        long_rows = []
        for _, r in res.iterrows():
            t = tasks.loc[int(r['task_id'])]
            gid = int(r['task_id']) * 2 + int(r['ordering'])
            for letter, p in zip(LETTERS, PREFIXES):
                long_rows.append(dict(
                    group_id=gid, chosen=int(r['choice'] == letter),
                    **option_features(t, p)))
        ldf = pd.DataFrame(long_rows)
        try:
            X = ldf[CONTROLS_LONG + ['log_price']].astype(float)
            m5 = ConditionalLogit(ldf['chosen'], X,
                                  groups=ldf['group_id']).fit(
                disp=False, method='bfgs', maxiter=500)
            row['b_logp_5opt'] = float(m5.params['log_price'])
            row['b_logp_5opt_se'] = float(m5.bse['log_price'])
            row['b_review_5opt'] = float(m5.params['review_score'])
        except Exception as e:
            print(f'  {name}: 5opt clogit failed: {e}')

        rows.append(row)
        print(f'  {name:<18s} slot1={row["first_slot_rate"]:.1%}  '
              f'hit={row.get("hit_rate", float("nan")):.1%}  '
              f'mean_rank={row.get("mean_rank", float("nan")):.2f}  '
              f'b_lnp 5opt={row.get("b_logp_5opt", float("nan")):+.2f} '
              f'vs bin={row.get("b_logp_binary", float("nan")):+.2f}')

    if not rows:
        print('No 5opt results found yet.')
        return
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, 'fiveopt_results.csv'), index=False,
               encoding='utf-8')

    lines = [
        '# Five-option tasks',
        '',
        '300 quintets from the same pool (seed 2027), both orderings (full',
        'reversal), temperature 0. Prediction uses the paper binary log-price',
        'logit utilities; chance: hit 20%, top-2 40%, mean rank 3.0.',
        '',
        '| Model | n | Slot1 | Slot2 | Slot3 | Slot4 | Slot5 | Hit | Top-2 |'
        ' Mean rank | b_lnp 5opt (SE) | b_lnp binary |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for _, r in out.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['n_valid'])} | "
            + ' | '.join(f"{r[f'slot{i}']:.1%}" for i in range(1, 6))
            + f" | {r.get('hit_rate', float('nan')):.1%} | "
            f"{r.get('top2_rate', float('nan')):.1%} | "
            f"{r.get('mean_rank', float('nan')):.2f} | "
            f"{r.get('b_logp_5opt', float('nan')):+.2f} "
            f"({r.get('b_logp_5opt_se', float('nan')):.2f}) | "
            f"{r.get('b_logp_binary', float('nan')):+.2f} |")
    lines.append('')
    with open(os.path.join(OUT_DIR, 'fiveopt_results.md'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nSaved fiveopt_results.csv / .md ({len(out)} models)')


if __name__ == '__main__':
    main()
