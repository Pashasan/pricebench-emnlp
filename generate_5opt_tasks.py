"""
Generate 300 five-option choice tasks for the five-option robustness check.

Same pool, same attribute re-randomization as generate_conjoint_tasks.py
(imported directly so the logic cannot drift), new seed 2027 so quintets are
sampled independently of the main study's pairs/triples. One rep per quintet;
position counterbalancing at scoring time via full reversal (--swap).

Usage (from repo root):
    python generate_5opt_tasks.py
Output:
    tasks_5opt.csv   (task_type='fiveopt', prefixes a_..e_)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_conjoint_tasks import (load_hotel_pool, draw_variable_attributes,
                                     OPTION_KEYS)

N_QUINTETS = 300
SEED = 2027
PREFIXES = ['a', 'b', 'c', 'd', 'e']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--n', type=int, default=N_QUINTETS)
    ap.add_argument('--output', default='tasks_5opt.csv')
    args = ap.parse_args()

    hotels = load_hotel_pool()
    rng = np.random.default_rng(args.seed)
    n_hotels = len(hotels)

    quintets, seen = [], set()
    while len(quintets) < args.n:
        q = tuple(sorted(rng.choice(n_hotels, size=5, replace=False).tolist()))
        if q not in seen:
            seen.add(q)
            quintets.append(q)

    tasks = []
    for task_id, q in enumerate(quintets, start=1):
        row = {'task_id': task_id, 'task_type': 'fiveopt',
               'group_id': task_id, 'rep': 1}
        for prefix, idx in zip(PREFIXES, q):
            attrs = draw_variable_attributes(hotels[idx], rng)
            for key in OPTION_KEYS:
                row[f'{prefix}_{key}'] = attrs[key]
        tasks.append(row)

    df = pd.DataFrame(tasks)

    # Validation: unique hotels within each task, prices within bounds
    hotel_by_id = {h['id']: h for h in hotels}
    for _, row in df.iterrows():
        ids = [int(row[f'{p}_hotel_id']) for p in PREFIXES]
        assert len(set(ids)) == 5, f'duplicate hotel in task {row["task_id"]}'
        for p in PREFIXES:
            h = hotel_by_id[int(row[f'{p}_hotel_id'])]
            assert h['price_min'] <= row[f'{p}_price'] <= h['price_max']

    df.to_csv(args.output, index=False, encoding='utf-8')
    n_hotels_used = len({int(r[f'{p}_hotel_id'])
                         for _, r in df.iterrows() for p in PREFIXES})
    print(f'Generated {len(df)} five-option tasks -> {args.output} '
          f'(seed={args.seed}); hotels used: {n_hotels_used}/{n_hotels}')


if __name__ == '__main__':
    main()
