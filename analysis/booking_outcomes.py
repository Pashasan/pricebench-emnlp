"""
Per-model predicted-booking-outcomes table for the EMNLP paper.

For each model in MODELS, compute over the 1,800 binary tasks
(scored twice = 3,600 observations) the average booked properties:
  - mean chosen price ($/night)
  - mean chosen stars
  - share of chosen options by chain family
This is the operational dashboard view: 'if you deploy this model as
the booking agent, what booking mix does it produce on our tasks?'

Output: outputs/data/booking_outcomes.csv
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from config import MODELS, load_choices, TASKS_FILE
from analysis.brand import get_brand

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

tasks = pd.read_csv(os.path.join(BASE, TASKS_FILE))
binary = tasks[tasks['task_type'] == 'binary'].copy()

rows = []
for name in MODELS.keys():
    try:
        choices = load_choices(name)
    except FileNotFoundError:
        continue
    df = choices.merge(binary, on='task_id', how='inner')
    df = df[df['choice'].isin(['A', 'B'])].copy()
    if len(df) == 0:
        continue

    a = df['choice'] == 'A'
    chosen_price = np.where(a, df['a_price'], df['b_price'])
    chosen_stars = np.where(a, df['a_stars'], df['b_stars'])
    chosen_name = np.where(a, df['a_name'], df['b_name'])
    chosen_brand = [get_brand(n) for n in chosen_name]

    row = dict(
        model=name,
        n_obs=len(df),
        mean_price=float(np.mean(chosen_price)),
        median_price=float(np.median(chosen_price)),
        mean_stars=float(np.mean(chosen_stars)),
    )
    chain_share = pd.Series(chosen_brand).value_counts(normalize=True) * 100
    for brand in ['Independent', 'Hilton', 'Marriott', 'IHG', 'Hyatt', 'Wyndham']:
        row[f'pct_{brand}'] = float(chain_share.get(brand, 0.0))
    rows.append(row)

out = pd.DataFrame(rows)
out_path = os.path.join(BASE, 'outputs', 'data', 'booking_outcomes.csv')
out.to_csv(out_path, index=False)
print(f'Wrote {out_path} with {len(out)} rows.')

cols = ['model', 'mean_price', 'mean_stars', 'pct_Independent',
        'pct_Wyndham', 'pct_Marriott', 'pct_Hilton']
print('\nBooking outcomes per model (binary tasks pooled):')
print(out[cols].to_string(index=False, float_format=lambda x: f'{x:6.1f}'))
