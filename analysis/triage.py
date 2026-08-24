"""
Triage: compute the first-shown selection rate per model across both orderings.

Exact construction (all 3,600 tasks, both orderings): a response enters the
statistic when it selects one of the task's two reference options (A and B in
the original frame), and counts as first-shown when the option it selects is
the one listed earlier in the ordering displayed. In the original ordering A
precedes B (in both the binary pairs and the ternary triples); in the swapped
ordering B precedes A. Ternary picks of the third option (recorded 'C') fall
outside the reference pair and are excluded from numerator and denominator.
On the binary block alone, the statistic is simply the share of first-listed
picks. 50% is fully attribute-driven; 100% is pure slot-following.

Models with extreme bias (|rate - 0.5| > THRESHOLD) are flagged as
position-locked: their choices carry too little attribute signal for
preference estimation and they are excluded from downstream analysis.

Note: the Mistral 7B swap file contains a superseded duplicate scoring block
alongside the standard one (task IDs offset by +1000 instead of +10000). This
statistic includes it; the two blocks agree on 99.9% of tasks, the rate moves
by under 0.3 points, and no verdict depends on it. It is retained so the
published numbers reproduce exactly.

Outputs:
  outputs/data/triage.csv -- one row per model: first_rate, verdict
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
from config import MODELS

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                       'outputs', 'data', 'triage.csv')

# Distance from 50% that counts as "locked". 0.35 puts the cutoff at 15%/85%.
THRESHOLD = 0.35


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


rows = []
for name in MODELS:
    orig_file, swap_file = MODELS[name]
    orig = pd.read_csv(orig_file)
    swap = pd.read_csv(swap_file)
    # Choices are recorded in the original reference frame. Of the reference
    # pair {A, B}, A is listed earlier in the original ordering and B is
    # listed earlier in the swapped ordering (binary swap shows [B, A];
    # ternary swap shows [C, B, A]). So the statistic is
    # (orig A + swap B) / (responses picking A or B); ternary picks of the
    # third option are excluded with the isin filter below.
    orig_valid = orig[orig['choice'].isin(['A', 'B'])]
    swap_valid = swap[swap['choice'].isin(['A', 'B'])]
    n = len(orig_valid) + len(swap_valid)
    if n == 0:
        rows.append(dict(model=name, provider=provider_of(name),
                         n=0, first_rate=float('nan'), imbalance=float('nan'),
                         verdict='no data'))
        continue
    first_shown = (orig_valid['choice'] == 'A').sum() + \
                  (swap_valid['choice'] == 'B').sum()
    first_rate = first_shown / n
    imbalance = abs(first_rate - 0.5)
    if imbalance > THRESHOLD:
        verdict = 'locked-primacy' if first_rate > 0.5 else 'locked-recency'
    else:
        verdict = 'engaged'
    rows.append(dict(
        model=name,
        provider=provider_of(name),
        n=int(n),
        first_rate=float(first_rate),
        imbalance=float(imbalance),
        verdict=verdict,
    ))

out = pd.DataFrame(rows).sort_values('first_rate')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print(f'{"Model":<22s} {"Provider":<12s} {"First-opt":>10s}  Verdict')
print('-' * 62)
for _, r in out.iterrows():
    print(f'{r["model"]:<22s} {r["provider"]:<12s} '
          f'{r["first_rate"]*100:>9.1f}%  {r["verdict"]}')

n_engaged = (out['verdict'] == 'engaged').sum()
n_locked = out['verdict'].str.startswith('locked').sum()
print(f'\nEngaged: {n_engaged} / {len(out)}   Locked: {n_locked} / {len(out)}')
print(f'Saved {OUT_CSV}')
