"""
Shared helpers for the robustness analysis scripts.

Loads result CSVs produced by the robustness runners without touching the
config.MODELS registry, then reuses config's difference-coding machinery so
the estimation spec is byte-identical to the paper's.
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import TASKS_FILE, TASK_ID_SWAP_OFFSET, _build_binary  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR = os.path.join(ROOT, 'outputs', 'robustness')

CTRLS = ['d_stars', 'd_bed', 'd_cancel', 'd_breakfast',
         'd_review_score', 'd_review_count_100']

# result-file tag -> paper display name
TAG2NAME = {
    'llama3_1_8b': 'Llama3.1 8B',
    'llama3_8b-instruct-fp16': 'Llama3 8B',
    'mistral_7b-instruct-fp16': 'Mistral 7B',
    'qwen3_0_6b': 'Qwen3 0.6B',
    'llama3_2_1b': 'Llama3.2 1B',
    'gemma3_latest': 'Gemma3 4B',
    'qwen3_4b-instruct': 'Qwen3 4B',
    'gemma2_9b': 'Gemma2 9B',
    'phi4-mini': 'Phi-4 Mini',
    'gemma3_27b': 'Gemma3 27B',
    'mistral-nemo': 'Mistral-Nemo 12B',
    'openaigpt-5_4-nano': 'GPT-5.4 Nano',
    'openaigpt-5_4-mini': 'GPT-5.4 Mini',
}


def load_result_pair(orig_path, swap_path):
    """Concatenate an orig+swap result pair; swap ids un-offset.
    Missing files are skipped (returns whatever exists, or None)."""
    frames = []
    for path, is_swap in [(orig_path, False), (swap_path, True)]:
        if os.path.exists(path):
            r = pd.read_csv(path)
            if is_swap:
                r = r.copy()
                r['task_id'] = r['task_id'] - TASK_ID_SWAP_OFFSET
            r['ordering'] = int(is_swap)
            frames.append(r)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build_binary_dataset(results, task_ids=None):
    """Merge results with the binary tasks and difference-code (paper spec)."""
    tasks = pd.read_csv(TASKS_FILE)
    tasks = tasks[tasks['task_type'] == 'binary']
    if task_ids is not None:
        tasks = tasks[tasks['task_id'].isin(task_ids)]
    df = results.merge(tasks, on='task_id', how='inner')
    df = df[df['choice'].isin(['A', 'B'])]
    return _build_binary(df.copy())


def fit_log_spec(df):
    """Paper's log-price logit. Returns dict of key coefficients or None."""
    if df is None or len(df) < 50 or df['choice_A'].std() < 1e-6:
        return None
    X = sm.add_constant(df[CTRLS + ['d_log_price']])
    try:
        m = sm.Logit(df['choice_A'], X).fit(disp=False, maxiter=200)
    except Exception:
        return None
    b_logp = m.params['d_log_price']
    b_rev = m.params['d_review_score']
    out = dict(
        n=int(m.nobs), r2=float(m.prsquared),
        b_logp=float(b_logp), b_logp_se=float(m.bse['d_log_price']),
        b_logp_p=float(m.pvalues['d_log_price']),
        b_review=float(b_rev), b_review_se=float(m.bse['d_review_score']),
        b_review_p=float(m.pvalues['d_review_score']),
        wtp250=float(b_rev / abs(b_logp) * 250) if abs(b_logp) > 1e-9
        else float('nan'),
    )
    # delta-method SE for the WTP ratio (independence approximation)
    if abs(b_logp) > 1e-9:
        r = b_rev / b_logp
        var = (r ** 2) * ((m.bse['d_review_score'] / b_rev) ** 2
                          + (m.bse['d_log_price'] / b_logp) ** 2) \
            if abs(b_rev) > 1e-9 else float('nan')
        out['wtp250_se'] = float(abs(250 * np.sqrt(var))) \
            if np.isfinite(var) else float('nan')
    else:
        out['wtp250_se'] = float('nan')
    return out


def first_shown_rates(results):
    """P(pick first-shown) per ordering and pooled, plus valid rate.
    Recorded choices are always in the original frame: first-shown is 'A' in
    the original ordering and 'B' in the swapped ordering."""
    res = results[results['task_id'] <= 1800] if results is not None else None
    if res is None or len(res) == 0:
        return None
    out = {}
    valid = res[res['choice'].isin(['A', 'B'])]
    out['n_scored'] = int(len(res))
    out['valid_rate'] = float(len(valid) / len(res)) if len(res) else np.nan
    first_flags = []
    for ordering, first_letter in [(0, 'A'), (1, 'B')]:
        sub = valid[valid['ordering'] == ordering]
        rate = float((sub['choice'] == first_letter).mean()) if len(sub) \
            else np.nan
        out[f'first_shown_ord{ordering}'] = rate
        first_flags.append((sub['choice'] == first_letter))
    pooled = pd.concat(first_flags) if first_flags else pd.Series(dtype=float)
    out['first_shown_pooled'] = float(pooled.mean()) if len(pooled) else np.nan
    out['n_valid'] = int(len(valid))
    return out
