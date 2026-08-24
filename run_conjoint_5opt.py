"""
Five-option conjoint scorer.

Same card format as run_conjoint_llm.py, extended to Options A-E.
--swap reverses the displayed order (shown A..E = original E..A); recorded
choices are flipped back to the original frame, task_ids offset by +10000.

Usage (from repo root):
    python run_conjoint_5opt.py --model gemma2:9b
    python run_conjoint_5opt.py --model gemma2:9b --swap
Outputs:
    conjoint_results_{model_tag}_5opt[_swap].csv
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from ollama import Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_conjoint_llm import (parse_choice, model_tag, yes_no,
                              TASK_ID_SWAP_OFFSET)

MAX_RETRIES = 3
PREFIXES = ['a', 'b', 'c', 'd', 'e']
LETTERS = ['A', 'B', 'C', 'D', 'E']
FIVE_FLIP = {'A': 'E', 'B': 'D', 'C': 'C', 'D': 'B', 'E': 'A'}

INTRO = ("You are booking a hotel room in New York City for a one-night "
         "stay. You must choose one of the following five options.")
OUTRO = "Which option do you choose? Reply with only the letter A, B, C, D, or E."

CARD = """\
Option {letter}:
  Hotel: {name}
  Star rating: {stars} stars
  Neighborhood: {neighborhood}
  Guest review score: {review_score}/10 ({review_count:,} reviews)
  Room type: {room_type}
  Free cancellation: {cancellation_free}
  Breakfast included: {breakfast_included}
  Key amenities: {amenities}
  Price per night: ${price:,}"""


def option_dict(row, prefix):
    return {
        'name': row[f'{prefix}_name'],
        'stars': int(row[f'{prefix}_stars']),
        'neighborhood': row[f'{prefix}_neighborhood'],
        'review_score': row[f'{prefix}_review_score'],
        'review_count': int(row[f'{prefix}_review_count']),
        'room_type': row[f'{prefix}_room_type'],
        'cancellation_free': yes_no(row[f'{prefix}_cancellation_free']),
        'breakfast_included': yes_no(row[f'{prefix}_breakfast_included']),
        'amenities': row[f'{prefix}_amenities'],
        'price': int(row[f'{prefix}_price']),
    }


def build_prompt(row, swap):
    order = list(reversed(PREFIXES)) if swap else PREFIXES
    cards = [CARD.format(letter=letter, **option_dict(row, prefix))
             for letter, prefix in zip(LETTERS, order)]
    return f'{INTRO}\n\n' + '\n\n'.join(cards) + f'\n\n{OUTRO}'


def main():
    ap = argparse.ArgumentParser(description='Five-option conjoint scorer')
    ap.add_argument('--model', required=True)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_think', action='store_true')
    ap.add_argument('--swap', action='store_true')
    ap.add_argument('--tasks_file', default='tasks_5opt.csv')
    ap.add_argument('--checkpoint_every', type=int, default=10)
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()

    tag = model_tag(args.model)
    suffix = '_swap' if args.swap else ''
    results_file = os.path.join(args.out_dir,
                                f'conjoint_results_{tag}_5opt{suffix}.csv')
    client = Client()

    tasks = pd.read_csv(args.tasks_file)
    if args.swap:
        tasks['task_id'] = tasks['task_id'] + TASK_ID_SWAP_OFFSET

    done_ids = set()
    if os.path.exists(results_file):
        try:
            already = pd.read_csv(results_file)
            valid = already[already['choice'].isin(LETTERS)]
            done_ids = set(valid['task_id'].astype(int).tolist())
        except Exception:
            pass
    to_do = tasks[~tasks['task_id'].isin(done_ids)]
    if len(to_do) == 0:
        print(f'All {len(tasks)} tasks already scored in {results_file}')
        return

    swap_label = ' [SWAPPED]' if args.swap else ''
    print(f'Scoring {len(to_do)} five-option tasks{swap_label} '
          f'(model={args.model}, temp={args.temperature}, seed={args.seed})')

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['task_id', 'choice', 'raw_output']).to_csv(
            results_file, index=False)

    buffer = []
    with tqdm(total=len(to_do), desc=f'5opt{suffix}',
              dynamic_ncols=True) as pbar:
        for _, row in to_do.iterrows():
            prompt = build_prompt(row, swap=args.swap)
            choice, raw = None, ''
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    retry_prompt = prompt if attempt == 1 else (
                        prompt + '\n\nRespond with ONLY one letter: '
                                 'A, B, C, D, or E.')
                    gen_kwargs = dict(
                        model=args.model,
                        prompt=retry_prompt,
                        options={
                            'temperature': args.temperature,
                            'top_p': args.top_p,
                            'top_k': args.top_k,
                            'seed': args.seed,
                            'num_predict': 16,
                        },
                        keep_alive='30m',
                    )
                    if args.no_think:
                        gen_kwargs['think'] = False
                    resp = client.generate(**gen_kwargs)
                    raw = (resp.get('response') or '').strip()
                    choice = parse_choice(raw, tuple(LETTERS))
                    if choice:
                        break
                    print(f'  [RETRY] task_id={row["task_id"]} bad output '
                          f'(attempt {attempt}): {raw[:60]!r}')
                except Exception as e:
                    print(f'  [ERROR] task_id={row["task_id"]} '
                          f'(attempt {attempt}): {e}')
                    raw = ''

            recorded = FIVE_FLIP.get(choice, choice) \
                if (args.swap and choice) else choice
            buffer.append({'task_id': int(row['task_id']),
                           'choice': recorded,
                           'raw_output': raw[:200]})
            if len(buffer) >= args.checkpoint_every:
                pd.DataFrame(buffer).to_csv(results_file, mode='a',
                                            header=False, index=False)
                buffer.clear()
            pbar.update(1)

    if buffer:
        pd.DataFrame(buffer).to_csv(results_file, mode='a',
                                    header=False, index=False)
    print(f'\nDone. Results saved to {results_file}')


if __name__ == '__main__':
    main()
