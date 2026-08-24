"""
Prompt-format ablation scorer.

Scores binary conjoint tasks under four prompt variants, keeping the
estimation-relevant structure identical to run_conjoint_llm.py (same tasks,
same swap/counterbalancing logic, same output format):

  json    - options rendered as JSON records instead of prose cards
  reorder - same prose card, attribute rows in a different fixed order
  persona - mild travel-assistant-for-a-client persona wrapper
  cot     - think step by step, then 'Final answer: X' (parsed strictly)

Backends: ollama (local) or openrouter (OpenAI-compatible, needs
OPENROUTER_API_KEY).

Usage (from repo root):
    python run_conjoint_variants.py --model gemma2:9b --variant json \
        --tasks_file tasks_binary_r2.csv
    python run_conjoint_variants.py --model openai/gpt-5.4-nano \
        --backend openrouter --variant cot --tasks_file tasks_binary_r2.csv --swap

Outputs (repo root, matching the existing naming convention):
    conjoint_results_{model_tag}_{variant}[_swap].csv
"""

import os
import re
import sys
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_conjoint_llm import (parse_choice, model_tag, yes_no,
                              BINARY_FLIP, TASK_ID_SWAP_OFFSET)

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, API backoff base


# ── Variant templates ────────────────────────────────────────────────────────
# Baseline card order (run_conjoint_llm.py): name, stars, neighborhood,
# review, room, cancellation, breakfast, amenities, price.

INTRO_NEUTRAL = ("You are booking a hotel room in New York City for a "
                 "one-night stay. You must choose between the following "
                 "two options.")
INTRO_PERSONA = ("You are a travel assistant booking a hotel room in New York "
                 "City on behalf of a client for a one-night stay. Your "
                 "client trusts you to make the best choice for them. You "
                 "must choose between the following two options.")
OUTRO_LETTER = "Which option do you choose? Reply with only the letter A or B."
OUTRO_COT = ("Which option do you choose? Think through the trade-offs step "
             "by step in at most a few sentences, then give your final answer "
             "on a new line in exactly the form 'Final answer: A' or "
             "'Final answer: B'.")

CARD_BASELINE = """\
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

CARD_REORDER = """\
Option {letter}:
  Hotel: {name}
  Price per night: ${price:,}
  Room type: {room_type}
  Breakfast included: {breakfast_included}
  Free cancellation: {cancellation_free}
  Key amenities: {amenities}
  Guest review score: {review_score}/10 ({review_count:,} reviews)
  Neighborhood: {neighborhood}
  Star rating: {stars} stars"""


def option_dict(row, prefix):
    """Extract one option's attributes from a task row."""
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


def json_card(opt, letter):
    """One option as a compact JSON record (same information as the card)."""
    rec = {
        'option': letter,
        'hotel': opt['name'],
        'star_rating': opt['stars'],
        'neighborhood': opt['neighborhood'],
        'guest_review_score': f"{opt['review_score']}/10",
        'review_count': opt['review_count'],
        'room_type': opt['room_type'],
        'free_cancellation': opt['cancellation_free'] == 'Yes',
        'breakfast_included': opt['breakfast_included'] == 'Yes',
        'key_amenities': opt['amenities'],
        'price_per_night_usd': opt['price'],
    }
    return json.dumps(rec, ensure_ascii=False)


def build_prompt(row, variant, swap):
    """Build the variant prompt. swap shows original B first (as Option A)."""
    first, second = ('b', 'a') if swap else ('a', 'b')
    opt1 = option_dict(row, first)   # shown as Option A
    opt2 = option_dict(row, second)  # shown as Option B

    if variant == 'json':
        body = (json_card(opt1, 'A') + '\n' + json_card(opt2, 'B'))
        intro = INTRO_NEUTRAL + (' Each option is given as a JSON record.')
        outro = OUTRO_LETTER
    elif variant == 'reorder':
        body = (CARD_REORDER.format(letter='A', **opt1) + '\n\n'
                + CARD_REORDER.format(letter='B', **opt2))
        intro = INTRO_NEUTRAL
        outro = OUTRO_LETTER
    elif variant == 'persona':
        body = (CARD_BASELINE.format(letter='A', **opt1) + '\n\n'
                + CARD_BASELINE.format(letter='B', **opt2))
        intro = INTRO_PERSONA
        outro = OUTRO_LETTER
    elif variant == 'cot':
        body = (CARD_BASELINE.format(letter='A', **opt1) + '\n\n'
                + CARD_BASELINE.format(letter='B', **opt2))
        intro = INTRO_NEUTRAL
        outro = OUTRO_COT
    else:
        raise ValueError(f'unknown variant {variant!r}')

    return f'{intro}\n\n{body}\n\n{outro}'


def parse_cot(text):
    """Parse a CoT response: prefer the last 'Final answer: X', then fall
    back to the last standalone A/B token, then to parse_choice."""
    if not isinstance(text, str) or not text.strip():
        return None
    matches = re.findall(r'final\s+answer\s*[:\-]?\s*\(?\**([AB])\b',
                         text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    tokens = re.findall(r'\b(?:option\s+)?([AB])\b', text, re.IGNORECASE)
    if tokens:
        return tokens[-1].upper()
    return parse_choice(text, ('A', 'B'))


# ── Backends ─────────────────────────────────────────────────────────────────

def make_generate(args):
    """Return generate(prompt, max_tokens) -> raw text for chosen backend."""
    if args.backend == 'ollama':
        from ollama import Client
        client = Client()

        def generate(prompt, max_tokens):
            gen_kwargs = dict(
                model=args.model,
                prompt=prompt,
                options={
                    'temperature': args.temperature,
                    'top_p': args.top_p,
                    'top_k': args.top_k,
                    'seed': args.seed,
                    'num_predict': max_tokens,
                },
                keep_alive='30m',
            )
            if args.no_think:
                gen_kwargs['think'] = False
            resp = client.generate(**gen_kwargs)
            return (resp.get('response') or '').strip()
        return generate

    elif args.backend == 'openrouter':
        from openai import OpenAI
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise SystemExit('ERROR: OPENROUTER_API_KEY not set')
        client = OpenAI(api_key=api_key,
                        base_url='https://openrouter.ai/api/v1')

        def generate(prompt, max_tokens):
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=args.temperature,
                seed=args.seed,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or '').strip()
        return generate

    raise ValueError(f'unknown backend {args.backend!r}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Prompt-variant conjoint scorer')
    ap.add_argument('--model', required=True)
    ap.add_argument('--variant', required=True,
                    choices=['json', 'reorder', 'persona', 'cot'])
    ap.add_argument('--backend', default='ollama',
                    choices=['ollama', 'openrouter'])
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--top_k', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_think', action='store_true',
                    help='Disable thinking for reasoning models (ollama)')
    ap.add_argument('--swap', action='store_true')
    ap.add_argument('--tasks_file', default='tasks_binary_r2.csv')
    ap.add_argument('--checkpoint_every', type=int, default=10)
    ap.add_argument('--out_dir', default='.',
                    help='Directory for the results CSV (default: cwd)')
    args = ap.parse_args()

    max_tokens = 512 if args.variant == 'cot' else 16
    parse = parse_cot if args.variant == 'cot' else (
        lambda t: parse_choice(t, ('A', 'B')))

    tag = model_tag(args.model)
    suffix = '_swap' if args.swap else ''
    results_file = os.path.join(
        args.out_dir, f'conjoint_results_{tag}_{args.variant}{suffix}.csv')

    tasks = pd.read_csv(args.tasks_file)
    tasks = tasks[tasks['task_type'] == 'binary'].copy()
    if args.swap:
        tasks['task_id'] = tasks['task_id'] + TASK_ID_SWAP_OFFSET

    # Resume support (same convention as run_conjoint_llm.py)
    done_ids = set()
    if os.path.exists(results_file):
        try:
            already = pd.read_csv(results_file)
            valid = already[already['choice'].isin(['A', 'B'])]
            done_ids = set(valid['task_id'].astype(int).tolist())
        except Exception:
            pass
    to_do = tasks[~tasks['task_id'].isin(done_ids)]
    if len(to_do) == 0:
        print(f'All {len(tasks)} tasks already scored in {results_file}')
        return

    generate = make_generate(args)
    swap_label = ' [SWAPPED]' if args.swap else ''
    print(f'Scoring {len(to_do)} tasks{swap_label}  variant={args.variant}  '
          f'(model={args.model}, backend={args.backend}, '
          f'temp={args.temperature}, seed={args.seed})')

    if not os.path.exists(results_file):
        pd.DataFrame(columns=['task_id', 'choice', 'raw_output']).to_csv(
            results_file, index=False)

    buffer = []
    with tqdm(total=len(to_do), desc=f'{args.variant}{suffix}',
              dynamic_ncols=True) as pbar:
        for _, row in to_do.iterrows():
            prompt = build_prompt(row, args.variant, swap=args.swap)
            choice, raw = None, ''
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    retry_prompt = prompt if attempt == 1 else (
                        prompt + '\n\nRespond with ONLY the letter A or B.'
                        if args.variant != 'cot' else
                        prompt + "\n\nEnd your reply with 'Final answer: A' "
                                 "or 'Final answer: B'.")
                    raw = generate(retry_prompt, max_tokens)
                    choice = parse(raw)
                    if choice:
                        break
                    print(f'  [RETRY] task_id={row["task_id"]} bad output '
                          f'(attempt {attempt}): {raw[:80]!r}')
                except Exception as e:
                    print(f'  [ERROR] task_id={row["task_id"]} '
                          f'(attempt {attempt}): {e}')
                    raw = ''
                    if args.backend == 'openrouter':
                        time.sleep(RETRY_DELAY * attempt)

            # Flip back to the original task frame
            recorded = BINARY_FLIP.get(choice, choice) \
                if (args.swap and choice) else choice

            buffer.append({'task_id': int(row['task_id']),
                           'choice': recorded,
                           'raw_output': raw[:500]})
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
