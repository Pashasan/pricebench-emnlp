# PriceBench

Replication package for the paper *PriceBench: A Diagnostic Benchmark for
Price, Quality, and Brand Preferences in LLM Booking Agents.*

PriceBench recovers an LLM's price, quality, and brand preferences from its
hotel-booking choices with a logit choice model. This repository contains the
benchmark tasks, the scoring and analysis code, and the complete set of model
response files, so every number and figure in the paper can be reproduced.

## What's here

```
config.py                     single source of truth: model registry, attribute
                              encoding, dataset builders (imported everywhere)
build_hotel_pool.py           constructs hotel_pool.json (179 NYC profiles)
generate_conjoint_tasks.py    builds conjoint_tasks.csv (3,600 tasks, seed 2026)
hotel_pool.json               frozen pool of 179 real NYC hotel profiles
conjoint_tasks.csv            3,600 choice tasks (1,800 binary + 1,800 ternary)

run_conjoint_llm.py           Ollama scorer (open-weight models)
run_conjoint_llm_api.py       OpenAI-compatible API scorer
run_conjoint_claude.py        Anthropic SDK scorer
run_conjoint_variants.py      prompt-format variant scorer (JSON / reorder /
                              persona / chain-of-thought)
run_conjoint_5opt.py          five-option scorer
generate_5opt_tasks.py        builds tasks_5opt.csv (300 five-option tasks,
                              seed 2027)

conjoint_results_*.csv        all model response sets (original + swapped
                              ordering) — the raw data behind every result,
                              including the robustness runs (variant, temp07,
                              and 5opt tags)
tasks_5opt.csv                300 five-option tasks
tasks_binary_r1.csv           450-task binary subset (prompt-variant runs)
tasks_binary_r2.csv           900-task binary subset (prompt-variant runs)
capability_scores.csv         externally sourced capability scores per model,
                              with per-score source URLs

analysis/                     estimation + figure pipeline
  triage.py        first-shown rate per model; engaged vs. position-locked
  pooled.py        pooled binary+ternary conditional logit; pooling LR test
  ternary.py       ternary-only conditional logit (binary-vs-ternary check)
  personality.py   per-model price/quality coefficients and WTP
  brand.py         chain-family coefficients + within-provider permutation test
  booking_outcomes.py  mean booked price per model
  figures.py       renders all figures from the data CSVs
  robustness_common.py  shared helpers for the robustness analyses
  analyze_variants.py   prompt-format ablation (coefficients per variant)
  analyze_temp.py       temperature-0.7 check on locked models + controls
  analyze_5opt.py       five-option out-of-sample prediction
  rank_stability.py     cluster-bootstrap CIs, rank stability, multiplicity
  capability_axis.py    external capability index vs. decisiveness and WTP
  brand_robustness.py   brand specification checks + permutation variants
  wtp_reference.py      reference-price invariance of the WTP ranking
  locked_forensics.py   locked-model response forensics

outputs/data/                     analysis outputs (CSV/JSON) consumed by figures.py
outputs/figures/                  rendered paper figures (PNG, 200 DPI)
outputs/robustness/               reports written by the robustness analyses
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

Re-scoring models requires credentials, supplied via environment variables
(never hardcoded):

```bash
export OPENAI_API_KEY=...       # for run_conjoint_llm_api.py
export ANTHROPIC_API_KEY=...    # for run_conjoint_claude.py
# open-weight models run locally via Ollama (https://ollama.com); start it first
```

See `.env.example`.

## Reproduce the paper

The response files are included, so the full analysis runs without re-scoring
any model:

```bash
python analysis/triage.py          # engaged vs. position-locked verdict
python analysis/pooled.py          # pooled conditional logit + decile curves
python analysis/ternary.py         # binary-vs-ternary robustness
python analysis/personality.py     # price/quality coefficients, WTP
python analysis/brand.py           # brand coefficients + permutation test
python analysis/booking_outcomes.py
python analysis/figures.py         # all figures -> outputs/figures/
```

Each script reads `config.py` and iterates over the model registry
automatically; outputs land in `outputs/data/` and `outputs/figures/`.

## Robustness experiments

The response files for the paper's robustness section are included, so these
also run without re-scoring (reports land in `outputs/robustness/`):

```bash
python analysis/analyze_variants.py   # prompt-format ablation
python analysis/analyze_temp.py       # temperature 0.7 on locked models
python analysis/analyze_5opt.py       # five-option out-of-sample check
python analysis/rank_stability.py     # bootstrap CIs, rank stability, multiplicity
python analysis/capability_axis.py    # external capability index correlations
python analysis/brand_robustness.py   # brand specification + permutation checks
python analysis/wtp_reference.py      # WTP reference-price invariance
python analysis/locked_forensics.py   # locked-model response forensics
```

To re-score the robustness runs themselves:

```bash
# prompt-format variants (json | reorder | persona | cot), both orderings
python run_conjoint_variants.py --model gemma2:9b --variant json --tasks_file tasks_binary_r1.csv
python run_conjoint_variants.py --model gemma2:9b --variant json --tasks_file tasks_binary_r1.csv --swap

# temperature 0.7 (binary block)
python run_conjoint_llm.py --model llama3.1:8b --temperature 0.7 --top_p 0.95 --top_k 40 --tag temp07 --task_type binary
python run_conjoint_llm.py --model llama3.1:8b --temperature 0.7 --top_p 0.95 --top_k 40 --tag temp07 --task_type binary --swap

# five-option tasks
python generate_5opt_tasks.py
python run_conjoint_5opt.py --model gemma2:9b
python run_conjoint_5opt.py --model gemma2:9b --swap
```

## Re-score a model from scratch

Each runner writes a pair of CSVs — original ordering and reversed ordering
(`--swap`, task IDs offset by +10000):

```bash
# open-weight via Ollama
python run_conjoint_llm.py --model gemma3:latest
python run_conjoint_llm.py --model gemma3:latest --swap

# OpenAI-compatible API
python run_conjoint_llm_api.py --model gpt-4.1-nano
python run_conjoint_llm_api.py --model gpt-4.1-nano --swap

# Anthropic
python run_conjoint_claude.py --model claude-haiku-4-5
python run_conjoint_claude.py --model claude-haiku-4-5 --swap
```

To register a new model, add one entry to the `MODELS` dict in `config.py`
mapping the display name to its `(original, swap)` CSV pair, then re-run the
analysis pipeline above.

## Data notes

- The hotel pool is a frozen research artifact derived from public OTA listings,
  with prices and review fields re-randomized within each property's own listed
  range (price uniform in range; review score within ±0.2; review count within
  ±10%) so each presentation is a plausible real listing.
- All main runs use temperature 0 and greedy decoding; the temperature-0.7
  robustness runs are tagged `temp07`.
- Decile cut-points are computed once on the pooled price distribution and
  reused across models for comparability.
- The Qwen3 1.7B response pair is retained for completeness but that run
  failed (no valid choices); the paper's 28-model set excludes it.

See `DATASHEET.md` for full dataset documentation (motivation, composition,
collection, preprocessing, uses, distribution, maintenance).

## Citation

If you use the benchmark, tasks, or response data, please cite:

```bibtex
@inproceedings{kireyev2026pricebench,
  author    = {Pavel Kireyev},
  title     = {{PriceBench}: A Diagnostic Benchmark for Price, Quality, and
               Brand Preferences in {LLM} Booking Agents},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing: Industry Track},
  year      = {2026},
  address   = {Budapest, Hungary},
}
```

## License

Code is released under the MIT License (`LICENSE`). The task and response data
are released for research use; please cite the paper if you use them.
