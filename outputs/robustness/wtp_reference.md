# WTP reference-price invariance

Source: `outputs/data/personality.csv` (binary tasks, standard controls). Script: `analysis/wtp_reference.py`.

## Sample

Engaged models: 23. Of these, 20 have both `b_logp_p < 0.001` and `b_d_review_score_p < 0.001` (matches the 20 claimed in the paper).

Engaged but excluded: Gemma3 1B (b_logp_p=0.093, b_rev_p=0.0022); DeepSeek-R1 1.5B (b_logp_p=1e-32, b_rev_p=0.8); Phi-2 2.7B (b_logp_p=0.41, b_rev_p=0.15).

## Structural fact

The log-price WTP is `WTP(ref) = b_review / |b_logp| * ref`. The reference price `ref` is the same multiplicative constant for every model, so it cancels from every between-model ratio and cannot change the ranking. The choice of $250 affects only the dollar units of the reported numbers, not any comparative claim.

Numerical check: rankings under ref = $100, $250, $600 are IDENTICAL.

## Spread at each reference price

| ref | min WTP (model) | max WTP (model) | ratio |
|---|---|---|---|
| $100 | $27 (Gemma2 9B) | $429 (Phi-4 Mini) | 15.6x |
| $250 | $68 (Gemma2 9B) | $1,071 (Phi-4 Mini) | 15.6x |
| $600 | $164 (Gemma2 9B) | $2,571 (Phi-4 Mini) | 15.6x |

The ratio is identical at every reference price, as it must be.

## Cross-form check: linear-price WTP needs no reference price

`wtp_review` in personality.csv is `-b_review_lin / b_linp` from the linear-price logit; it is denominated in dollars directly and involves no reference price.

- Spearman rho (log WTP@250 vs linear WTP) = 0.974 (p = 4e-13), n = 20
- Pearson r = 0.914 (p = 1.8e-08)
- Models moving more than 3 rank positions: 0

- Linear-spec WTP span: $70 (Gemma2 9B) to $2,188 (Phi-4 Mini), ratio 31.4x. Note this differs from the log-spec 16x figure; the wide-spread claim is robust in direction (large heterogeneity) but the exact multiple is specification-dependent.
- AIC-preferred form among the 20 models: linear = 10, log = 9, tie = 1.

## Per-model table

| model | WTP@$100 | WTP@$250 | WTP@$600 | rank (all refs) | linear WTP | linear rank | shift | fit_form |
|---|---|---|---|---|---|---|---|---|
| Phi-4 Mini | 429 | 1,071 | 2,571 | 1 | 2,188 | 1 | +0 | tie |
| Phi-3 Mini | 258 | 646 | 1,550 | 2 | 1,636 | 3 | +1 | log |
| Llama3.2 3B | 210 | 525 | 1,259 | 3 | 1,584 | 4 | +1 | log |
| Qwen3 30B-A3B | 204 | 511 | 1,226 | 4 | 2,073 | 2 | -2 | log |
| GPT-5.4 Nano | 200 | 500 | 1,201 | 5 | 911 | 6 | +1 | linear |
| GPT-4.1 Nano | 167 | 417 | 1,001 | 6 | 1,020 | 5 | -1 | log |
| Gemma3 12B | 164 | 410 | 983 | 7 | 819 | 7 | +0 | linear |
| DeepSeek-R1 7B | 136 | 340 | 816 | 8 | 716 | 9 | +1 | log |
| Gemma3 4B | 135 | 337 | 809 | 9 | 617 | 11 | +2 | linear |
| Phi-3 Medium | 110 | 276 | 661 | 10 | 680 | 10 | +0 | log |
| Qwen3 8B | 105 | 263 | 631 | 11 | 769 | 8 | -3 | log |
| GPT-5.4 Mini | 89 | 222 | 534 | 12 | 350 | 13 | +1 | linear |
| Gemma3 27B | 84 | 210 | 504 | 13 | 346 | 15 | +2 | linear |
| Mistral-Nemo 12B | 81 | 203 | 488 | 14 | 414 | 12 | -2 | log |
| Phi-4 14B | 81 | 202 | 484 | 15 | 348 | 14 | -1 | linear |
| Claude Haiku 4.5 | 55 | 139 | 333 | 16 | 214 | 17 | +1 | linear |
| Qwen3 4B | 51 | 127 | 306 | 17 | 331 | 16 | -1 | log |
| GPT-4.1 Mini | 47 | 119 | 285 | 18 | 153 | 18 | +0 | linear |
| GPT-5.4 | 43 | 107 | 257 | 19 | 115 | 19 | +0 | linear |
| Gemma2 9B | 27 | 68 | 164 | 20 | 70 | 20 | +0 | linear |
