# Rank stability, bootstrap CIs, multiplicity

Cluster bootstrap over the 450 unique binary group_id values, B=500, numpy seed 42; identical group draws applied to every model so per-draw rankings are comparable. Log-price logit spec: CTRLS + d_log_price (the paper spec). WTP = b_review / |b_logp| x 250.

- Engaged models: 23; well-identified screen (b_logp_p < .001 and b_d_review_score_p < .001): 20 models.
- Failed / non-converged bootstrap fits: 0 of 11500.
- Draws with all 23 engaged fits valid: 500; draws with all 20 screen fits valid: 500.

## Per-model 95% percentile CIs and rank CIs

Sorted by point-estimate |b_logp| rank (1 = most price sensitive). WTP ranks are over the 20-model screen (1 = highest WTP). Full detail in rank_stability_cis.csv.

| Model | b_logp [95% CI] | WTP [95% CI] | rank |b_logp| [95% CI] | rank WTP [95% CI] |
|---|---|---|---|---|
| GPT-5.4 | -7.02 [-8.08, -6.29] | 107 [94, 121] | 1 [1, 1] | 19 [18, 19] |
| Gemma2 9B | -5.71 [-6.24, -5.27] | 68 [59, 78] | 2 [2, 3] | 20 [20, 20] |
| GPT-4.1 Mini | -5.68 [-6.35, -5.15] | 119 [108, 131] | 3 [2, 3] | 18 [17, 19] |
| Claude Haiku 4.5 | -4.75 [-5.28, -4.31] | 139 [122, 157] | 4 [4, 6] | 16 [16, 17] |
| GPT-5.4 Mini | -4.56 [-5.18, -4.11] | 222 [203, 246] | 5 [4, 6] | 12 [11, 14] |
| Gemma3 27B | -4.38 [-4.91, -3.92] | 210 [191, 233] | 6 [4, 6] | 13 [12, 15] |
| Phi-4 14B | -3.27 [-3.69, -2.94] | 202 [181, 228] | 7 [7, 7] | 15 [13, 15] |
| GPT-5.4 Nano | -2.47 [-2.86, -2.13] | 500 [446, 572] | 8 [8, 10] | 5 [3, 5] |
| GPT-4.1 Nano | -2.42 [-2.77, -2.12] | 417 [377, 469] | 9 [8, 10] | 6 [5, 8] |
| Phi-3 Medium | -2.18 [-2.48, -1.94] | 276 [240, 309] | 10 [9, 11] | 10 [10, 11] |
| Gemma3 12B | -2.15 [-2.45, -1.92] | 410 [367, 461] | 11 [9, 11] | 7 [5, 8] |
| Qwen3 4B | -1.59 [-1.80, -1.41] | 127 [109, 147] | 12 [12, 13] | 17 [16, 19] |
| Mistral-Nemo 12B | -1.37 [-1.54, -1.24] | 203 [179, 233] | 13 [13, 14] | 14 [12, 15] |
| Gemma3 4B | -1.33 [-1.55, -1.11] | 337 [291, 396] | 14 [13, 14] | 9 [8, 10] |
| DeepSeek-R1 1.5B | -0.96 [-1.12, -0.83] | 4 [-23, 29] | 15 [15, 17] | not in screen |
| Llama3.2 3B | -0.92 [-1.14, -0.71] | 525 [428, 656] | 16 [15, 17] | 3 [2, 6] |
| Phi-4 Mini | -0.76 [-1.04, -0.50] | 1071 [819, 1574] | 17 [15, 19] | 1 [1, 1] |
| DeepSeek-R1 7B | -0.67 [-0.83, -0.51] | 340 [284, 438] | 18 [17, 19] | 8 [6, 9] |
| Qwen3 8B | -0.55 [-0.67, -0.44] | 263 [220, 321] | 19 [18, 20] | 11 [10, 12] |
| Phi-3 Mini | -0.47 [-0.61, -0.32] | 646 [501, 926] | 20 [19, 21] | 2 [2, 4] |
| Qwen3 30B-A3B | -0.40 [-0.50, -0.30] | 511 [414, 677] | 21 [20, 21] | 4 [2, 6] |
| Gemma3 1B | 0.13 [-0.04, 0.28] | 391 [106, 7269] | 22 [22, 23] | not in screen |
| Phi-2 2.7B | -0.07 [-0.22, 0.09] | 373 [-258, 6945] | 23 [22, 23] | not in screen |

## Rank stability

- Mean Spearman correlation between each bootstrap draw's |b_logp| ranking (23 engaged models) and the point-estimate ranking: 0.996 (over 500 complete draws).
- Mean Spearman for the WTP ranking (20 models): 0.991 (over 500 complete draws).
- P(Phi-4 Mini ranks #1 on WTP) = 0.996.
- P(Gemma2 9B in bottom 2 on WTP) = 1.000.
- P(GPT-5.4 in bottom 3 on WTP) = 0.990.

## The 16x WTP spread

- Point estimate: max(WTP)/min(WTP) over the 20 models = 15.6x (Phi-4 Mini 1071 / Gemma2 9B 68).
- Bootstrap 95% percentile CI for the ratio: [11.7x, 23.7x]; median 15.7x (over 500 draws; 0 draws with min(WTP) <= 0 were excluded from the ratio).

## WTP CI overlap among the 20 models

- Of the 190 pairwise comparisons, 148 pairs (78%) have non-overlapping 95% bootstrap CIs.
- Interpretation: individual adjacent models are often not separable, but the cross-model spread is not noise; the majority of pairwise contrasts, and in particular the top-vs-bottom contrasts behind the headline spread, remain separated after accounting for sampling uncertainty.

## Multiplicity audit (184 tests)

- All 8 estimated coefficients x 23 engaged models = 184 tests; Bonferroni and Benjamini-Hochberg at alpha = 0.05.
- Significant at raw p < .05: 155; after Bonferroni: 138; after BH: 155.
- Paper-claimed coefficients (b_logp and b_d_review_score for the 20-model screen, 40 tests): largest RAW p = 7.73e-07 (Qwen3 30B-A3B, b_logp).
- Claimed coefficients losing significance under Bonferroni: 0; under BH: 0.
- Bonferroni-adjusted p for the weakest claimed coefficient: 1.42e-04 (threshold 0.05).

## Capability correlations

Open-weight engaged models (n = 17); parameter counts reused verbatim from analysis/figures.py (MoE = active params). Proprietary models excluded (tier-based size estimates only).

- WTP vs log10(params): Pearson r = -0.233 (p = 0.368); Spearman rho = -0.371 (p = 0.142).
- Same, restricted to open-weight members of the 20-model screen (n = 14): Pearson r = -0.537 (p = 0.048); Spearman rho = -0.625 (p = 0.017).
- r2_log (decisiveness) vs log10(params): Pearson r = 0.624 (p = 0.007); Spearman rho = 0.659 (p = 0.004).
- Per-family Spearman of r2_log vs params (engaged members only):
  - Gemma3 (n = 4): rho = 1.000 (p = 0.000) [Gemma3 1B; Gemma3 27B; Gemma3 12B; Gemma3 4B]
  - Qwen3 (n = 3): rho = -0.500 (p = 0.667) [Qwen3 8B; Qwen3 4B; Qwen3 30B-A3B]
  - Phi (n = 5): rho = 0.527 (p = 0.361) [Phi-2 2.7B; Phi-3 Mini; Phi-4 Mini; Phi-4 14B; Phi-3 Medium]
