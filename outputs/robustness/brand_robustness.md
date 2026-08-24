# Brand robustness checks

Script: `analysis/brand_robustness.py`. Engaged models: 23. All specs: binary logit on difference-coded attributes, controls (stars, bed, cancel, breakfast, review score, review count/100) + price-decile dummies + brand difference dummies (Hilton/Marriott/IHG/Hyatt/Wyndham vs Independent). Permutation test: mean pairwise Pearson r of 5-dim brand vectors, within-provider minus across-provider gap, 2000 label permutations, seed 42, one-sided p.

## 1. Baseline replication

Reproduced analysis/brand.py exactly: within-provider mean r +0.0789, across +0.0901, gap -0.0112, permutation p = 0.4810 (stored values in outputs/data/brand_cluster.json: +0.0789 / +0.0901 / p = 0.481). Exact match: yes.

GPT-5.4 baseline Wyndham coefficient: +2.1729 (paper: +2.17). Models with all five chain coefficients > 0: 11 of 23.

## 2. Most influential Wyndham listing

Wyndham-family properties in the pool (5):

| id | name | stars | neighborhood | price range |
|---|---|---|---|---|
| 125 | Wyndham New Yorker Hotel | 3 | Midtown | $120-$240 |
| 126 | Ramada by Wyndham New York/Eastside | 2 | Midtown East | $80-$160 |
| 127 | Days Inn by Wyndham NYC Midtown 45 | 2 | Midtown | $75-$150 |
| 128 | La Quinta Inn & Suites New York Times Square South | 2 | Midtown | $90-$180 |
| 129 | Best Western Plus Hospitality House Suites | 3 | Midtown East | $130-$260 |

Leave-one-property-out for GPT-5.4 (baseline Wyndham +2.173):

| dropped property | Wyndham coef | tasks dropped |
|---|---|---|
| Wyndham New Yorker Hotel | +2.0835 | 80 |
| Ramada by Wyndham New York/Eastside | +2.3816 | 40 |
| Days Inn by Wyndham NYC Midtown 45 | +2.0729 | 16 |
| La Quinta Inn & Suites New York Times Square South | +2.1969 | 8 |
| Best Western Plus Hospitality House Suites | +2.1735 | 24 |

Largest move toward zero: **Days Inn by Wyndham NYC Midtown 45** (2 stars, Midtown, $75-$150 base-rate range). Dropping it moves the GPT-5.4 Wyndham coefficient from +2.173 to +2.073.

**Note:** no single property dominates. The largest leave-one-out movement of the GPT-5.4 Wyndham coefficient is 0.209 (coefficient stays in [+2.073, +2.382] across all five drops). Dropping the Wyndham New Yorker Hotel, the most centrally located listing, leaves the coefficient at +2.084. The premium is spread across the whole Wyndham family, and (see S2) is partly explained by location controls but remains large and significant.

## 3. Permutation test across specs

| spec | within r | across r | gap | perm. p | all-5-pos models |
|---|---|---|---|---|---|
| baseline | +0.0789 | +0.0901 | -0.0112 | 0.4810 | 11/23 |
| S1_drop_culprit | +0.0639 | +0.0697 | -0.0058 | 0.4660 | 12/23 |
| S2_area_controls | +0.0756 | +0.0648 | +0.0108 | 0.4175 | 7/23 |
| S3_luxury_tier | +0.0612 | +0.1117 | -0.0505 | 0.6270 | 11/23 |

## 4. Wyndham coefficients, focus models

| model | baseline | S1 (drop listing) | S2 (+area) | S3 (+luxury) |
|---|---|---|---|---|
| GPT-5.4 | +2.173*** | +2.073*** | +1.225*** | +2.178*** |
| GPT-5.4 Mini | +1.702*** | +1.638*** | +1.343*** | +1.767*** |
| Gemma3 27B | +1.147*** | +1.150*** | +0.971*** | +1.243*** |
| GPT-4.1 Mini | +1.062*** | +0.966*** | +0.567* | +1.076*** |

significance: * p<0.05, ** p<0.01, *** p<0.001

## 5. Mean brand coefficient across engaged models

| spec | Hilton | Marriott | IHG | Hyatt | Wyndham |
|---|---|---|---|---|---|
| baseline | +0.225 | +0.275 | +0.139 | +0.276 | +0.395 |
| S1_drop_culprit | +0.224 | +0.276 | +0.147 | +0.277 | +0.349 |
| S2_area_controls | +0.180 | +0.250 | +0.078 | +0.149 | +0.240 |
| S3_luxury_tier | +0.246 | +0.304 | +0.164 | +0.321 | +0.418 |

## 6. S3 luxury-tier dummy

Luxury sub-brands matched on hotel name: Ritz-Carlton, St. Regis, EDITION, W New York, Conrad, Park Hyatt, InterContinental, Lotte, Andaz, Thompson. Properties matched in pool: 15. Luxury coefficient: 7/23 models positive, 1 significantly positive (p<0.05), 4 significantly negative.

## 7. Takeaways

- The headline permutation result (no provider clustering of brand preferences) is robust: one-sided p = 0.481 (baseline), 0.466 (S1), 0.417 (S2), 0.627 (S3); the within-minus-across gap is never significantly positive.
- The GPT-5.4 Wyndham premium is not driven by any single listing (see section 2): it is spread across the whole Wyndham family.
- Area controls (S2) absorb part of the chain premia: the count of models with all five chain coefficients positive falls from 11/23 to 7/23, and mean chain coefficients shrink but all stay positive. GPT-5.4 Wyndham remains +1.225 (p = 1.7e-04) under area controls. Models with significantly positive Wyndham coefficients (p<0.05): 7 (baseline), 7 (S1), 5 (S2), 7 (S3).
- The luxury-tier dummy (S3) barely moves the chain coefficients (all-five-positive count 11/23) and is itself mostly non-positive, so the chain premia are not a disguised luxury effect.

## Files

- `brand_robustness_coefs.csv` -- spec, model, brand, coef, se, p (brand=LuxuryTier rows are the S3 luxury dummy)
- `brand_robustness_permutation.csv` -- spec, within_r, across_r, gap, p_perm
