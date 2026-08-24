# Prompt-format ablation

Log-price logit (paper spec) per model x prompt variant, baseline
re-estimated on the identical task subset. WTP at $250 reference.

## Coefficients by model and variant

| Model | Variant | b_lnp (SE) | b_review (SE) | WTP@250 (SE) | R2 | n | z vs base (lnp) |
|---|---|---|---|---|---|---|---|
| Qwen3 4B | baseline | -1.56 (0.14) | +0.80 (0.11) | 128 (20) | 0.129 | 1800 | - |
| Qwen3 4B | json | -0.39 (0.11) | +0.32 (0.09) | 203 (84) | 0.013 | 1800 | +6.55 |
| Qwen3 4B | reorder | -1.49 (0.14) | +0.80 (0.11) | 134 (22) | 0.141 | 1800 | +0.38 |
| Qwen3 4B | persona | -1.79 (0.14) | +1.07 (0.11) | 150 (20) | 0.176 | 1800 | -1.17 |
| Qwen3 4B | cot | -1.99 (0.16) | +0.45 (0.12) | 56 (15) | 0.296 | 1800 | -2.04 |
| Gemma2 9B | baseline | -5.56 (0.30) | +1.56 (0.16) | 70 (8) | 0.516 | 1800 | - |
| Gemma2 9B | json | -7.17 (0.40) | +1.22 (0.17) | 43 (6) | 0.607 | 1800 | -3.22 |
| Gemma2 9B | reorder | -3.78 (0.21) | +1.48 (0.14) | 98 (11) | 0.382 | 1800 | +4.79 |
| Gemma2 9B | persona | -2.01 (0.15) | +1.73 (0.14) | 215 (23) | 0.222 | 1800 | +10.55 |
| Gemma2 9B | cot | -2.01 (0.16) | +0.30 (0.11) | 37 (14) | 0.253 | 1800 | +10.45 |
| Phi-4 Mini | baseline | -0.80 (0.16) | +3.25 (0.20) | 1020 (216) | 0.424 | 1800 | - |
| Phi-4 Mini | json | +0.03 (0.14) | +1.15 (0.13) | 9969 (46708) | 0.248 | 1800 | +3.92 |
| Phi-4 Mini | reorder | -0.46 (0.16) | +2.57 (0.17) | 1406 (493) | 0.424 | 1800 | +1.51 |
| Phi-4 Mini | persona | -0.81 (0.17) | +3.43 (0.21) | 1063 (234) | 0.483 | 1800 | -0.04 |
| Phi-4 Mini | cot | -1.73 (0.14) | +1.02 (0.11) | 147 (20) | 0.161 | 1800 | -4.35 |
| Gemma3 27B | baseline | -4.12 (0.34) | +3.45 (0.29) | 209 (25) | 0.435 | 900 | - |
| Gemma3 27B | json | -3.41 (0.22) | +3.10 (0.20) | 228 (21) | 0.465 | 1800 | +1.76 |
| Gemma3 27B | reorder | -2.99 (0.20) | +2.89 (0.18) | 241 (22) | 0.354 | 1800 | +2.87 |
| Gemma3 27B | persona | -1.90 (0.21) | +4.81 (0.27) | 632 (78) | 0.528 | 1800 | +5.57 |
| GPT-5.4 Nano | baseline | -2.57 (0.21) | +4.74 (0.28) | 462 (46) | 0.538 | 1800 | - |
| GPT-5.4 Nano | json | -1.91 (0.19) | +4.73 (0.27) | 619 (72) | 0.525 | 1800 | +2.30 |
| GPT-5.4 Nano | reorder | -2.00 (0.18) | +3.72 (0.21) | 465 (49) | 0.424 | 1800 | +2.07 |
| GPT-5.4 Nano | persona | -1.75 (0.20) | +5.24 (0.30) | 748 (97) | 0.572 | 1800 | +2.80 |
| GPT-5.4 Nano | cot | -3.73 (0.23) | +4.20 (0.24) | 281 (24) | 0.501 | 1800 | -3.75 |
| GPT-5.4 Mini | baseline | -4.34 (0.24) | +3.69 (0.21) | 212 (17) | 0.430 | 1800 | - |
| GPT-5.4 Mini | json | -4.03 (0.23) | +4.09 (0.22) | 253 (20) | 0.422 | 1800 | +0.93 |
| GPT-5.4 Mini | reorder | -4.88 (0.26) | +3.37 (0.21) | 173 (14) | 0.450 | 1800 | -1.54 |
| GPT-5.4 Mini | persona | -3.04 (0.22) | +5.19 (0.28) | 427 (38) | 0.497 | 1800 | +4.04 |
| GPT-5.4 Mini | cot | -3.52 (0.20) | +2.47 (0.17) | 176 (16) | 0.336 | 1800 | +2.62 |

## Cross-model ordering stability (Spearman vs baseline ordering)

| Variant | Models | rho |b_lnp| | rho WTP |
|---|---|---|---|
| json | 6 | +1.00 | +1.00 |
| reorder | 6 | +0.94 | +0.94 |
| persona | 6 | +0.89 | +0.89 |
| cot | 5 | +0.60 | +0.70 |

Coverage: Qwen3 4B: ['baseline', 'cot', 'json', 'persona', 'reorder'], Gemma2 9B: ['baseline', 'cot', 'json', 'persona', 'reorder'], Phi-4 Mini: ['baseline', 'cot', 'json', 'persona', 'reorder'], Gemma3 27B: ['baseline', 'json', 'persona', 'reorder'], GPT-5.4 Nano: ['baseline', 'cot', 'json', 'persona', 'reorder'], GPT-5.4 Mini: ['baseline', 'cot', 'json', 'persona', 'reorder']
