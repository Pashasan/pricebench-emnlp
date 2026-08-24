# Temperature check: position-locking at temperature 0.7

Settings: temperature 0.7, top_p 0.95, top_k 40, seed 42; binary
tasks, both orderings; first-shown rate pooled over orderings.
T=0 baseline from outputs/data/triage.csv. Verdict band: locked if
first-shown rate outside [15%, 85%].

| Model | Role | First (T=0) | First (T=0.7) | Verdict T=0.7 | b_lnp T=0.7 (SE) | b_lnp T=0 | R2 T=0.7 / T=0 | n |
|---|---|---|---|---|---|---|---|---|
| Llama3.1 8B | locked | 88.2% | 89.2% | position-locked | -0.73 (0.08) | -0.55 | 0.040 / 0.022 | 3600 |
| Llama3 8B | locked | 88.2% | 91.4% | position-locked | -0.69 (0.08) | -0.59 | 0.037 / 0.029 | 3600 |
| Mistral 7B | locked | 99.5% | 98.9% | position-locked | -0.08 (0.08) | -0.00 | 0.001 / 0.000 | 3600 |
| Qwen3 0.6B | locked | 98.5% | 52.2% | engaged | -0.17 (0.08) | -0.01 | 0.031 / 0.002 | 3600 |
| Llama3.2 1B | locked | 100.0% | 99.3% | position-locked | +0.00 (0.08) | -0.00 | 0.001 / -0.000 | 3600 |
| Gemma3 4B | engaged-control | 66.2% | 58.9% | engaged | -2.51 (0.12) | -1.33 | 0.280 / 0.216 | 3600 |
| Qwen3 4B | engaged-control | 22.7% | 17.8% | engaged | -1.42 (0.09) | -1.59 | 0.114 / 0.131 | 3600 |

Coverage: 7/7 models scored.
