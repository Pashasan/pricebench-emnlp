# Five-option tasks

300 quintets from the same pool (seed 2027), both orderings (full
reversal), temperature 0. Prediction uses the paper binary log-price
logit utilities; chance: hit 20%, top-2 40%, mean rank 3.0.

| Model | n | Slot1 | Slot2 | Slot3 | Slot4 | Slot5 | Hit | Top-2 | Mean rank | b_lnp 5opt (SE) | b_lnp binary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma2 9B | 600 | 10.2% | 26.3% | 24.2% | 25.3% | 14.0% | 52.2% | 84.2% | 1.67 | -3.49 (0.25) | -5.71 |
| Qwen3 4B | 600 | 1.8% | 14.3% | 24.7% | 26.8% | 32.3% | 50.7% | 78.0% | 1.80 | -2.61 (0.21) | -1.59 |
| Phi-4 Mini | 600 | 6.3% | 41.0% | 38.8% | 2.7% | 11.2% | 44.3% | 69.8% | 2.04 | -0.48 (0.15) | -0.76 |
| Mistral-Nemo 12B | 600 | 55.5% | 17.0% | 9.3% | 11.7% | 6.5% | 46.7% | 67.7% | 2.12 | -1.82 (0.18) | -1.37 |
| Llama3.1 8B | 600 | 61.2% | 16.2% | 4.8% | 8.7% | 9.2% | 42.3% | 66.2% | 2.12 | -1.81 (0.17) | -0.55 |
