# Locked-model response forensics: parse validity, slot-following, and deviation signal

Script: `analysis/locked_forensics.py`. Data: raw result CSV pairs registered in `config.MODELS`, binary tasks only (task_id 1-1800 original, 10001-11800 swap). Swap-file choices are stored flipped back to the original frame, so the first-shown option corresponds to recorded choice A in the original file and recorded choice B in the swap file.

## 1. Valid-choice rates (binary tasks)

| Model | Group | n orig | n swap | Valid orig | Valid swap | Invalid orig | Invalid swap | Empty/NaN |
|---|---|---|---|---|---|---|---|---|
| Llama3.2 1B | locked | 1800 | 1800 | 100.00% | 100.00% | 0 | 0 | 0 |
| Mistral 7B | locked | 1800 | 1800 | 100.00% | 100.00% | 0 | 0 | 0 |
| Qwen3 0.6B | locked | 1800 | 1800 | 100.00% | 100.00% | 0 | 0 | 0 |
| Llama3 8B | locked | 1800 | 1800 | 100.00% | 100.00% | 0 | 0 | 0 |
| Llama3.1 8B | locked | 1800 | 1800 | 100.00% | 100.00% | 0 | 0 | 0 |
| Gemma3 4B | engaged | 1800 | 1800 | 100.00% | 100.00% | 0 | 0 | 0 |
| Qwen3 4B | engaged | 1800 | 1800 | 100.00% | 100.00% | 0 | 0 | 0 |

## 2. Raw-output character (binary tasks, both orderings)

Categories: bare-letter = output is exactly one of A/B/C after stripping whitespace; short-verbose = <= 50 chars containing a choice letter (e.g. `A.` or `Option A.`); long-verbose = > 50 chars; empty/other = empty or no choice letter found.

| Model | Group | Bare-letter | Short-verbose | Long-verbose | Empty/other |
|---|---|---|---|---|---|
| Llama3.2 1B | locked | 0.2% | 99.8% | 0.0% | 0.0% |
| Mistral 7B | locked | 0.0% | 100.0% | 0.0% | 0.0% |
| Qwen3 0.6B | locked | 99.5% | 0.0% | 0.5% | 0.0% |
| Llama3 8B | locked | 100.0% | 0.0% | 0.0% | 0.0% |
| Llama3.1 8B | locked | 100.0% | 0.0% | 0.0% | 0.0% |
| Gemma3 4B | engaged | 100.0% | 0.0% | 0.0% | 0.0% |
| Qwen3 4B | engaged | 100.0% | 0.0% | 0.0% | 0.0% |

### Representative raw outputs (5 most frequent per model, truncated to 80 chars)

**Llama3.2 1B** (locked):

- `A.` (x3592)
- `A` (x8)

**Mistral 7B** (locked):

- `Option A.` (x3588)
- `Option B.` (x5)
- `Option A: The Standard High Line` (x4)
- `Option A: The Pierre, A Taj Hotel` (x2)
- `Option A` (x1)

**Qwen3 0.6B** (locked):

- `A` (x3504)
- `B` (x79)
- `A \n \n The options are compared based on price, amenities, and location. Option` (x6)
- `A \n \n The option with the higher price per night is A, so I choose` (x2)
- `A \n \n The option with the higher price per night is **Option B**, so` (x2)

**Llama3 8B** (locked):

- `A` (x3334)
- `B` (x266)

**Llama3.1 8B** (locked):

- `A` (x3358)
- `B` (x242)

**Gemma3 4B** (engaged contrast):

- `A` (x2585)
- `B` (x1015)

**Qwen3 4B** (engaged contrast):

- `B` (x2914)
- `A` (x686)

## 3. Slot-following vs hotel-habit (binary tasks)

P(pick first-SHOWN option) by ordering. A slot-follower is high on both columns; a model with a stable preference for one hotel would be high on one and low on the other.

| Model | Group | P(first-shown), original | P(first-shown), swap | Triage first-rate (all tasks) |
|---|---|---|---|---|
| Llama3.2 1B | locked | 100.0% | 100.0% | 100.0% |
| Mistral 7B | locked | 99.8% | 99.9% | 99.5% |
| Qwen3 0.6B | locked | 98.2% | 97.3% | 98.5% |
| Llama3 8B | locked | 94.3% | 90.9% | 88.2% |
| Llama3.1 8B | locked | 94.2% | 92.4% | 88.2% |
| Gemma3 4B | engaged | 75.7% | 67.9% | 66.2% |
| Qwen3 4B | engaged | 24.4% | 13.7% | 22.7% |

## 4. Attribute signal in deviations (standard log-price logit)

Spec: `Logit(choice_A ~ const + d_stars + d_bed + d_cancel + d_breakfast + d_review_score + d_review_count_100 + d_log_price)` on the pooled original+swap binary data (identical to `analysis/personality.py`).

| Model | Group | n | b_logp | SE | p | Pseudo-R2 | Coefs p<.05 (of 7) |
|---|---|---|---|---|---|---|---|
| Llama3.2 1B | locked | 3600 | -0.000 | 0.076 | 1.00e+00 | -0.0000 | 0 |
| Mistral 7B | locked | 3600 | -0.001 | 0.076 | 9.86e-01 | 0.0000 | 0 |
| Qwen3 0.6B | locked | 3600 | -0.010 | 0.076 | 9.00e-01 | 0.0019 | 0 |
| Llama3 8B | locked | 3600 | -0.594 | 0.079 | 5.53e-14 | 0.0286 | 5 |
| Llama3.1 8B | locked | 3600 | -0.553 | 0.078 | 1.56e-12 | 0.0217 | 5 |
| Gemma3 4B | engaged | 3600 | -1.326 | 0.095 | 2.23e-44 | 0.2164 | 7 |
| Qwen3 4B | engaged | 3600 | -1.590 | 0.094 | 3.61e-64 | 0.1309 | 6 |

## 5. Retry evidence (original runs)

The runner retried a task up to 3 times when no choice letter could be parsed, and logged only the final attempt's raw_output, so exact retry counts are not recoverable from the CSVs. Two observable proxies: (a) share of final outputs that are not a bare letter (weak instruction following), and (b) rows whose final choice is empty/NaN (all 3 attempts failed).

| Model | Group | Bare-letter, orig run | Not-bare, orig run | All-attempts-failed rows (orig+swap) |
|---|---|---|---|---|
| Llama3.2 1B | locked | 0.2% | 99.8% | 0 |
| Mistral 7B | locked | 0.0% | 100.0% | 0 |
| Qwen3 0.6B | locked | 99.5% | 0.5% | 0 |
| Llama3 8B | locked | 100.0% | 0.0% | 0 |
| Llama3.1 8B | locked | 100.0% | 0.0% | 0 |
| Gemma3 4B | engaged | 100.0% | 0.0% | 0 |
| Qwen3 4B | engaged | 100.0% | 0.0% | 0 |

Interpretation caveat: the not-bare outputs of Llama3.2 1B (`A.`) and Mistral 7B (`Option A.`) parse deterministically on the first attempt, so a not-bare final output does not imply a retry happened. The zero unparseable/empty rows show that no task ever exhausted its 3 attempts.

## Bottom line

1. Parse failure is ruled out: every one of the 3,600 binary responses per model (1,800 per ordering) parsed to a valid A/B choice; there are zero empty, NaN, or unparseable outputs for any of the five locked models. Three locked models (Qwen3 0.6B, Llama3 8B, Llama3.1 8B) emit a perfectly bare letter in 99.5-100% of cases, matching the format discipline of the engaged contrasts, yet still land in the locked band.

2. The behavior is slot-following, not a hotel habit: locked models pick the first-SHOWN option at 90.9-100% under BOTH orderings, which only position can explain. Engaged contrasts sit far from the band in both orderings (Gemma3 4B 75.7%/67.9% primacy lean; Qwen3 4B 24.4%/13.7% recency lean).

3. A nuance: the rare deviations of Llama3 8B and Llama3.1 8B (first-shown 90.9-94.3%) do carry attribute signal (b_logp -0.59 and -0.55, p < 1e-11, 5 of 7 coefficients p < .05), but the fit is an order of magnitude weaker than the engaged contrasts (pseudo-R2 0.02-0.03 vs 0.13-0.22) and the price coefficient is attenuated roughly 2.5x by the dominant position behavior, so excluding them is conservative: it avoids reporting attenuated preference estimates rather than hiding parse failures. The other three locked models carry no attribute signal at all (b_logp p >= 0.90, 0 of 7 coefficients significant).

## Data notes

- Mistral 7B: swap file also holds a legacy +1000-offset block (3600 rows, excluded); its binary choices replicate the standard block at 99.9% agreement.
- Llama3.1 8B: 290 ternary rows absent across the pair (does not affect binary analysis).
