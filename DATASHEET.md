# Datasheet for PriceBench

We document PriceBench following the *Datasheets for Datasets* framework
(Gebru et al., 2021).

## Motivation

PriceBench was created to measure the price, quality, and brand preferences an
LLM reveals when it chooses among qualifying options as a booking agent, rather
than whether it completes the task. It was created by Pavel Kireyev, the author of the
accompanying paper, at the Department of Management, London School of
Economics and Political Science, which funded the work.

## Composition

The benchmark has two parts:

1. A frozen pool of **179 real New York City hotel profiles** spanning 1–5
   stars, 36 neighborhoods, five hotel chains plus independents, and
   \$45–\$1,650 per night. Each profile carries ten attributes — four **fixed**
   (star rating, neighborhood, chain, amenities) and six **re-randomized** per
   appearance (price, room type, free-cancellation flag, breakfast flag, guest
   review score, review count).
2. **3,600 forced-choice tasks** built from the pool: 450 unique pairs each
   shown 4 times (1,800 binary) and 300 unique triples each shown 6 times
   (1,800 ternary).

The release also includes the **response sets**: 28 LLMs from 8 providers, each
task presented in both orderings, up to 7,200 observations per LLM. Instances
are commercial hotel listings; no personal or human-subject data is included.

## Collection

Hotel profiles were compiled from public listings on four online travel
agencies (Booking.com, Expedia, KAYAK, TripAdvisor) and then frozen; the data
describes lodging, not individuals. Model responses were collected by prompting
each LLM with plain-text option cards (see the paper's prompt appendix) at
temperature 0 with greedy decoding.

## Preprocessing and cleaning

On each appearance, six attributes are re-randomized so repeated presentations
are non-identical: price drawn uniformly within the property's listed range, review score perturbed within ±0.2, review count
within ±10%, and room type, cancellation, and breakfast flags resampled.
Perturbations are confined to each property's listed range so every profile
stays a plausible real listing. Task generation uses a fixed seed (2026); price
deciles for the non-parametric analysis are computed once on the pooled price
distribution.

## Uses

PriceBench is intended for measuring and comparing the revealed preferences of
LLM booking agents and for discrete-choice analysis of model behavior. It is a
diagnostic instrument, not a source of booking recommendations, and the
perturbed prices and reviews should not be treated as current commercial
information.

## Distribution

The tasks, scoring code, analysis pipeline, and all response sets are released
publicly at https://github.com/Pashasan/pricebench-emnlp, together with the
prompt-variant, decoding-temperature, and five-option runners, their response
sets, and the 300 five-option tasks. Code is under the MIT License; the task and response data are released
for research use. The hotel pool is a derived research artifact built from
public listings with re-randomized prices and review fields; users should
consult the source platforms' terms before any non-research reuse.

## Maintenance

The benchmark is versioned with the code release and maintained by the authors;
a new LLM is added by scoring it twice and registering the response pair, after
which the full analysis reproduces automatically.

---

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H.,
Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets.
*Communications of the ACM*, 64(12), 86–92.
