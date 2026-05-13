# Majority-Voting Ensemble: 6 Base Models, IAB × Sentient

This doc reports the **6-base-model majority-voting** experiment from `src/model/voting.py`. We train Logistic Regression, Ridge, and kNN on **both** feature spaces (IAB content × user features, and sentient pair × design features) — six base models in total — and combine them by majority vote.

## TL;DR

The headline number: **majority voting (AUC 0.572) underperforms the weighted ensemble from the previous experiment (AUC 0.623)**. This is a useful negative result that surfaces three real lessons about ensembling weak heterogeneous models, documented at the bottom of this doc.

If you only read one table:

| Strategy | Pair AUC | Pair AP | Accuracy | Predicted-pos rate |
|---|---:|---:|---:|---:|
| Best single base model (`sent_ridge`) | 0.5681 | 0.3942 | 0.5432 | 50.3% |
| `vote_count` / `majority_6` (≥4 of 6) | 0.5721 | 0.3945 | 0.6567 | 6.0% |
| `majority_iab` (≥2 of 3 IAB) | 0.5306 | 0.3667 | 0.6635 | 2.7% |
| `majority_sent` (≥2 of 3 sentient) | 0.5649 | 0.3807 | 0.5607 | 45.3% |
| `unanimous_6` (=6 of 6) | 0.5721 | 0.3945 | 0.6579 | 0.01% |
| `any_6` (≥1 of 6) | 0.5721 | 0.3945 | 0.4292 | 82.2% |
| **Reference: weighted α=0.15 (prev ensemble)** | **0.6234** | **0.4565** | **0.6652** | — |

True pair-positive rate is 34.2% across the 36k (user, ad) pairs.

## Setup

### The six base models

All six are trained out-of-fold on the **same** 5-fold user-level split (held-out users are identical across all six models so their predictions can be compared row-for-row).

| Side | Name | Estimator | Trained on | Target | Vote rule |
|---|---|---|---|---|---|
| IAB | `iab_lr` | LogisticRegression (classifier) | `(120, 136)` user features | binary `net_likes ≥ 1` per IAB cat | projected score ≥ 0.5 |
| IAB | `iab_ridge` | Ridge (regressor) | `(120, 136)` user features | signed `net_likes` per IAB cat | projected score > 0 |
| IAB | `iab_knn` | KNeighborsRegressor (k=10, cosine) | `(120, 136)` user features | signed `net_likes` per IAB cat | projected score > 0 |
| Sentient | `sent_lr` | LogisticRegression (classifier) | `(36k, 225)` pair features | binary `rating > user_mean` | prob ≥ 0.5 |
| Sentient | `sent_ridge` | Ridge (regressor) | `(36k, 225)` pair features | signed `rating − user_mean` | score > 0 |
| Sentient | `sent_knn` | KNeighborsRegressor (k=25, cosine) | `(36k, 225)` pair features | signed `rating − user_mean` | score > 0 |

### Structural symmetry

Both sides use the same three estimator families (LR / Ridge / kNN). The signed-real continuous signals are mirrored: `net_likes` per IAB cat on the IAB side, `rating − user_mean` per pair on the sentient side. LR is always the binary classifier on the binarized version of that signal. Ridge and kNN are always regressors on the continuous version. So when we vote, the threshold is the same kind of decision rule on both sides:

- **classifier**: `prob ≥ 0.5` ⇒ vote like
- **regressor**: `predicted score > 0` ⇒ vote like

### IAB → pair projection

The three IAB models produce per-`(user, IAB-cat)` scores (probabilities for LR, signed reals for Ridge/kNN). For the per-pair vote, each is projected to `(user, ad)` using the multi-hot:

```
proj_score[u, ad] = mean over { k : multihot[ad, k] = 1 } of model_score[u, k]
```

Ads tagged only with non-scorable cats fall back to the global mean. Same projection function as in `src/model/ensemble.py` so the IAB-LR projection is identical to what was used in the previous (weighted-mix) experiment.

## Per-base-model results

```
                   model     kind    auc     ap  accuracy  precision  recall     f1  predicted_pos_rate
                  iab_lr     base 0.5497 0.3896    0.6059     0.3912  0.2720 0.3209              0.2380
               iab_ridge     base 0.4950 0.3666    0.6641     0.6101  0.0519 0.0957              0.0291
                 iab_knn     base 0.4954 0.3410    0.6577     0.5294  0.0007 0.0015              0.0005
                 sent_lr     base 0.5320 0.3644    0.5542     0.3693  0.4270 0.3960              0.3958
              sent_ridge     base 0.5681 0.3942    0.5432     0.3861  0.5671 0.4594              0.5028
                sent_knn     base 0.5562 0.3901    0.5401     0.3798  0.5429 0.4469              0.4892
```

Two surprises here:

1. **`iab_ridge` and `iab_knn` are essentially at chance (AUC ≈ 0.495)**. Their per-`(user, IAB-cat)` AUC was around 0.55 in `Data/model_comparison.csv`, but **the projection from cat-level to pair-level dilutes the signal**: most ads are tagged with multiple IAB cats, so per-pair scores get averaged across cats with mixed predictive value.
2. **`iab_ridge` and `iab_knn` almost never vote "like"** — predicted-positive rates of 2.9% and 0.05% versus the true 34%. Their continuous score after projection is rarely > 0, because Ridge and kNN regressors get pulled toward the dataset mean (≈ 0 for centered net_likes). So in the 6-way vote they almost always cast a "dislike" vote.

The sentient regressors (`sent_ridge`, `sent_knn`) are the opposite: they predict-pos at ~50% (vs true 34%), more liberal than the truth.

So the six voters split into three behavioural buckets:

| Behaviour | Voters |
|---|---|
| Conservative (almost always vote dislike) | `iab_ridge`, `iab_knn` |
| Calibrated-ish (mid pos rate) | `iab_lr`, `sent_lr` |
| Liberal (almost always vote like) | `sent_ridge`, `sent_knn` |

That's a structural problem for an unweighted vote — see "Why voting underperforms" below.

## Voting strategy results

```
                   model     kind    auc     ap  accuracy  precision  recall     f1  predicted_pos_rate
       vote_count (0..6) ensemble 0.5721 0.3945    0.6567     0.4914  0.0858 0.1461              0.0598
        majority_6 (>=4) ensemble 0.5721 0.3945    0.6567     0.4914  0.0858 0.1461              0.0598
 majority_iab (>=2 of 3) ensemble 0.5306 0.3667    0.6635     0.6086  0.0471 0.0874              0.0265
majority_sent (>=2 of 3) ensemble 0.5649 0.3807    0.5607     0.3929  0.5196 0.4474              0.4527
       unanimous_6 (==6) ensemble 0.5721 0.3945    0.6579     1.0000  0.0004 0.0008              0.0001
             any_6 (>=1) ensemble 0.5721 0.3945    0.4292     0.3611  0.8672 0.5098              0.8221
```

`vote_count`, `majority_6`, `unanimous_6`, `any_6` all share the same continuous score (the 0–6 vote count) so their **AUC and AP are identical** — they only differ in where the binary cutoff lands.

### Vote distribution

```
votes  count
    0   6403
    1  10458
    2  10789
    3   6199
    4   1801
    5    345
    6      5
```

- 79% of pairs get 0–2 votes; only 6% reach the strict-majority threshold (≥ 4).
- Only 5 pairs out of 36 000 are unanimous (vote = 6) — when the conservative IAB regressors and the liberal sentient regressors all agree, the prediction is essentially never "like". Hence `unanimous_6` precision = 1.0 but recall = 0.0004.
- The 0-vote bucket (6 403 pairs) means six independent models all agreed it's a "dislike" — which is correct often enough to drive the high accuracy of the conservative strategies.

## Why voting underperforms the weighted ensemble

The weighted ensemble from `docs/ensemble_results.md` got AUC **0.623** at α=0.15. This 6-way voting tops out at **0.572**. Three reasons:

### 1. Hard thresholding throws away the magnitude signal

Each base model produces a continuous score (probability or signed real). Voting collapses that to a single bit — "did you cross your threshold or not". A model that's *barely* over the threshold and a model that's *strongly* over the threshold contribute the same vote. The weighted ensemble averages the actual probabilities, preserving confidence.

This is the dominant effect. It's also why all four of `vote_count`, `majority_6`, `unanimous_6`, `any_6` have the same AUC: AUC ranks predictions, and the only ranking signal in the binary votes is "how many models voted yes, 0–6".

### 2. Adding weak voters dilutes the strong ones

`iab_ridge` and `iab_knn` are essentially at chance (AUC 0.495). In an unweighted vote they have the same say as `sent_ridge` (0.568) or `iab_lr` (0.550). Weighted ensembling can downweight or zero out a weak model; majority voting can't.

The previous weighted-α experiment effectively gave `iab_lr` only 15% weight (with `sent_gbm` getting 85%) — close to optimal because `iab_lr` is barely additive over `sent_gbm`. Voting forces all six voters to count equally.

### 3. The vote-count axis is too coarse for AUC

AUC is computed by ranking 36 000 predictions. With six voters there are only 7 distinct vote counts (0..6), so 36 000 predictions get bucketed into 7 levels. Within each level every prediction is tied. Tied scores hurt AUC because positives and negatives in the same bucket count as ambiguous pairs.

Continuous probability scores from `weighted_alpha=0.15` produce ~36 000 distinct values — much finer ranking, and that maps to higher AUC.

## What the conservative-majority numbers actually mean

`majority_iab (≥2 of 3)` reaches **66.4% accuracy** with just **2.7% predicted-positive rate**. Compare to the trivial baseline "always predict dislike" → 65.8% accuracy (since 34.2% are actual likes). So `majority_iab` accuracy gains almost nothing from the always-no baseline, while sacrificing ~70% of the recall.

It's a textbook case of **accuracy being misleading on imbalanced labels** — the 66% accuracy is essentially the trivial baseline plus a tiny number of high-precision "like" calls.

This is also why we lead with AUC and AP in the comparison: those metrics are insensitive to class prior and threshold choice.

## What the strategies *do* tell us, beyond the headline AUC

Despite the worse AUC, the voting experiment surfaces useful side-information:

| Strategy | Use case |
|---|---|
| `unanimous_6 (=6)` | Precision 1.0 on the 5 pairs where all six models agreed. Tiny recall, but if you only want to surface ads you're certain a user will like, this is the most conservative selector. |
| `majority_iab` | High precision (0.61), low recall (0.05). Useful when false positives are very costly (e.g. ad budget). |
| `any_6 (≥1)` | Recall 0.87 at the cost of accuracy collapsing to 0.43. Useful as a "candidate set" filter — start here, then re-rank with the weighted-α ensemble. |
| `vote_count` (continuous score 0–6) | Coarse but cheap: a single 36k-element integer column you can sort by. Top-N selection from this column is a perfectly reasonable batch ranker. |

## How to reproduce

```bash
python -m src.model.voting --save-predictions
```

Runs in ~30s (the kNN-on-pairs fold is the slow part). Outputs:
- `Data/voting_metrics.csv` — the per-base-model + per-strategy table above
- `Data/voting_pair_predictions.csv` — one row per (user, ad) with each model's binary vote and the total vote count

Useful overrides:

```bash
# Lower regularization on Ridge so it predicts a wider range of net_likes
python -m src.model.voting --iab-ridge-a 10 --sent-ridge-a 10

# Smaller k for kNN
python -m src.model.voting --iab-knn-k 5 --sent-knn-k 10

# Compact sentient profile (60-dim) for faster + more interpretable runs
python -m src.model.voting --sentient-profile compact
```

## Where to look next

1. **Soft voting**. Instead of `prob ≥ 0.5`, sum the underlying probabilities (after rescaling the regressor scores via sigmoid or rank percentile). This recovers the magnitude signal that hard thresholding throws away. Likely lands close to the weighted-α 0.62 number.
2. **Drop the weak voters**. Re-run majority voting with just `iab_lr`, `sent_ridge`, `sent_knn` — the three voters with AUC > 0.55. With three voters majority is `≥ 2`, and you avoid the chance-level base learners diluting the vote.
3. **Calibration before voting**. The IAB-Ridge / IAB-kNN regressors output signed reals that are heavily concentrated near zero — fitting a per-model isotonic regression to map their scores to calibrated probabilities (and then voting at 0.5 of the calibrated prob) might give them a fairer share of the "yes" votes.
4. **Take this comparison to the deck as the negative result**. "We tried 6-model majority voting and it lost to a 2-model weighted average; here's why" is a clean, defensible slide that demonstrates rigour.
