# Ensemble: Content IAB × Sentient Pair Model

This doc explains the ensemble experiment, the upgraded IAB model that drives it, what each strategy means (in particular `oracle`), and what the numbers in `Data/ensemble_metrics.csv` tell us.

## TL;DR — current best

| Strategy | Pair AUC | Pair AP | Accuracy | Δ vs sentient | Δ vs IAB | Deployable? |
|---|---:|---:|---:|---:|---:|---|
| `iab_only` (knn_frac, k=25) | 0.5929 | 0.4202 | 0.589 | −0.023 | 0.000 | yes |
| `sentient_only` (GBM, with_demographics) | 0.6157 | 0.4497 | 0.662 | 0.000 | +0.023 | yes |
| **`mean` (50/50)** | **0.6336** | **0.4621** | **0.661** | **+0.018** | **+0.041** | **yes** |
| **`weighted α=0.45`** (best honest combo) | **0.6336** | **0.4623** | **0.663** | **+0.018** | **+0.041** | **yes** |
| `stack_lr_oof` | 0.6272 | 0.4565 | 0.663 | +0.012 | +0.034 | yes |
| `max_confidence` | 0.6197 | 0.4517 | 0.661 | +0.004 | +0.027 | yes |
| `oracle` (upper bound) | 0.9207 | 0.8555 | 0.811 | +0.305 | +0.328 | **no** |

**Headline**: ship `mean` (or any `weighted_alpha` in 0.40–0.50; the plateau is flat) for **+0.018 AUC over sentient alone** and **+0.041 AUC over IAB alone**. The optimal mix is now near-equal weight — both signals contribute roughly the same after the IAB upgrade. Compared to the previous ensemble (LR-on-binary IAB, same sentient): **+0.014 AUC at the best ensemble strategy** and **+0.051 AUC on the IAB side alone**.

## What changed since the previous run

Earlier results topped out at AUC 0.6234 with `weighted_alpha=0.15` (i.e. mostly sentient, IAB barely helping). The fix was on the IAB side. The numbers below are **apples-to-apples ablations** — same fold split, same sentient model, same feature filtering on both sides; the only difference is the IAB configuration. (Re-run yourself with `--iab-model lr --iab-target binary --iab-hparam 1.0` for the old config.)

| Stage | Old (LR-on-binary IAB) | New (kNN-on-frac_positive IAB) | Δ |
|---|---|---|---:|
| IAB model | LogisticRegression on binary `y_iab = (net_likes >= 1)` | **kNeighborsRegressor (k=25, cosine)** on `frac_positive` | — |
| IAB target | Binary classification | `frac_positive[u, k] = (rating >= 3) @ multihot[:, k] / coverage[k]` ∈ [0, 1] | — |
| User features | 136 raw demographic one-hots (in original run) | **112 cols after dropping <5-user coverage** (cheap-win #3) | — |
| IAB-only pair AUC | 0.5415 | **0.5929** | **+0.051** |
| Best ensemble pair AUC | 0.6193 (α=0.10 weighted) | **0.6336** (α=0.45 weighted, or mean) | **+0.014** |
| Optimal α (weight on IAB) | 0.10 | 0.45 | — |
| Naïve mean | worse than sentient alone | **best, tied with weighted** | — |

The interesting finding from `src/model/iab_best.py` is in the next section.

## Why `frac_positive` was the real win (not just wider alpha sweep)

`src/model/iab_best.py` swept five IAB configurations × multiple hyperparameters and ranked them by **pair AUC after projection** (the metric the ensemble actually cares about). Result:

```
config                  hparam  per_cat_macro_auc  pair_auc  pair_ap
knn_frac    (k=25)        25.0           0.5901    0.5929   0.4202   <-- winner
knn_frac    (k=5)          5.0           0.6096    0.5917   0.4265
knn_frac    (k=10)        10.0           0.6099    0.5899   0.4344
ridge_frac  (alpha=200)  200.0           0.6206    0.5881   0.4216
ridge_frac  (alpha=100)  100.0           0.6222    0.5774   0.4120
knn_frac    (k=3)          3.0           0.6002    0.5708   0.3986
ridge_frac  (alpha=50)    50.0           0.6172    0.5652   0.4015
ridge_frac  (alpha=25)    25.0           0.6113    0.5533   0.3931
ridge_frac  (alpha=10)    10.0           0.6016    0.5410   0.3908
lr_bin      (C=0.1)        0.1           0.5977    0.5405   0.3838   <-- old default
knn_net     (k=5)          5.0           0.5849    0.5154   0.3623
ridge_net   (alpha=50)    50.0           0.6211    0.4967   0.3552
ridge_net   (alpha=100)  100.0           0.6218    0.4934   0.3508
...
```

Two stories here:

### 1. The PER-CAT story and the PAIR story disagree

Looking at the `per_cat_macro_auc` column, the best per-cat model is `ridge_net (alpha=100)` at 0.6218. But its **pair AUC** is only 0.4934 — barely above chance. The winner `knn_frac (k=25)` is mid-pack on per-cat AUC (0.5901) but tops the pair AUC table at 0.5929.

This is **the whole reason `iab_best.py` exists**: per-cat AUC and pair AUC after projection rank configurations differently, and the ensemble cares only about the latter. The cheap-wins suggestion ("wider alpha sweep") was tuned for per-cat AUC; doing it correctly required also testing the projection step.

### 2. `frac_positive` retains ranking under multi-hot averaging; `net_likes` doesn't

Why does `net_likes` get such poor pair AUC after projection?

- `signal = +1/-1/0` against the user mean. The mean over 300 ads dominates — only a few signal=+1 entries cross the threshold for any one user. After regression, per-cat predictions are **signed reals centred near zero** (mean −0.34, ~0.1 std on the prediction matrix).
- When you average those signed scores over the 1–3 multi-hot cats per ad, the variance across ads collapses (you're averaging numbers near zero) and the ranking signal mostly goes away.

`frac_positive` fixes that:
- Per-ad threshold is **fixed at rating ≥ 3** (binarization absorbs LLM tagging noise — that was cheap-win #2).
- Per-cat target is **strictly in [0, 1]**, with population mean 0.247. Predictions are bounded and informative across the full range.
- Averaging over multi-hot cats preserves both spread and ranking → the projection retains AUC.

This is why `frac_positive` configs **all** beat all `net_likes` configs on pair AUC, even when per-cat AUCs are similar.

### 3. The wider alpha sweep matters too — but for a different reason

The per-cat alpha sweep (compare_models.py) shows:

```
alpha=    1:  AUC = 0.5883
alpha=   25:  AUC = 0.6205
alpha=   50:  AUC = 0.6221   <-- new per-cat optimum (old grid topped at 1000)
alpha=  100:  AUC = 0.6186
alpha= 1000:  AUC = 0.5665
alpha=10000:  AUC = 0.4216
```

The old "best alpha=1000 at top of grid" was an artifact of the old grid not exploring low enough. With the wider sweep, the **per-cat optimum is α≈50**. But that finding doesn't transfer to the pair-AUC winner — `ridge_net (alpha=50)` is still bad at pair AUC (0.4967) because of the projection issue above. Use `iab_best.py` for IAB-in-ensemble decisions; use `compare_models.py` for per-cat decisions.

## What "oracle" means

`oracle` is **not a deployable model**. It's a theoretical upper bound that peeks at the ground-truth label to make its choice.

For each (user, ad) pair `i` with binary label `y_i ∈ {0, 1}` and the two base predictions `p_iab[i], p_sent[i] ∈ [0, 1]`:

```
err_iab[i]  = |y_i - p_iab[i]|
err_sent[i] = |y_i - p_sent[i]|
oracle[i]   = p_iab[i]   if err_iab[i] <= err_sent[i]
              else p_sent[i]
```

**For every pair, look at the truth, and copy whichever model got closer to it.** It's the score you'd get from a magical perfect router.

### Why we report it even though we can't ship it

| Use | What it tells us |
|---|---|
| **Headroom check** | The gap between `oracle` and any deployable strategy is the lift a *perfect* router could deliver. Our gap (0.92 vs 0.63) says there's substantial complementary signal we're still leaving on the table. |
| **Complementarity check** | If `oracle ≈ max(iab_only, sentient_only)`, the two models are redundant. `oracle = 0.92` vs the best single model 0.62 → strongly complementary. |
| **Why weighted/stacking can't reach it** | A weighted average and an LR meta-learner only see the two probability scores. They can't condition on *which features* of the user or ad would tell us "trust IAB here, trust sentient there". The oracle gap = the value of feature-conditional routing. |

### A concrete pair-by-pair example

Suppose for one pair `y=1`:
- IAB says `p_iab = 0.20` → error 0.80
- Sentient says `p_sent = 0.85` → error 0.15
- Oracle picks **sentient**, contributes 0.85.

For another pair `y=0`:
- IAB says `p_iab = 0.10` → error 0.10
- Sentient says `p_sent = 0.60` → error 0.60
- Oracle picks **IAB**, contributes 0.10.

So the oracle-score array is a per-pair-best-of-two picked using the label. Then we compute `roc_auc_score(y_pair, oracle)` over all 36 000 rows. It's not 1.0 because for pairs where **both** base models are wrong, oracle still has to pick the less-bad one.

## How the experiment is structured

`src/model/ensemble.py` produces the table at the top. Both models use the **exact same 5-fold user-level split** so OOF scores can be combined honestly.

### Stage-by-stage flow (with the new defaults)

| Stage | Model | What gets computed |
|---|---|---|
| 1 | — | Build the (user, ad) pair dataset (36k rows) — features for sentient, image-id alignment for IAB projection. |
| 2 | — | Build IAB targets: `y_iab` (binary), `net_likes` (signed sum), and **`frac_positive`** ([0, 1] continuous). |
| 3 | IAB | Per-fold per-cat **kNeighborsRegressor(k=25, cosine)** on the 112-col coverage-filtered user_features matrix, target = `frac_positive`. → OOF `(120 users, 26 cats)` score matrix. |
| 4 | IAB | **Project to pair grid**: for each `(u, ad)`, average `p_iab[u, k]` over `{k : multihot[ad, k] = 1}`. Then min-max scale to [0, 1] so it can be combined with the sentient probability. |
| 5 | sentient | Per-fold HistGradientBoostingClassifier (`with_demographics` profile) on the 36k pair matrix → OOF `(120, 300)` matrix of `p_sent[u, ad]`. |
| 6 | both | Compute every ensemble strategy, evaluated against `y_pair[u, ad] = (rating[u, ad] > user_mean[u])`. |

### The seven strategies

| Strategy | Formula | Intuition |
|---|---|---|
| `iab_only` | `p_iab[u, ad]` | Baseline 1: the IAB content model, projected to the pair grid. |
| `sentient_only` | `p_sent[u, ad]` | Baseline 2: the sentient pair model. |
| `mean` | `0.5 p_iab + 0.5 p_sent` | Naïve equal-weight. **Now optimal** thanks to balanced base models. |
| `weighted_alpha=α` | `α p_iab + (1−α) p_sent`, α ∈ {0.00, 0.05, …, 1.00} | Sweep α for the best AUC. Plateau around α=0.40–0.50. |
| `stack_lr_oof` | OOF logistic regression with inputs `[p_iab, p_sent]` | A meta-learner. In practice lands near the best weighted α. |
| `max_confidence` | Per pair: pick whichever score is further from 0.5 | "Trust the more decisive model." Now small positive lift (was negative). |
| `oracle (upper bound)` | Per pair: pick whichever score is closer to the true label | Cheats on the label. NOT deployable. Reports the max lift a perfect router could achieve. |

## Reading the numbers (current run)

### 1. Both models now contribute roughly equally
Old optimal α was 0.15 (IAB was a weak signal — barely helped). New optimal α is **0.45–0.50**. The IAB upgrade closed almost all of the per-base AUC gap (0.59 vs 0.62), and the ensemble now actually *averages* useful information rather than just down-weighting noise.

### 2. Naïve `mean` is now optimal
50/50 averaging used to underperform sentient alone (0.5968 vs 0.6157) because IAB was diluting it. Now naive mean is **tied with the best weighted α** at 0.6336. This is a healthier ensemble: insensitive to the mixing weight.

### 3. The α sweep plateau is wide
AUC ≥ 0.633 across α ∈ [0.40, 0.50], dropping smoothly outside. This is the signature of two roughly-equal complementary signals. The previous run had a sharp peak at α=0.15 — the signature of one strong + one weak signal.

### 4. `max_confidence` no longer backfires
Used to be 0.578 (below sentient alone). Now 0.620 (above sentient alone, below weighted). The IAB upgrade made its "confident" predictions less catastrophically wrong, so picking the more decisive model is now mildly useful.

### 5. Stacking lands slightly below the weighted ensemble (0.627 vs 0.634)
With only two inputs, the LR meta-learner has fewer degrees of freedom than the α sweep gets to use. Surprising it's not stronger — but consistent with the plateau being flat: there's not much extra info to be learned.

### 6. Oracle gap (~0.29 AUC) is still huge → router headroom is still real
`oracle = 0.921` vs `mean = 0.634`. A perfect per-pair router would give us +0.29 AUC. We can't have a perfect router, but a learned one — taking `(user features, ad features, p_iab, p_sent)` and predicting "which model wins on this pair" — could realistically capture some of that gap. Bigger gap than the +0.018 honest lift, so this is where the next 10× improvement lives.

## How to reproduce

```bash
# Step 1: find the best IAB config (this is what produced the new defaults)
python -m src.model.iab_best
# -> Data/iab_config_search.csv

# Step 2: ensemble with the winner pinned in as the default
python -m src.model.ensemble --save-predictions
# -> Data/ensemble_metrics.csv
# -> Data/ensemble_pair_predictions.csv
```

Runs in ~20s total. The defaults of `ensemble.py` now correspond to the `iab_best.py` winner: `--iab-model knn --iab-target frac_positive --iab-hparam 25`.

Useful overrides for ablations:

```bash
# Restore the old IAB (LR-on-binary) for direct comparison
python -m src.model.ensemble \
    --iab-model lr --iab-target binary --iab-hparam 1.0

# Try Ridge on frac_positive (the runner-up family)
python -m src.model.ensemble \
    --iab-model ridge --iab-target frac_positive --iab-hparam 200

# LR-on-sentient instead of GBM
python -m src.model.ensemble --sentient-model lr --sentient-lr-C 1.0

# Compact (60-dim) sentient profile to ablate demographics
python -m src.model.ensemble --sentient-profile compact
```

## Where to look next

1. **Learned router** — train a classifier with inputs `[user features, ad features, p_iab, p_sent]` and target `argmin_{model} |y − p_model|`. At inference, route to the predicted-better model. Closing even a third of the oracle gap would put us at AUC ~0.73.
2. **Calibration** — both base models are uncalibrated (Ridge-then-min-max for IAB, GBM logistic head for sentient). Isotonic regression on the OOF scores before averaging would make `weighted_alpha` and `stack_lr_oof` use scores on comparable scales.
3. **Re-examine the per-cat → pair projection** — the multi-hot averaging is naive. A weighted projection (e.g. inverse-cat-frequency weighting, or learning the projection coefficients per ad) might recover more of the per-cat signal at the pair level.
