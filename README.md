# Creative Engagement Simulator

As part of the Decision Sciences Personalization Technologies team, we are mostly interested in how we can personalize Creative Delivery to our client's customers based on their past engagement history (e.g. what ad did they click on) and their personal profiles.

In this Hackathon project, we explore the use of Agentic Workflow to characterize images, and more importantly, to understand why a customer like or dislike a creative image. This understand can help us optimize our Advertisement Campaign towards better engagement.

## Data used

We used [Kaggle's ADS-16 dataset](https://www.kaggle.com/datasets/groffo/ads16-dataset) to build this demo.

## Web app overview (`app/`)

The **Image affinity ranker** (FastAPI + Jinja2 + HTMX under `app/`) lets a user attach **up to five** ad images, enter a **customer profile** (ADS16-style demographics and preferences, or a local CSV prefill), and run **Predict**. The backend scores each image, then shows a **ranked gallery**; clicking a thumbnail loads **per-image attributes** and **short reasoning** in a detail panel.

The table below summarizes the flow. Thumbnails are **downscaled** in the README; **click a screenshot** in the third column to open the **full-size** PNG on GitHub.

<table>
  <thead>
    <tr>
      <th align="left">Step</th>
      <th align="left">What you do</th>
      <th align="left">Screenshot (click for full size)</th>
    </tr>
  </thead>
  <tbody>
    <tr valign="top">
      <td><strong>1. Customer profile</strong></td>
      <td>Open the <strong>Customer profile</strong> tab. Fill numerical fields and categorical dropdowns, or load a one-row CSV in the browser to prefill the form (you can still edit values before submit).</td>
      <td>
        <a href="sample_screenshots/customer_profile_tab.png"><img src="sample_screenshots/customer_profile_tab.png" width="280" alt="Customer profile tab—click for full size"/></a>
      </td>
    </tr>
    <tr valign="top">
      <td><strong>2. Images</strong></td>
      <td>Open the <strong>Images</strong> tab. Attach between one and five images (JPEG, PNG, WebP, or GIF).</td>
      <td>
        <a href="sample_screenshots/image_attachment_tab.png"><img src="sample_screenshots/image_attachment_tab.png" width="280" alt="Images tab—click for full size"/></a>
      </td>
    </tr>
    <tr valign="top">
      <td><strong>3. Results</strong></td>
      <td>After <strong>Predict</strong>, the <strong>results</strong> page shows thumbnails <strong>sorted by score</strong> (best match first). Click a thumbnail to inspect model output and explanation. The capture reflects the <strong>HybridAgent</strong> inference path (Bedrock AgentCore image-ranking pipeline when <code>IMAGE_RANKING_AGENT_ARN</code> and related config are set).</td>
      <td>
        <a href="sample_screenshots/result_tab.png"><img src="sample_screenshots/result_tab.png" width="280" alt="Results tab—click for full size"/></a>
      </td>
    </tr>
  </tbody>
</table>

Add the three captures under [`sample_screenshots/`](sample_screenshots/) as `customer_profile_tab.png`, `image_attachment_tab.png`, and `result_tab.png` (create the folder if needed; you supply the PNGs).

## To run the simulator

- You first have to set up a conda environment with all dependencies, see `environment.yml` at the repository root.
- You also have to get access to the AWS Bedrock agent, and set up the corresponding environment variables with proper credentials. See `.kiro/specs/agentic-image-classification/agentcore-deployment.md` for how to set this up.

### Choosing the inference backend (`CustomInferenceInterface`)

The web app loads **one** of two backends (both subclasses of `CustomInferenceInterface` in `app/services/model_service.py`):

| Model | Role |
| ----- | ---- |
| **`ImageRankingAgentModel`** (default) | Calls the deployed **image ranking** Bedrock AgentCore runtime (`IMAGE_RANKING_AGENT_ARN`, plus `FEATURE_EXTRACTION_AGENT_ARN` as configured in `config/agentcore.env` or the environment). |
| **`IabAgentInferenceModel`** | Runs the **LR user profile → IAB scores → in-process ranking agent** path (expects the saved LR bundle under `Data/models/`, etc.). |

**How to select it**

1. **Environment variable** (works with plain `uvicorn`):

   ```bash
   export AGENT_MODEL=IabAgentInferenceModel   # or ImageRankingAgentModel
   uvicorn app.main:app --reload --app-dir .
   ```

   If `AGENT_MODEL` is unset or empty, the app defaults to **`ImageRankingAgentModel`**.

2. **Module runner** (parses `--agent-model` then starts uvicorn for you). Stock `uvicorn` does **not** accept a custom `--agent-model` flag on its own CLI, so use:

   ```bash
   python -m app --agent-model ImageRankingAgentModel --reload --app-dir .
   ```

   This sets `AGENT_MODEL` internally and forwards the remaining arguments to `uvicorn`.

On startup, the server prints a line such as `[agent-model] Loaded …` (or a message that it fell back to the hash **stub** if the requested model could not be constructed).

- After choosing the backend, start the demo as above (`uvicorn …` or `python -m app …`).

# ADS-16 User-Interest Modeling

This repo runs two complementary models on the ADS-16 corpus:

1. **Content / IAB model** (`src/model/compare_models.py`) — predicts per-`(user, IAB Tier-1 category)` interest from the user's demographic vector. Three baselines: Logistic Regression, Ridge, kNN.
2. **Sentient / pair model** (`src/model/train_sentient.py`) — predicts per-`(user, ad)` like/dislike from a compact user profile (50-cluster pos/neg taste + Big-5 personality, optionally + 136 demographics) and an LLM-extracted ad design / sentiment fingerprint. LR + HistGradientBoostingClassifier.

```
.
├── src/
│   ├── data_loader/        # data-prep pipeline (LLM tagging -> multi-hot -> per-user weighting)
│   │   ├── main.py         # 4-stage orchestrator (discover -> invoke -> multihot -> weight)
│   │   ├── ads16_processor.py
│   │   ├── multihot_from_responses.py
│   │   └── agent_processing/
│   │       ├── batch_invoke_ads.py    # parallel Bedrock AgentCore invocation
│   │       ├── categories_t1.py       # canonical IAB-t1 list + prompt builder
│   │       ├── categories_t2.py       # same for IAB-t2
│   │       ├── IAB-t1.csv             # 34 canonical Tier-1 categories
│   │       └── IAB-t2.csv
│   ├── ad_design/          # LLM-extracted per-ad design / sentiment features (20 fields)
│   │   ├── extract.py      # batch invoke (300 ads, both ADS-16 parts)
│   │   ├── parse.py        # JSONL -> Data/ads16_design_features.csv
│   │   ├── prompt.py / schema.py
│   │   └── validate.py     # test-retest reliability report
│   ├── model/
│   │   ├── interest_matrix.py
│   │   ├── train_logistic.py / train_ridge.py
│   │   ├── compare_models.py          # content IAB head-to-head (per-cat)
│   │   ├── iab_best.py                # IAB winner picker (pair AUC after projection)
│   │   ├── sentient_dataset.py        # builds the (user, ad) pair matrix
│   │   ├── train_sentient.py          # sentient model train + eval + bundle save
│   │   └── loader.py                  # load saved per-(user, IAB-cat) bundles
│   └── agentcore/          # Bedrock AgentCore deploy/delete helpers
└── Data/                   # raw inputs + every artifact produced below
```

## Quick start: run the model comparison

The repo already ships the artifacts you need (`Data/ads16_multihot_t1.csv` and `Data/user_features.csv`), so the comparison runs end-to-end without re-invoking the LLM.

```bash
python -m src.model.compare_models
```

Runs in ~20s on CPU. Output:

- Macro AUC / AP / Spearman per model after a hyperparameter sweep
- Per-category AUC table with the winning model per category
- Recommendation summary
- Side-by-side metrics written to `Data/model_comparison.csv`

Useful knobs:

```bash
# Stricter labels (more balanced, higher per-category AUC, fewer scorable cats)
python -m src.model.compare_models --min-net-likes 3

# Custom hyperparameter grids
python -m src.model.compare_models --Cs 0.1 1 --alphas 50 100 200 --knn-ks 3 5 7

# Use a different multi-hot file
python -m src.model.compare_models --multihot Data/ads16_multihot_t1.csv
```

Single-model runs:

```bash
python -m src.model.train_logistic    # LR only, with full per-category metrics
python -m src.model.train_ridge       # Ridge + kNN, with full per-category metrics
```

## Quick start: run the sentient (pair-level) model

This model predicts `P(rating > user_mean)` for every `(user, ad)` pair using ad design/sentiment features as inputs (not labels). Default user side is the 50-cluster pos/neg taste vector + Big-5; the ablation profile adds the 136 demographic columns.

```bash
python -m src.model.train_sentient --save-models
```

Runs in ~80s on CPU. Output:

- 5-fold **GroupKFold-by-user** cross-validation (predicting for users never seen at train time)
- Per-model **pair-level** AUC / AP / accuracy across the 36k `(user, ad)` rows
- Per-`(user, ADS-16 product cat)` aggregated metrics (mean over 15 ads in the cell, evaluated against the same `net_likes >= 1` label the content IAB model uses) → `Data/sentient_per_category_metrics.csv`
- Pair-level metric table → `Data/sentient_pair_metrics.csv`
- Refit-on-full-data joblib bundles → `Data/models/sentient_{lr,gbm}_{compact,with_demographics}.joblib`

Useful knobs:

```bash
# Single profile / single model
python -m src.model.train_sentient --profiles compact --models gbm

# Add IAB t1 multi-hot to the ad side (ablation)
# (use --include-iab when calling sentient_dataset directly; train_sentient
# ablates only across the user-side profile)

# Inspect the pair dataset without training
python -m src.model.sentient_dataset --head 5
python -m src.model.sentient_dataset --include-demographics --head 5
```

### Side-by-side reading guide

The two models answer **different questions** even though both use AUC:


| Model                      | Question                                                           | Granularity                        | Where to read AUC                        |
| -------------------------- | ------------------------------------------------------------------ | ---------------------------------- | ---------------------------------------- |
| Content IAB (LR/Ridge/kNN) | "Does user U have net positive interest in IAB-t1 category K?"     | per-`(user, 1-of-34 IAB cats)`     | `Data/model_comparison.csv`              |
| Sentient pair (LR/GBM)     | "Will user U like ad A specifically?"                              | per-`(user, 1-of-300 ads)`         | `Data/sentient_pair_metrics.csv`         |
| Sentient aggregated        | (sentient predictions averaged over 15 ads per ADS-16 product cat) | per-`(user, 1-of-20 product cats)` | `Data/sentient_per_category_metrics.csv` |


The aggregated AUCs are on the same scale as `model_comparison.csv` but the category axis is different (20 ADS-16 product cats vs 34 IAB-t1 cats), so use them as a "is the new model in the same neighbourhood?" sanity check rather than a strict winner-vs-loser comparison.

## Quick start: pick the best IAB content model

`src/model/iab_best.py` finds the best IAB-side configuration for **ensemble use** (i.e. ranks configs by pair AUC after projection to the (user, ad) grid, not just per-cat AUC). It sweeps three model families × two regression targets × multiple hyperparameters:

```bash
python -m src.model.iab_best
```

Runs in ~10s. Output: `Data/iab_config_search.csv` (one row per config: `per_cat_macro_auc, pair_auc, pair_ap`) and a printed winner.

**Winner**: `knn_frac (k=25)` — kNeighborsRegressor with cosine distance on the `frac_positive` target (continuous in [0, 1]). Pair AUC **0.5929**, vs the previous LR-on-binary default at **0.5405** (+0.052). The key insight: `frac_positive` retains ranking under multi-hot averaging, while signed `net_likes` collapses to noise after projection. See `[docs/ensemble_results.md](docs/ensemble_results.md)` for the full explanation.

## Quick start: ensemble the IAB and sentient models

`src/model/ensemble.py` puts both models on the same per-`(user, ad)` grid and combines them. Both are trained with the **same 5-fold user-level split** so out-of-fold predictions are honestly comparable, then ensembled in several ways.

```bash
python -m src.model.ensemble --save-predictions
```

Defaults pin in the winner from `iab_best.py`: kNN regressor on `frac_positive`, k=25, with low-coverage (<5 users) feature columns dropped. Runs in ~20s. What it does:

1. Train the IAB content model (default: kNN-on-frac_positive) per fold, get per-`(user, IAB-cat)` OOF scores.
2. Project to per-`(user, ad)` by averaging the IAB scores over the cats each ad is tagged with (ads with no scorable cats fall back to the global mean), then min-max scale to [0, 1] so it can be combined with the sentient probability.
3. Train the sentient model per fold (default: GBM `with_demographics`), get per-`(user, ad)` OOF probabilities directly.
4. Evaluate every strategy on the same 36k pair grid against the same `rating > user_mean` label.

Output: `Data/ensemble_metrics.csv` (one row per strategy: AUC, AP, accuracy, delta vs each base model) and optionally `Data/ensemble_pair_predictions.csv` (one row per pair: `user_id, image_id, rating, y_pair, p_iab, p_sent, p_stack`).

**Current best**: `mean` and `weighted_alpha=0.45` both hit pair AUC **0.6336** (up from 0.6234 with the previous IAB; +0.018 over sentient alone, +0.041 over IAB alone). The optimal mixing weight shifted from 0.15 (mostly sentient) to ~0.45 (near-equal), reflecting the IAB upgrade. Strategies evaluated:


| Strategy               | What it does                                                                                                                              | Deployable? |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `iab_only`             | Just the IAB projection, no ensemble                                                                                                      | yes         |
| `sentient_only`        | Just the sentient model                                                                                                                   | yes         |
| `mean`                 | `0.5 * p_iab + 0.5 * p_sentient`                                                                                                          | yes         |
| `weighted_alpha=X`     | `α * p_iab + (1-α) * p_sentient`, α swept on the 21-point grid `{0, 0.05, …, 1.0}` and the AUC-best α reported                            | yes         |
| `stack_lr_oof`         | Logistic-regression meta-learner on `(p_iab, p_sentient)`, also OOF-trained per fold                                                      | yes         |
| `max_confidence`       | Per-pair: pick whichever score is further from 0.5 (the model that's "more decisive") — your "best of the 2 models per prediction"        | yes         |
| `oracle (upper bound)` | Per-pair: pick whichever model's score is closer to truth. Cheats on the label, NOT deployable — shows the headroom from a perfect router | no          |


Useful flags for ablations:

```bash
# Restore the old IAB (LR-on-binary) for direct comparison
python -m src.model.ensemble --iab-model lr --iab-target binary --iab-hparam 1.0

# Try Ridge on frac_positive (the runner-up family)
python -m src.model.ensemble --iab-model ridge --iab-target frac_positive --iab-hparam 200
```

Detailed write-up + interpretation: `[docs/ensemble_results.md](docs/ensemble_results.md)`.

## Quick start: 6-base majority voting

`src/model/voting.py` trains all three estimator families (LR, Ridge, kNN) on **both** feature spaces — six base models in total — and combines them by majority vote.

```bash
python -m src.model.voting --save-predictions
```

Runs in ~30s. Each base model casts a binary `like` vote per pair; we then evaluate `majority_6 (≥4)`, `majority_iab (≥2 of 3)`, `majority_sent (≥2 of 3)`, `unanimous_6`, `any_6 (≥1)`, plus the continuous `vote_count` (0..6) used for AUC ranking. Output: `Data/voting_metrics.csv` (per-base-model + per-strategy metrics) and optionally `Data/voting_pair_predictions.csv` (each model's votes per pair).

**Result**: voting tops out at AUC 0.572 — *worse* than the weighted ensemble at 0.623. Hard thresholding throws away each base model's confidence, weak voters dilute strong ones, and the 0..6 vote axis is too coarse for AUC ranking. Detailed analysis + the three lessons in `[docs/voting_results.md](docs/voting_results.md)`.

## Re-running the data pipeline (only if you want fresh LLM tags)

The full orchestrator at `src/data_loader/main.py` produces `ads16_multihot_t1.csv` and `user_vectors_t1.csv` from scratch. It needs an already-deployed Bedrock AgentCore runtime (or pass `--deploy` to spin one up):

```bash
# Use the existing runtime hardcoded in batch_invoke_ads.py
python -m src.data_loader.main

# Skip stages you've already completed
python -m src.data_loader.main --skip invoke      # reuse existing JSONL
python -m src.data_loader.main --skip invoke multihot

# Smoke test on one user / first 6 images
python -m src.data_loader.main --users U0001 --limit 6
```

## Inputs


| File                                          | Shape      | What it is                                                                                                                                                                                                                                                                                            |
| --------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Data/user_features.csv`                      | 120 × 136  | One row per user. All columns are 0/1 indicators except `inf__age`, `inf__weekly_working_hours`, `inf__income` (numeric). Prefixes: `inf__` (demographics), `pref__` (self-reported preferences for music / books / movies / TV / websites).                                                          |
| `Data/ads16_multihot_t1.csv`                  | 300 × 40   | One row per ad image, indexed by `image_id` of the form `"<category>_<image>"` (e.g. `1_1` ... `20_15`). 34 of the columns are the canonical IAB Tier-1 categories with values in `{0, 1}`; the remaining 6 are metadata (`category`, `image_index`, `path`, `raw_text`, `n_matched`, `n_unmatched`). |
| `Data/ADS-16/.../*-RT.csv`                    | 1 per user | Raw 5-point ratings the user gave to each of the 300 ads. Loaded by `ADS16DataProcessor` and aligned to the multi-hot via `image_id`.                                                                                                                                                                 |
| `Data/ADS-16/.../*-B5.csv`                    | 1 per user | TIPI-style 10-question Big-5 personality answers (`;`-delimited, `Question#;Answer`). Used by the sentient model.                                                                                                                                                                                     |
| `Data/sentiment_multihot_clusters.csv`        | 120 × 51   | Per-user 50-dim revealed visual taste over image clusters. Each `cluster_NN` ∈ {-1, 0, +1} encoding "user has a personally-supplied IM-NEG / IM-POS image in this cluster". Built externally by clustering all `UXXXX-IM-{POS,NEG}/*.png` by visual embedding. Used by the sentient model.            |
| `Data/ads16_design_features.csv`              | 300 × 20   | Per-ad LLM-extracted design / sentiment features (4 ordered enums, 3 unordered enums, 5 booleans, 8 ints in 1–10). Indexed by `image_id`. Built by `src/ad_design/`.                                                                                                                                  |
| `src/data_loader/agent_processing/IAB-t1.csv` | 34 lines   | Canonical IAB Tier-1 category list. Used both to build the LLM prompt and to filter multi-hot columns down to "real" categories.                                                                                                                                                                      |


## Labels produced and used by the models

`src/model/compare_models.py` builds two label matrices from the inputs above. Both are signed/binarized **per-user**, which removes the bias from each user having a different baseline rating tendency.

For each user `u` with mean rating `mu_u`:

```
signal[u, i]      = +1 if rating[u, i] > mu_u
                    -1 if rating[u, i] < mu_u
                     0 otherwise           # equals user's mean
net_likes[u, k]   = sum_i  signal[u, i] * multihot[i, k]
                    # net (likes - dislikes) on cat-k ads
y_binary[u, k]    = 1 if net_likes[u, k] >= min_net_likes (default 1) else 0
```

Two upstream filters are applied to the multi-hot columns first:

1. **Canonical filter** — drop any column not in `IAB-t1.csv` (handles LLM-introduced columns from earlier prompt versions).
2. **Zero-exposure filter** — drop any canonical category whose column sums to 0 across all 300 ads (no ad ever tagged with it). With the current corpus this drops 8 categories (Attractions, Careers, Crime, Disasters, Law, Politics, Religion & Spirituality, Science), leaving **26 usable categories**.

Categories where `y_binary` doesn't have at least 5 positives **and** 5 negatives are skipped at scoring time (LR/AUC are undefined on a single-class label).

The three models predict on these labels as follows:

- **LR** treats `y_binary` as the target and predicts `P(y=1)` via `LogisticRegression(class_weight="balanced")`.
- **Ridge** treats `net_likes` (continuous, signed) as the target and predicts a continuous score.
- **kNN** treats `net_likes` as the target with `KNeighborsRegressor(metric="cosine", weights="distance")`.

All three predictions are evaluated against the same `y_binary` (for AUC, AP) and the same `net_likes` (for Spearman). All three pipelines wrap the model in `StandardScaler` + 5-fold `KFold` cross-validation. Hyperparameters (`C`, `alpha`, `k`) are picked on macro AUC.

## Outputs


| File                                                               | Produced by                       | What's in it                                                                                                                                                                                                                        |
| ------------------------------------------------------------------ | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Data/ads16_agent_responses_t1.jsonl`                              | `main.py` stage 2                 | Raw LLM response per image (resumable).                                                                                                                                                                                             |
| `Data/ads16_multihot_t1.csv`                                       | `main.py` stage 3                 | The multi-hot matrix described above.                                                                                                                                                                                               |
| `Data/user_vectors_t1.csv`                                         | `main.py` stage 4                 | `ratings @ multihot` per user (centered + L2-normalized by default). Not used by the model comparison; produced for downstream similarity work.                                                                                     |
| `Data/user_labels_above_mean.csv`                                  | `train_logistic.py`               | 120 × 26 binary `y_binary` matrix.                                                                                                                                                                                                  |
| `Data/user_labels_above_mean_net.csv`                              | `train_logistic.py`               | 120 × 26 signed `net_likes` matrix.                                                                                                                                                                                                 |
| `Data/model_comparison.csv`                                        | `compare_models.py`               | Content IAB: per-category AUC / AP / Spearman for LR/Ridge/kNN + winner column.                                                                                                                                                     |
| `Data/iab_config_search.csv`                                       | `iab_best.py`                     | IAB winner search: one row per `(model, target, hparam)` with `per_cat_macro_auc, pair_auc, pair_ap`. Used to pin the default IAB config in `ensemble.py`.                                                                          |
| `Data/{ridge,knn,logistic}_per_category_metrics.csv`               | the matching trainer              | Single-model per-category metrics (more detailed than the comparison file).                                                                                                                                                         |
| `Data/models/{lr,ridge,knn}_model.joblib`                          | `compare_models.py --save-models` | Per-category fitted pipelines for the content IAB models, loadable via `src.model.loader.load_bundle`.                                                                                                                              |
| `Data/sentient_pair_metrics.csv`                                   | `train_sentient.py`               | Pair-level AUC / AP / accuracy for `(profile, model)` combinations.                                                                                                                                                                 |
| `Data/sentient_per_category_metrics.csv`                           | `train_sentient.py`               | Sentient model predictions averaged to per-(user, ADS-16 product cat) and scored on the same `net_likes >= 1` label as the content models.                                                                                          |
| `Data/models/sentient_{lr,gbm}_{compact,with_demographics}.joblib` | `train_sentient.py --save-models` | Refit-on-full-data sentient pipelines.                                                                                                                                                                                              |
| `Data/ensemble_metrics.csv`                                        | `ensemble.py`                     | Pair-level AUC / AP / accuracy for `iab_only`, `sentient_only`, `mean`, `weighted_alpha=`*, `stack_lr_oof`, `max_confidence`, `oracle`. See `[docs/ensemble_results.md](docs/ensemble_results.md)`.                                 |
| `Data/ensemble_pair_predictions.csv`                               | `ensemble.py --save-predictions`  | One row per (user, ad): `user_id, image_id, rating, y_pair, p_iab, p_sent, p_stack`.                                                                                                                                                |
| `Data/voting_metrics.csv`                                          | `voting.py`                       | 6 base models (LR/Ridge/kNN on each of IAB and sentient sides) + 6 voting strategies (`majority_6`, `majority_iab`, `majority_sent`, `unanimous_6`, `any_6`, `vote_count`). See `[docs/voting_results.md](docs/voting_results.md)`. |
| `Data/voting_pair_predictions.csv`                                 | `voting.py --save-predictions`    | One row per (user, ad) with each base model's binary vote and the total vote count.                                                                                                                                                 |


## Dependencies

The code runs against the conda env at `~/miniconda3_x64/envs/cos_env`. Required packages: `numpy`, `pandas`, `scikit-learn`, `scipy`, `boto3` (only for re-running the LLM stage).