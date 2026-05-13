# ADS-16 User-Interest Modeling

This part of the repo turns ADS-16 ad images + user ratings into a per-`(user, IAB-category)` interest matrix and benchmarks three baseline recommender models on it (Logistic Regression, Ridge, kNN).

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
│   ├── model/              # supervised baselines on top of the interest matrix
│   │   ├── interest_matrix.py
│   │   ├── train_logistic.py
│   │   ├── train_ridge.py             # Ridge + kNN regressors
│   │   └── compare_models.py          # head-to-head LR vs Ridge vs kNN
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

| File | Shape | What it is |
|---|---|---|
| `Data/user_features.csv` | 120 × 136 | One row per user. All columns are 0/1 indicators except `inf__age`, `inf__weekly_working_hours`, `inf__income` (numeric). Prefixes: `inf__` (demographics), `pref__` (self-reported preferences for music / books / movies / TV / websites). |
| `Data/ads16_multihot_t1.csv` | 300 × 40 | One row per ad image, indexed by `image_id` of the form `"<category>_<image>"` (e.g. `1_1` ... `20_15`). 34 of the columns are the canonical IAB Tier-1 categories with values in `{0, 1}`; the remaining 6 are metadata (`category`, `image_index`, `path`, `raw_text`, `n_matched`, `n_unmatched`). |
| `Data/ADS-16/.../*-RT.csv` | 1 per user | Raw 5-point ratings the user gave to each of the 300 ads. Loaded by `ADS16DataProcessor` and aligned to the multi-hot via `image_id`. |
| `src/data_loader/agent_processing/IAB-t1.csv` | 34 lines | Canonical IAB Tier-1 category list. Used both to build the LLM prompt and to filter multi-hot columns down to "real" categories. |

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

| File | Produced by | What's in it |
|---|---|---|
| `Data/ads16_agent_responses_t1.jsonl` | `main.py` stage 2 | Raw LLM response per image (resumable). |
| `Data/ads16_multihot_t1.csv` | `main.py` stage 3 | The multi-hot matrix described above. |
| `Data/user_vectors_t1.csv` | `main.py` stage 4 | `ratings @ multihot` per user (centered + L2-normalized by default). Not used by the model comparison; produced for downstream similarity work. |
| `Data/user_labels_above_mean.csv` | `train_logistic.py` | 120 × 26 binary `y_binary` matrix. |
| `Data/user_labels_above_mean_net.csv` | `train_logistic.py` | 120 × 26 signed `net_likes` matrix. |
| `Data/model_comparison.csv` | `compare_models.py` | Per-category AUC / AP / Spearman for all 3 models + winner column. |
| `Data/{ridge,knn,logistic}_per_category_metrics.csv` | the matching trainer | Single-model per-category metrics (more detailed than the comparison file). |

## Dependencies

The code runs against the conda env at `~/miniconda3_x64/envs/cos_env`. Required packages: `numpy`, `pandas`, `scikit-learn`, `scipy`, `boto3` (only for re-running the LLM stage).
