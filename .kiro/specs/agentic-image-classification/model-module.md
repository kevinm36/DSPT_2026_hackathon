# Module Spec: `src/model/` — User Interest Modeling & Prediction

## Overview

The `src/model/` package trains and evaluates supervised models that predict per-category user interest from demographic/behavioral features. It sits downstream of the classification pipeline: once ads have IAB multi-hot vectors and users have rated those ads, this module learns the mapping from sparse user metadata to rich IAB preference profiles.

## Module Structure

```
src/model/
├── __init__.py              # Public API: build_interest_matrix, load_multihot
├── interest_matrix.py       # Builds (user × category) interest scores from ratings + multi-hot
├── train_logistic.py        # Per-category Logistic Regression (binary: does user like cat k?)
├── train_ridge.py           # Ridge + kNN regression on continuous interest scores
└── compare_models.py        # Head-to-head comparison: LR vs Ridge vs kNN
```

## Components

### `interest_matrix.py` — Ground Truth Construction

Computes a `(n_users, n_categories)` interest matrix that answers: "given an ad has category k, how does this user feel about it?"

**Key functions:**

| Function | Purpose |
|----------|---------|
| `load_multihot(csv)` | Load per-image multi-hot CSV, return `(DataFrame, cat_cols)` |
| `build_interest_matrix(...)` | Main entry point — produces the interest matrix |

**Rating normalization modes** (`rating_norm`):
- `"none"` — raw 1–5 ratings
- `"center"` — subtract per-user mean (default, removes user-level bias)
- `"zscore"` — z-score per user

**Aggregation modes** (`aggregate`):
- `"mean"` — average residual across ads tagged with category k (default)
- `"sum"` — raw weighted sum (the original `ads16_processor` formula)
- `"frac_positive"` — fraction of cat-k ads rated ≥ threshold
- `"like_dislike"` — signed deviation from a fixed neutral value, averaged

**Design rationale:** The standard `user_vector = ratings @ multihot` conflates exposure with preference (popular categories accumulate larger sums). The interest matrix normalizes by coverage, producing a signed, exposure-corrected score that's comparable across users.

---

### `train_logistic.py` — Binary Classification per Category

Predicts a binary label: "does user u have net positive interest in category k?"

**Label construction:**
1. Compute per-user mean rating μ_u
2. Signal: +1 if rating > μ_u, −1 if < μ_u, 0 if equal
3. Net likes per category: `net[u, k] = signal @ multihot[:, k]`
4. Binary label: `y[u, k] = 1 if net[u, k] >= min_net_likes`

**Key functions:**

| Function | Purpose |
|----------|---------|
| `build_above_mean_labels(...)` | Construct binary labels + net scores from ratings |
| `_filter_features(X_df, min_coverage)` | Drop sparse feature columns |
| `_build_logistic(C)` | StandardScaler → LogisticRegression pipeline |
| `_sweep_C(...)` | Hyperparameter sweep selecting C by macro AUC |
| `_evaluate_per_category(...)` | Full CV evaluation at fixed C |
| `run(...)` | End-to-end training driver |

**Pipeline:**
1. Build signed net-like labels (canonical IAB-t1 filtered, zero-exposure dropped)
2. Align user features with labels
3. Sweep regularization C via macro AUC across scorable categories
4. Evaluate per-category: AUC, AP, F1, precision, recall, accuracy

**Outputs:**
- `Data/user_labels_above_mean.csv` — binary label matrix
- `Data/logistic_per_category_metrics.csv` — per-category evaluation
- `Data/logistic_cv_predictions.csv` — cross-validated probability predictions

**CLI:** `python -m src.model.train_logistic [--min-net-likes N] [--Cs ...] [--multihot ...]`

---

### `train_ridge.py` — Continuous Regression

Predicts the continuous interest score (from `interest_matrix`) rather than a binary label. Trains both Ridge regression and kNN as competing baselines.

**Key functions:**

| Function | Purpose |
|----------|---------|
| `_build_ridge(alpha)` | StandardScaler → MultiOutputRegressor(Ridge) pipeline |
| `_build_knn(k)` | StandardScaler → KNeighborsRegressor(cosine, distance-weighted) |
| `_filter_features(...)` | Drop sparse columns (same logic as logistic) |
| `_sweep_hyperparam(...)` | Generic hyperparameter sweep by macro R² |
| `_per_category_metrics(...)` | R², RMSE, MAE, Spearman per category |
| `run(...)` | End-to-end driver with head-to-head comparison |

**Pipeline:**
1. Build interest matrix (configurable norm + aggregation)
2. Align features, drop sparse columns and zero-variance targets
3. Sweep Ridge alpha and kNN k via 5-fold macro R²
4. Cross-validated predictions for both models
5. Per-category metrics + head-to-head summary

**Outputs:**
- `Data/user_interest_matrix.csv` — the computed interest matrix
- `Data/ridge_per_category_metrics.csv` / `Data/knn_per_category_metrics.csv`
- `Data/ridge_cv_predictions.csv` / `Data/knn_cv_predictions.csv` (optional)

**CLI:** `python -m src.model.train_ridge [--aggregate mean|sum|frac_positive|like_dislike] [--rating-norm center|none|zscore] [--alphas ...] [--knn-ks ...]`

---

### `compare_models.py` — Unified Model Comparison

Runs all three model families (LR, Ridge, kNN) on the same binary labels with threshold-free ranking metrics for an apples-to-apples comparison.

**Evaluation metrics (per category):**
- **AUC** — predicted score vs binary "user has net positive interest"
- **AP** — precision-recall AUC on the same
- **Spearman** — predicted score vs continuous signed net-like score

**Pipeline:**
1. Build binary labels (same as `train_logistic`)
2. Align features
3. Per-model hyperparameter sweep (C for LR, alpha for Ridge, k for kNN)
4. Per-category CV predictions at best hyperparameter
5. Side-by-side macro/median metrics, per-category winner analysis

**Outputs:**
- `Data/model_comparison.csv` — full per-category metrics for all three models
- Console recommendation based on macro AUC ranking

**CLI:** `python -m src.model.compare_models [--min-net-likes N] [--cv-folds N]`

---

## Data Flow

```
user_features.csv (120 users × 136 features)
        │
        │   ads16_multihot_t1.csv (300 ads × K IAB categories)
        │         │
        │         │   user rating CSVs (120 users × 300 ratings)
        │         │         │
        ▼         ▼         ▼
   ┌─────────────────────────────┐
   │  interest_matrix.py         │
   │  build_interest_matrix()    │──► user_interest_matrix.csv
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────────┐
   │  train_ridge.py          train_logistic.py   │
   │  (continuous target)     (binary target)     │
   └──────────────┬───────────────────┬───────────┘
                  │                   │
                  ▼                   ▼
   ┌──────────────────────────────────────────────┐
   │  compare_models.py                           │
   │  Unified evaluation: LR vs Ridge vs kNN      │
   └──────────────────────────────────────────────┘
                  │
                  ▼
        model_comparison.csv
        (per-category AUC, AP, Spearman for each model)
```

## Dependencies

- scikit-learn (LogisticRegression, Ridge, KNeighborsRegressor, StandardScaler, KFold, cross_val_predict)
- pandas, numpy (data handling)
- scipy (spearmanr correlation)
- `src.data_loader` (ADS16DataProcessor, discover_users, corpus constants)

## Design Decisions

1. **Per-category models** rather than a single multi-output classifier — each IAB category has different base rates and different feature relevance. Per-category training allows independent hyperparameter selection and interpretable per-category diagnostics.

2. **Signed net-like labels** — distinguishes "user dislikes category k" from "user was never exposed to category k" by using the sign of (likes − dislikes) relative to the user's own mean.

3. **Canonical filtering** — multi-hot columns are restricted to the canonical IAB-t1 list before training. This drops hallucinated or non-standard categories from the LLM classification step.

4. **Feature coverage filter** — one-hot features with fewer than N non-zero users are dropped (default N=5). These ultra-sparse features add noise without signal at small sample sizes.

5. **Three model families** — LR for interpretable probabilities, Ridge for smooth continuous scores, kNN for non-parametric similarity. The comparison module determines which is best for this dataset.
