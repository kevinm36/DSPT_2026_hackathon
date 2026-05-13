"""Per-category Logistic Regression: predict 'does this user like category k'.

Label construction (signed +1/-1/0 to distinguish dislike from non-exposure):

    mu_u                = mean(rating[u, :])             # user's overall baseline
    signal[u, i]        = +1 if rating[u, i] > mu_u
                          -1 if rating[u, i] < mu_u
                           0 otherwise (rated exactly at the mean)
    net_likes[u, k]     = Σᵢ signal[u, i] · multihot[i, k]   # net (likes − dislikes)
    y[u, k]             = 1 if net_likes[u, k] >= min_net_likes else 0

Category columns are first restricted to the canonical IAB-t1 list, and any
canonical category with **zero corpus exposure** (no ads tagged) is dropped --
those are unlearnable by construction.

So ``y[u, k] = 1`` means "user u liked at least ``min_net_likes`` more cat-k
ad(s) than they disliked", where like/dislike are relative to the user's own
mean rating. The signed encoding lets a 'no positive interest' user with
balanced likes/dislikes (0 net) be distinguished from a 'never exposed' user
(handled by dropping zero-exposure categories upstream).

Run with::

    python -m src.model.train_logistic
    python -m src.model.train_logistic --min-net-likes 1
    python -m src.model.train_logistic --multihot Data/ads16_multihot_t1.csv \
        --Cs 0.01 0.1 1 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    ADS16DataProcessor,
    CORPUS_ROOTS,
    IMAGE_ID_FOR,
    IMAGES_PER_CATEGORY,
    NUM_CATEGORIES,
    discover_users,
)
from src.data_loader.agent_processing.categories_t1 import (
    DEFAULT_CATEGORIES_PATH as DEFAULT_T1_PATH,
    load_categories,
)
from src.model.interest_matrix import DEFAULT_META_COLS, load_multihot


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTIHOT = REPO_ROOT / "Data/ads16_multihot_t1.csv"
DEFAULT_FEATURES = REPO_ROOT / "Data/user_features.csv"
DEFAULT_CANONICAL = DEFAULT_T1_PATH
DEFAULT_LABELS_OUT = REPO_ROOT / "Data/user_labels_above_mean.csv"
DEFAULT_METRICS_OUT = REPO_ROOT / "Data/logistic_per_category_metrics.csv"
DEFAULT_PREDS_OUT = REPO_ROOT / "Data/logistic_cv_predictions.csv"


def build_above_mean_labels(
    rating_csv_paths: dict[str, Path],
    multihot_csv: Path,
    *,
    min_net_likes: int = 1,
    canonical_path: Optional[Path] = DEFAULT_CANONICAL,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return ``(binary_labels, net_like_scores, dropped_cols)``.

    Pipeline:
      1. Load multi-hot, restrict columns to the canonical IAB-t1 list (if
         ``canonical_path`` given). Non-canonical columns (e.g. metadata or
         hallucinated categories from the LLM) are dropped.
      2. Drop categories with **zero corpus exposure** (sum across images = 0)
         -- they're unlearnable since no ad was ever tagged with them.
      3. For each user u, compute signed per-image signal:
              +1 if rating > user's mean,  -1 if below,  0 if at mean
         then aggregate per category: ``net = signal @ multihot``.
      4. Binarize: ``label = (net >= min_net_likes)``.
    """
    M_df, all_cols = load_multihot(multihot_csv)

    # Step 1: filter to canonical T1 list (if provided)
    if canonical_path is not None:
        canonical = load_categories(canonical_path)
        keep_canonical = [c for c in all_cols if c in set(canonical)]
        dropped_non_canonical = [c for c in all_cols if c not in set(canonical)]
        cat_cols = keep_canonical
    else:
        canonical = None
        dropped_non_canonical = []
        cat_cols = all_cols

    # Step 2: drop zero-exposure (no ads tagged with this cat)
    M_full = M_df[cat_cols].values
    exposure = M_full.sum(axis=0)
    keep_exposed = [c for c, e in zip(cat_cols, exposure) if e > 0]
    dropped_zero_exposure = [c for c, e in zip(cat_cols, exposure) if e == 0]
    cat_cols = keep_exposed
    dropped_cols = dropped_non_canonical + dropped_zero_exposure

    image_ids = [
        IMAGE_ID_FOR(c, i)
        for c in range(NUM_CATEGORIES)
        for i in range(IMAGES_PER_CATEGORY)
    ]
    M = M_df.loc[image_ids, cat_cols].values  # (300, n_cats_kept)

    nets: dict[str, np.ndarray] = {}
    for uid, rt_path in rating_csv_paths.items():
        proc = ADS16DataProcessor(
            rating_csv_path=rt_path,
            multihot_csv_path=multihot_csv,
            image_id_for=IMAGE_ID_FOR,
            feature_columns=cat_cols,
        )
        r = proc.load_ratings().astype(np.float64)
        mu = r.mean()
        # Signed signal: +1 like, -1 dislike, 0 at-mean (preserves zero
        # for "no preference" so it doesn't get conflated with "disliked").
        signal = np.where(r > mu, 1.0, np.where(r < mu, -1.0, 0.0))
        nets[uid] = signal @ M  # (n_cats_kept,) - net (likes - dislikes) per cat

    net_df = pd.DataFrame(nets, index=cat_cols).T
    labels_df = (net_df >= min_net_likes).astype(int)
    return labels_df, net_df, dropped_cols


def _filter_features(
    X_df: pd.DataFrame, min_coverage: int,
) -> tuple[pd.DataFrame, list[str]]:
    if min_coverage <= 0:
        return X_df, []
    coverage = (X_df != 0).sum(axis=0)
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = [c for c in X_df.columns if c not in keep]
    return X_df[keep], dropped


def _build_logistic(C: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    solver="liblinear",      # robust at small N + L2
                    class_weight="balanced",  # handles label imbalance
                    max_iter=1000,
                ),
            ),
        ]
    )


def _scorable(y_k: np.ndarray, min_pos: int = 5, min_neg: int = 5) -> bool:
    """Drop labels that are essentially constant - LR can't learn them."""
    pos = int(y_k.sum())
    neg = int(len(y_k) - pos)
    return pos >= min_pos and neg >= min_neg


def _sweep_C(
    X: np.ndarray, y: np.ndarray, Cs: list[float], cv: KFold, cat_names: list[str],
) -> tuple[float, dict[float, float]]:
    """Pick C that maximizes macro AUC over scorable categories."""
    macro_auc: dict[float, float] = {}
    for C in Cs:
        aucs = []
        for k in range(y.shape[1]):
            y_k = y[:, k]
            if not _scorable(y_k):
                continue
            pipe = _build_logistic(C)
            proba = cross_val_predict(
                pipe, X, y_k, cv=cv, method="predict_proba", n_jobs=-1,
            )[:, 1]
            aucs.append(roc_auc_score(y_k, proba))
        macro_auc[C] = float(np.mean(aucs)) if aucs else float("nan")
    best_C = max(macro_auc, key=macro_auc.get)
    print("  [C] sweep:")
    for C, auc in macro_auc.items():
        marker = "  <-- best" if C == best_C else ""
        print(f"    C={C:>8g}: macro AUC = {auc:.4f}{marker}")
    return best_C, macro_auc


def _evaluate_per_category(
    X: np.ndarray, y: np.ndarray, C: float, cv: KFold, cat_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run per-category CV at fixed C, return (metrics_df, predictions_proba)."""
    proba_matrix = np.full(y.shape, np.nan, dtype=np.float64)
    rows = []
    for k, name in enumerate(cat_names):
        y_k = y[:, k]
        pos = int(y_k.sum())
        neg = int(len(y_k) - pos)
        if not _scorable(y_k):
            rows.append(
                {"category": name, "n_pos": pos, "n_neg": neg, "pos_rate": pos / len(y_k),
                 "auc": float("nan"), "ap": float("nan"),
                 "f1": float("nan"), "precision": float("nan"),
                 "recall": float("nan"), "accuracy": float("nan"),
                 "f1_baseline": float("nan")}
            )
            continue

        pipe = _build_logistic(C)
        proba = cross_val_predict(
            pipe, X, y_k, cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]
        proba_matrix[:, k] = proba
        pred = (proba >= 0.5).astype(int)

        # F1 of always-predict-positive baseline (sanity floor)
        f1_pos = f1_score(y_k, np.ones_like(y_k), zero_division=0)

        rows.append(
            {
                "category": name,
                "n_pos": pos,
                "n_neg": neg,
                "pos_rate": pos / len(y_k),
                "auc": roc_auc_score(y_k, proba),
                "ap": average_precision_score(y_k, proba),
                "f1": f1_score(y_k, pred, zero_division=0),
                "precision": precision_score(y_k, pred, zero_division=0),
                "recall": recall_score(y_k, pred, zero_division=0),
                "accuracy": accuracy_score(y_k, pred),
                "f1_baseline": f1_pos,
            }
        )
    return pd.DataFrame(rows).set_index("category"), proba_matrix


def run(
    *,
    features_path: Path,
    multihot_path: Path,
    canonical_path: Optional[Path],
    Cs: list[float],
    min_net_likes: int,
    min_feature_coverage: int,
    cv_folds: int,
    labels_out: Optional[Path],
    metrics_out: Optional[Path],
    save_predictions: bool,
    seed: int,
) -> None:
    print("=" * 72)
    print("Stage A: build signed net-like labels")
    print("=" * 72)
    print(f"  multihot:        {multihot_path}")
    print(f"  canonical T1:    {canonical_path}")
    print(f"  min_net_likes:   {min_net_likes}  (label=1 if net likes - dislikes "
          f">= this, where like = above user's personal mean)")

    rating_csv_paths = discover_users(CORPUS_ROOTS)
    print(f"  users found:     {len(rating_csv_paths)}")
    labels, nets, dropped_cols = build_above_mean_labels(
        rating_csv_paths, multihot_path,
        min_net_likes=min_net_likes,
        canonical_path=canonical_path,
    )
    if dropped_cols:
        print(f"  dropped {len(dropped_cols)} category column(s) "
              f"(non-canonical or zero exposure):")
        for c in dropped_cols:
            print(f"    - {c}")
    print(f"  kept {labels.shape[1]} category columns")
    print(f"  labels shape:    {labels.shape}, mean positive rate: "
          f"{labels.values.mean():.3f}")
    print(f"  net score range: [{int(nets.values.min())}, "
          f"{int(nets.values.max())}], mean: {nets.values.mean():.2f}")
    if labels_out:
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(labels_out, index_label="user_id")
        nets_out = labels_out.with_name(labels_out.stem + "_net.csv")
        nets.to_csv(nets_out, index_label="user_id")
        print(f"  wrote {labels_out}")
        print(f"  wrote {nets_out}")

    print()
    print("=" * 72)
    print("Stage B: align + filter")
    print("=" * 72)
    feats = pd.read_csv(features_path, index_col="user_id")
    common = sorted(set(labels.index) & set(feats.index))
    print(f"  features: {feats.shape}, common users: {len(common)}")

    X_df, dropped_feats = _filter_features(feats.loc[common], min_feature_coverage)
    if dropped_feats:
        print(f"  dropped {len(dropped_feats)} feature(s) with <{min_feature_coverage} "
              f"non-zero users (kept {X_df.shape[1]})")

    y_df = labels.loc[common]
    cat_names = list(y_df.columns)
    n_scorable = sum(_scorable(y_df.values[:, k]) for k in range(y_df.shape[1]))
    n_skipped = y_df.shape[1] - n_scorable
    print(f"  {n_scorable} of {y_df.shape[1]} categories scorable "
          f"(>= 5 pos AND >= 5 neg); {n_skipped} skipped as too imbalanced")

    X = X_df.values.astype(np.float64)
    y = y_df.values.astype(np.int64)
    print(f"  final X={X.shape}, y={y.shape}")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    print()
    print("=" * 72)
    print("Stage C: C sweep (macro AUC)")
    print("=" * 72)
    best_C, _ = _sweep_C(X, y, Cs, cv, cat_names)

    print()
    print("=" * 72)
    print(f"Stage D: per-category metrics at C={best_C}")
    print("=" * 72)
    per_cat, proba = _evaluate_per_category(X, y, best_C, cv, cat_names)

    if metrics_out:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        per_cat.to_csv(metrics_out)
        print(f"  wrote {metrics_out}")
    if save_predictions:
        DEFAULT_PREDS_OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(proba, index=common, columns=cat_names) \
            .to_csv(DEFAULT_PREDS_OUT, index_label="user_id")
        print(f"  wrote {DEFAULT_PREDS_OUT}")

    valid = per_cat.dropna(subset=["auc"])
    print()
    print("=" * 72)
    print(f"Summary ({len(valid)} scorable / {len(per_cat)} categories)")
    print("=" * 72)
    print(f"  macro AUC:        {valid.auc.mean():.4f}")
    print(f"  median AUC:       {valid.auc.median():.4f}")
    print(f"  % cats AUC > 0.55: {(valid.auc > 0.55).mean() * 100:.1f}%")
    print(f"  % cats AUC > 0.60: {(valid.auc > 0.60).mean() * 100:.1f}%")
    print(f"  macro AP:         {valid.ap.mean():.4f}")
    print(f"  macro F1:         {valid.f1.mean():.4f}")
    print(f"  macro F1-baseline: {valid.f1_baseline.mean():.4f}  (always-predict-positive)")
    print()
    print("Top 10 by AUC:")
    cols = ["auc", "ap", "f1", "precision", "recall", "n_pos"]
    print(valid.sort_values("auc", ascending=False).head(10)[cols].round(4).to_string())
    print()
    print("Bottom 10 by AUC:")
    print(valid.sort_values("auc").head(10)[cols].round(4).to_string())


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--multihot", type=Path, default=DEFAULT_MULTIHOT)
    parser.add_argument(
        "--canonical", type=Path, default=DEFAULT_CANONICAL,
        help="Path to canonical IAB-t1 list (one cat per line). Multihot "
             "columns not in this list are dropped. Pass empty string to "
             "skip canonical filtering.",
    )
    parser.add_argument(
        "--min-net-likes", type=int, default=1,
        help="Min net (likes - dislikes) for cat-k ads to count as y[u,k]=1. "
             "Default 1 (more likes than dislikes). Higher = stricter signal "
             "= more balanced labels = stronger AUC but fewer scorable cats.",
    )
    parser.add_argument("--Cs", nargs="+", type=float,
                        default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--min-feature-coverage", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    canonical_path = args.canonical if str(args.canonical) else None
    run(
        features_path=args.features,
        multihot_path=args.multihot,
        canonical_path=canonical_path,
        Cs=list(args.Cs),
        min_net_likes=args.min_net_likes,
        min_feature_coverage=args.min_feature_coverage,
        cv_folds=args.cv_folds,
        labels_out=None if args.no_save else args.labels_out,
        metrics_out=None if args.no_save else args.metrics_out,
        save_predictions=args.save_predictions and not args.no_save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
