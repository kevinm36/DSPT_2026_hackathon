"""Ridge baseline: predict per-(user, category) interest from user features.

Given:
  X = user_features.csv          (120 users x 136 features)
  y = interest_matrix            (120 users x  K categories)
                                 built from per-image ratings + multi-hot

train a regularized linear model and report cross-validated metrics per
category.

Run with::

    python -m src.model.train_ridge

Common knobs::

    python -m src.model.train_ridge --multihot Data/ads16_multihot_t1.csv
    python -m src.model.train_ridge --aggregate frac_positive --rating-norm none
    python -m src.model.train_ridge --alphas 0.1 1 10 100 --cv-folds 5
    python -m src.model.train_ridge --save-predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    CORPUS_ROOTS,
    IMAGE_ID_FOR,
    IMAGES_PER_CATEGORY,
    NUM_CATEGORIES,
    discover_users,
)
from src.model.interest_matrix import (
    AGGREGATES,
    RATING_NORMS,
    build_interest_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MULTIHOT = REPO_ROOT / "Data/ads16_multihot_t1.csv"
DEFAULT_FEATURES = REPO_ROOT / "Data/user_features.csv"
DEFAULT_INTEREST_OUT = REPO_ROOT / "Data/user_interest_matrix.csv"
DEFAULT_RIDGE_METRICS = REPO_ROOT / "Data/ridge_per_category_metrics.csv"
DEFAULT_KNN_METRICS = REPO_ROOT / "Data/knn_per_category_metrics.csv"
DEFAULT_RIDGE_PREDS = REPO_ROOT / "Data/ridge_cv_predictions.csv"
DEFAULT_KNN_PREDS = REPO_ROOT / "Data/knn_cv_predictions.csv"


def _build_ridge(alpha: float) -> Pipeline:
    # solver="svd" avoids the scipy>=1.11 incompatibility with sklearn's
    # default cholesky solver (which calls linalg.solve(sym_pos=True)).
    # SVD is exact and fast at this matrix size (~120 x 136).
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True)),
            ("model", MultiOutputRegressor(Ridge(alpha=alpha, solver="svd"))),
        ]
    )


def _build_knn(k: int) -> Pipeline:
    # Cosine distance after standardization is the safest choice when features
    # are a mix of binary one-hots and numeric (age, income).
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True)),
            ("model", KNeighborsRegressor(n_neighbors=k, metric="cosine",
                                          weights="distance")),
        ]
    )


def _filter_features(
    X_df: pd.DataFrame, min_coverage: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with fewer than ``min_coverage`` non-zero rows.

    Numeric columns (age, income) always survive because they're rarely 0 -
    the filter only bites the 130+ multi-hot one-hots that are mostly zero.
    """
    if min_coverage <= 0:
        return X_df, []
    coverage = (X_df != 0).sum(axis=0)
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = [c for c in X_df.columns if c not in keep]
    return X_df[keep], dropped


def _sweep_hyperparam(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    grid: list,
    builder,
    cv: KFold,
) -> tuple[object, dict]:
    """Sweep one hyperparameter; return ``(best_value, {value: macro_R2})``."""
    macro_r2: dict = {}
    for v in grid:
        pipe = builder(v)
        scores = cross_val_score(
            pipe, X, y, cv=cv, scoring="r2",
            n_jobs=-1, error_score="raise",
        )
        macro_r2[v] = float(scores.mean())
    best_v = max(macro_r2, key=macro_r2.get)
    print(f"  [{name}] sweep:")
    for v, r2 in macro_r2.items():
        marker = "  <-- best" if v == best_v else ""
        print(f"    {name}={v!s:>8}: macro R² = {r2:+.4f}{marker}")
    return best_v, macro_r2


def _per_category_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    category_names: list[str],
) -> pd.DataFrame:
    """Return one-row-per-category metrics: R², RMSE, MAE, Spearman, coverage."""
    n_cats = y_true.shape[1]
    rows = []
    for i in range(n_cats):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        var = yt.var()
        if var == 0:
            r2 = float("nan")
            rho = float("nan")
        else:
            r2 = r2_score(yt, yp)
            rho_val, _ = spearmanr(yt, yp)
            rho = float(rho_val) if rho_val == rho_val else float("nan")  # NaN check
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        rows.append(
            {
                "category": category_names[i],
                "target_var": float(var),
                "r2": r2,
                "spearman": rho,
                "rmse": rmse,
                "mae": mae,
            }
        )
    return pd.DataFrame(rows).set_index("category")


def run(
    *,
    features_path: Path,
    multihot_path: Path,
    rating_norm: str,
    aggregate: str,
    alphas: list[float],
    knn_ks: list[int],
    min_feature_coverage: int,
    cv_folds: int,
    interest_out: Optional[Path],
    ridge_metrics_out: Optional[Path],
    knn_metrics_out: Optional[Path],
    save_predictions: bool,
    seed: int,
) -> None:
    print("=" * 72)
    print("Stage A: build interest matrix")
    print("=" * 72)
    print(f"  multihot:    {multihot_path}")
    print(f"  rating_norm: {rating_norm}")
    print(f"  aggregate:   {aggregate}")

    rating_csv_paths = discover_users(CORPUS_ROOTS)
    print(f"  users found: {len(rating_csv_paths)}")

    interest = build_interest_matrix(
        rating_csv_paths=rating_csv_paths,
        multihot_csv=multihot_path,
        image_id_for=IMAGE_ID_FOR,
        num_categories=NUM_CATEGORIES,
        images_per_category=IMAGES_PER_CATEGORY,
        rating_norm=rating_norm,
        aggregate=aggregate,
    )
    print(f"  interest matrix shape: {interest.shape}")
    if interest_out:
        interest_out.parent.mkdir(parents=True, exist_ok=True)
        interest.to_csv(interest_out, index_label="user_id")
        print(f"  wrote {interest_out}")

    print()
    print("=" * 72)
    print("Stage B: align + filter")
    print("=" * 72)
    feats = pd.read_csv(features_path, index_col="user_id")
    common = sorted(set(interest.index) & set(feats.index))
    print(f"  features:     {feats.shape}")
    print(f"  common users: {len(common)}")
    if not common:
        raise ValueError("no overlapping users between features and interest matrix")

    X_df_full = feats.loc[common]
    y_df = interest.loc[common]

    # Drop dead user-feature columns (mostly-zero one-hots add only noise).
    X_df, dropped_feats = _filter_features(X_df_full, min_feature_coverage)
    if dropped_feats:
        print(
            f"  dropped {len(dropped_feats)} feature(s) with <{min_feature_coverage} "
            f"non-zero users (kept {X_df.shape[1]})."
        )

    # Drop zero-variance target columns (constants - unscoreable).
    variances = y_df.var(axis=0)
    keep = variances[variances > 0].index.tolist()
    dropped_cats = [c for c in y_df.columns if c not in keep]
    if dropped_cats:
        print(f"  dropped {len(dropped_cats)} zero-variance categor(ies): "
              f"{dropped_cats}")
    y_df = y_df[keep]

    X = X_df.values.astype(np.float64)
    y = y_df.values.astype(np.float64)
    cat_names = list(y_df.columns)
    print(f"  final X={X.shape}, y={y.shape}")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    print()
    print("=" * 72)
    print("Stage C: hyperparameter sweeps (5-fold CV macro-R²)")
    print("=" * 72)
    best_alpha, _ = _sweep_hyperparam("alpha", X, y, alphas, _build_ridge, cv)
    best_k, _ = _sweep_hyperparam("k", X, y, knn_ks, _build_knn, cv)
    print(f"  selected ridge alpha = {best_alpha}, knn k = {best_k}")

    # Properly cross-validated trivial baseline: predict per-fold-train mean.
    baseline_pred = np.zeros_like(y)
    for tr, te in cv.split(y):
        baseline_pred[te] = y[tr].mean(axis=0)

    print()
    print("=" * 72)
    print("Stage D: head-to-head CV predictions")
    print("=" * 72)
    results: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for name, pipe in [
        ("ridge", _build_ridge(best_alpha)),
        ("knn", _build_knn(best_k)),
    ]:
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
        per_cat = _per_category_metrics(y, y_pred, cat_names)
        per_cat["baseline_r2"] = [
            r2_score(y[:, i], baseline_pred[:, i]) if y[:, i].var() > 0 else float("nan")
            for i in range(y.shape[1])
        ]
        per_cat["beats_baseline"] = (per_cat["r2"] > per_cat["baseline_r2"]).astype(int)
        results[name] = (y_pred, per_cat)

    # Save artifacts
    if ridge_metrics_out:
        ridge_metrics_out.parent.mkdir(parents=True, exist_ok=True)
        results["ridge"][1].to_csv(ridge_metrics_out)
        print(f"  wrote {ridge_metrics_out}")
    if knn_metrics_out:
        knn_metrics_out.parent.mkdir(parents=True, exist_ok=True)
        results["knn"][1].to_csv(knn_metrics_out)
        print(f"  wrote {knn_metrics_out}")
    if save_predictions:
        for name, default in [("ridge", DEFAULT_RIDGE_PREDS), ("knn", DEFAULT_KNN_PREDS)]:
            default.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results[name][0], index=common, columns=cat_names) \
                .to_csv(default, index_label="user_id")
            print(f"  wrote {default}")

    # Head-to-head summary
    print()
    print("=" * 72)
    print("Head-to-head summary")
    print("=" * 72)
    summary_rows = []
    for name, (_, per_cat) in results.items():
        valid = per_cat.dropna(subset=["r2"])
        summary_rows.append(
            {
                "model": name,
                "macro_R²": valid.r2.mean(),
                "median_R²": valid.r2.median(),
                "% cats R²>0": (valid.r2 > 0).mean() * 100,
                "macro_RMSE": valid.rmse.mean(),
                "macro_MAE": valid.mae.mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows).set_index("model")
    print(summary_df.round(4).to_string())

    # Per-category winner table (top 10 by combined-best R²)
    print()
    print("Per-category R² (sorted by max across models, top 15):")
    cmp_df = pd.DataFrame({
        "ridge_R²": results["ridge"][1]["r2"],
        "knn_R²":   results["knn"][1]["r2"],
    })
    cmp_df["best"] = cmp_df.max(axis=1)
    cmp_df["winner"] = np.where(cmp_df["ridge_R²"] >= cmp_df["knn_R²"], "ridge", "knn")
    print(cmp_df.sort_values("best", ascending=False)
                 .head(15)
                 .drop(columns="best")
                 .round(4).to_string())


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--multihot", type=Path, default=DEFAULT_MULTIHOT,
                        help="Per-image multi-hot CSV (default: t1).")
    parser.add_argument("--rating-norm", default="center", choices=sorted(RATING_NORMS),
                        help="Per-user rating transform applied before aggregation.")
    parser.add_argument("--aggregate", default="mean", choices=sorted(AGGREGATES),
                        help="How to summarize a user's ratings within a category.")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[10.0, 100.0, 1000.0, 10000.0, 100000.0],
                        help="Ridge alpha values to sweep.")
    parser.add_argument("--knn-ks", nargs="+", type=int,
                        default=[3, 5, 10, 15, 20, 30],
                        help="kNN n_neighbors values to sweep.")
    parser.add_argument("--min-feature-coverage", type=int, default=5,
                        help="Drop user-feature columns with <N non-zero users "
                             "(set to 0 to disable). Defaults to 5.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interest-out", type=Path, default=DEFAULT_INTEREST_OUT)
    parser.add_argument("--ridge-metrics-out", type=Path, default=DEFAULT_RIDGE_METRICS)
    parser.add_argument("--knn-metrics-out", type=Path, default=DEFAULT_KNN_METRICS)
    parser.add_argument("--save-predictions", action="store_true",
                        help="Also write CV predictions for both models.")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write any output CSVs (just print to stdout).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run(
        features_path=args.features,
        multihot_path=args.multihot,
        rating_norm=args.rating_norm,
        aggregate=args.aggregate,
        alphas=list(args.alphas),
        knn_ks=list(args.knn_ks),
        min_feature_coverage=args.min_feature_coverage,
        cv_folds=args.cv_folds,
        interest_out=None if args.no_save else args.interest_out,
        ridge_metrics_out=None if args.no_save else args.ridge_metrics_out,
        knn_metrics_out=None if args.no_save else args.knn_metrics_out,
        save_predictions=args.save_predictions and not args.no_save,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
