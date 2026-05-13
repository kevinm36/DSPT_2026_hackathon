"""Train + evaluate the (user, ad) "sentient" pair-level model.

What this script does end-to-end
--------------------------------
1. Materialise the pair dataset via ``sentient_dataset.build_pair_dataset``
   (one row per (user, ad) pair, label = ``rating > user_mean``).
2. Run a GroupKFold-by-user 5-fold CV (so test users are never in train).
3. For each estimator (Logistic Regression, HistGradientBoostingClassifier),
   sweep a small hyperparameter grid and keep the best by **pair-level macro
   AUC**.
4. Report two metric families on the best model's out-of-fold predictions:
     - **pair-level**: AUC / AP / accuracy on every (user, ad) row
       directly. This is the cold-start "will user U like ad A" metric.
     - **per-category aggregated**: average the 15 ad-level probabilities
       inside each (user, category) cell, then evaluate AUC / AP / Spearman
       against the same ``net_likes >= 1`` label the content IAB models use
       in ``Data/model_comparison.csv``. Lets you stack rows side by side.
5. Refit the best config on the full dataset and dump a joblib bundle with
   the feature schema baked in, ready for the demo.

Two user-feature profiles run in sequence by default for ablation:
    * **compact**          : 50 cluster + 10 B5 + 29 ad design ~= 89 features
    * **with_demographics**: compact + 136 demographic = ~225 features

Run::

    python -m src.model.train_sentient
    python -m src.model.train_sentient --profiles compact          # one profile only
    python -m src.model.train_sentient --models lr                 # skip GBM
    python -m src.model.train_sentient --cv-folds 3 --max-iter 100 # quick smoke
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import IMAGES_PER_CATEGORY, NUM_CATEGORIES, REPO_ROOT
from src.model.sentient_dataset import PairDataset, build_pair_dataset


DEFAULT_OUT_DIR = REPO_ROOT / "Data"
DEFAULT_MODELS_DIR = REPO_ROOT / "Data/models"

# ADS-16 native product taxonomy (the 20 columns of every UXXXX-RT.csv).
# This is the natural per-category axis for the pair model -- 15 ads per cell,
# one cell per (user, product cat). NOT to be confused with the IAB-t1
# taxonomy used by the content model in train_logistic.py / compare_models.py
# (different axis, so side-by-side AUCs are "same metric, different task").
ADS16_PRODUCT_CATS: list[str] = [
    "Clothing & Shoes", "Automotive", "Baby Products", "Health & Beauty",
    "Media (BMVD)", "Consumer Electronics", "Console & Video Games",
    "DIY & Tools", "Garden & Outdoor living", "Grocery", "Kitchen & Home",
    "Betting", "Jewellery & Watches", "Musical Instruments",
    "Office Products", "Pet Supplies", "Computer Software",
    "Sports & Outdoors", "Toys & Games", "Dating Sites",
]
assert len(ADS16_PRODUCT_CATS) == NUM_CATEGORIES

PROFILES: dict[str, dict] = {
    "compact": dict(
        include_clusters=True, include_b5=True, include_demographics=False,
    ),
    "with_demographics": dict(
        include_clusters=True, include_b5=True, include_demographics=True,
    ),
}

# Hyperparameter grids. Kept small on purpose: the 36k-row dataset trains
# fast but every extra config multiplies the CV time.
LR_GRID: list[float] = [0.01, 0.1, 1.0]
GBM_GRID: list[dict] = [
    {"learning_rate": 0.05, "max_depth": 4,    "max_iter": 200},
    {"learning_rate": 0.10, "max_depth": 6,    "max_iter": 200},
    {"learning_rate": 0.10, "max_depth": None, "max_iter": 200},
]


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def _build_lr(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("model", LogisticRegression(
            C=C, solver="liblinear", class_weight="balanced", max_iter=2000,
        )),
    ])


def _build_gbm(params: dict) -> HistGradientBoostingClassifier:
    # sklearn 1.0.2 HGBT does not support class_weight; AUC/AP are
    # threshold-free so the 34/66 imbalance is harmless for ranking.
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        max_iter=params["max_iter"],
        random_state=0,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _pair_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Pair-level binary metrics on OOF predictions."""
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, (y_score >= 0.5).astype(int))
    return {"auc": float(auc), "ap": float(ap), "accuracy": float(acc)}


def _per_category_metrics(
    ds: PairDataset, y_score: np.ndarray, *, min_net_likes: int = 1,
    cat_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Aggregate pair-level scores to per-(user, ADS-16 product cat) and score.

    For each (user, cat) cell (20 cells per user, 15 ads each) we compute:
        * predicted score = mean of the 15 ad-level scores in that cell
        * binary label    = ``net_likes >= min_net_likes`` where ``net_likes``
          is the same signed-signal sum the content IAB model uses
          (+1 if rating > user_mean, -1 if <, 0 otherwise).

    Note this aggregates over the 20 ADS-16 product categories (the rating-CSV
    column axis), not the 34 IAB-t1 content categories used in
    ``Data/model_comparison.csv``. Same labelling rule, different category axis
    -- so AUC numbers are on the same scale but answer slightly different
    questions.
    """
    n_users = len(ds.user_ids)
    n_cats = NUM_CATEGORIES
    score_uc = y_score.reshape(n_users, n_cats, IMAGES_PER_CATEGORY).mean(axis=2)

    ratings_uc = ds.ratings.reshape(n_users, n_cats, IMAGES_PER_CATEGORY)
    means_u = ds.user_means.reshape(n_users, 1, 1)
    signed = np.where(
        ratings_uc > means_u, 1,
        np.where(ratings_uc < means_u, -1, 0),
    )
    net_likes = signed.sum(axis=2)
    label_uc = (net_likes >= min_net_likes).astype(int)
    cont_uc = net_likes.astype(np.float64)

    if cat_names is None:
        cat_names = [f"cat_{i}" for i in range(n_cats)]

    rows = []
    for k in range(n_cats):
        y_b = label_uc[:, k]
        y_n = cont_uc[:, k]
        s = score_uc[:, k]
        if y_b.sum() == 0 or y_b.sum() == len(y_b):
            rows.append({
                "category": cat_names[k],
                "auc": np.nan, "ap": np.nan, "spearman": np.nan,
                "n_pos": int(y_b.sum()),
            })
            continue
        rho, _ = spearmanr(y_n, s)
        rows.append({
            "category": cat_names[k],
            "auc": float(roc_auc_score(y_b, s)),
            "ap": float(average_precision_score(y_b, s)),
            "spearman": float(rho),
            "n_pos": int(y_b.sum()),
        })
    return pd.DataFrame(rows).set_index("category")


# --------------------------------------------------------------------------- #
# CV sweeps
# --------------------------------------------------------------------------- #
def _oof_predict_proba(
    estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv: GroupKFold,
) -> np.ndarray:
    """Out-of-fold P(y=1) via cross_val_predict."""
    probs = cross_val_predict(
        estimator, X, y, cv=cv.split(X, y, groups),
        method="predict_proba", n_jobs=-1,
    )
    return probs[:, 1]


def _sweep_lr(ds: PairDataset, cv: GroupKFold) -> tuple[float, np.ndarray]:
    print("  [LR] sweep:")
    best = (None, -np.inf, None)
    for C in LR_GRID:
        pipe = _build_lr(C)
        scores = _oof_predict_proba(pipe, ds.X, ds.y, ds.groups, cv)
        auc = roc_auc_score(ds.y, scores)
        print(f"    C={C:>5g} -> pair AUC={auc:.4f}")
        if auc > best[1]:
            best = (C, auc, scores)
    print(f"    -> best C={best[0]} (AUC={best[1]:.4f})")
    return best[0], best[2]


def _sweep_gbm(ds: PairDataset, cv: GroupKFold) -> tuple[dict, np.ndarray]:
    print("  [GBM] sweep:")
    best = (None, -np.inf, None)
    for params in GBM_GRID:
        gbm = _build_gbm(params)
        scores = _oof_predict_proba(gbm, ds.X, ds.y, ds.groups, cv)
        auc = roc_auc_score(ds.y, scores)
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"    {label} -> pair AUC={auc:.4f}")
        if auc > best[1]:
            best = (params, auc, scores)
    print(f"    -> best params={best[0]} (AUC={best[1]:.4f})")
    return best[0], best[2]


# --------------------------------------------------------------------------- #
# Bundle export
# --------------------------------------------------------------------------- #
def _build_bundle(
    *, model_name: str, profile: str, best_hparam, ds: PairDataset,
    estimator_factory: Callable, score_units: str,
) -> dict:
    fitted = estimator_factory(best_hparam)
    fitted.fit(ds.X, ds.y)
    return {
        "model_name": model_name,
        "profile": profile,
        "best_hparam": best_hparam,
        "feature_names": list(ds.feature_names),
        "pipeline": fitted,
        "score_kind": "predict_proba",
        "score_units": score_units,
        "training_metadata": {
            "n_users": len(ds.user_ids),
            "n_pairs": int(len(ds.y)),
            "n_features": int(ds.X.shape[1]),
            "pos_rate": float(ds.y.mean()),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_versions": {
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }


def _save(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"  saved -> {path}")


# --------------------------------------------------------------------------- #
# Run one profile
# --------------------------------------------------------------------------- #
def run_profile(
    profile_name: str, *,
    models_to_train: list[str], cv_folds: int,
    cat_names: list[str], save_models: bool, models_dir: Path,
    out_dir: Path,
) -> tuple[dict[str, dict], dict[str, pd.DataFrame]]:
    """Train + evaluate every requested model for a single user-side profile.

    Returns ``(pair_metrics_per_model, per_cat_per_model)``.
    """
    print()
    print("=" * 72)
    print(f"PROFILE: {profile_name}")
    print("=" * 72)

    ds = build_pair_dataset(**PROFILES[profile_name])
    print(
        f"dataset: {len(ds.user_ids)} users x {len(ds.image_ids)} ads = "
        f"{len(ds.y)} pairs; {ds.X.shape[1]} features; "
        f"pos rate = {ds.y.mean():.1%}"
    )

    cv = GroupKFold(n_splits=cv_folds)
    pair_metrics: dict[str, dict] = {}
    per_cat: dict[str, pd.DataFrame] = {}

    n_feat = int(ds.X.shape[1])

    if "lr" in models_to_train:
        print("\n--- Logistic Regression ---")
        best_C, oof = _sweep_lr(ds, cv)
        pair_metrics["sentient_lr"] = {
            "profile": profile_name, "n_features": n_feat,
            "hparam": f"C={best_C}",
            **_pair_metrics(ds.y, oof),
        }
        per_cat["sentient_lr"] = _per_category_metrics(ds, oof, cat_names=cat_names)
        if save_models:
            bundle = _build_bundle(
                model_name="sentient_lr", profile=profile_name,
                best_hparam={"C": best_C}, ds=ds,
                estimator_factory=lambda h: _build_lr(h["C"]),
                score_units="P(rating > user_mean), in [0, 1]",
            )
            _save(bundle, models_dir / f"sentient_lr_{profile_name}.joblib")

    if "gbm" in models_to_train:
        print("\n--- HistGradientBoostingClassifier ---")
        best_params, oof = _sweep_gbm(ds, cv)
        pair_metrics["sentient_gbm"] = {
            "profile": profile_name, "n_features": n_feat,
            "hparam": ", ".join(f"{k}={v}" for k, v in best_params.items()),
            **_pair_metrics(ds.y, oof),
        }
        per_cat["sentient_gbm"] = _per_category_metrics(ds, oof, cat_names=cat_names)
        if save_models:
            bundle = _build_bundle(
                model_name="sentient_gbm", profile=profile_name,
                best_hparam=best_params, ds=ds,
                estimator_factory=_build_gbm,
                score_units="P(rating > user_mean), in [0, 1]",
            )
            _save(bundle, models_dir / f"sentient_gbm_{profile_name}.joblib")

    return pair_metrics, per_cat


# --------------------------------------------------------------------------- #
# Stitch everything into the comparison output files
# --------------------------------------------------------------------------- #
def _write_pair_metrics(
    all_pair: dict[str, dict[str, dict]], out_path: Path,
) -> None:
    """``all_pair[profile][model] = {auc, ap, accuracy, hparam, ...}``."""
    rows = []
    for profile, mdict in all_pair.items():
        for model, m in mdict.items():
            rows.append({
                "profile": profile, "model": model,
                "n_features": m["n_features"],
                "hparam": m["hparam"],
                "auc": m["auc"], "ap": m["ap"], "accuracy": m["accuracy"],
            })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nPair-level metrics -> {out_path}")
    print(df.to_string(index=False))


def _write_per_category(
    all_per_cat: dict[str, dict[str, pd.DataFrame]],
    out_path: Path, cat_names: list[str],
) -> None:
    """One row per IAB-t1 category, columns = AUC/AP for each (profile, model).

    Same shape as ``Data/model_comparison.csv`` so you can paste rows
    side-by-side with the content IAB models.
    """
    out = pd.DataFrame(index=pd.Index(cat_names, name="category"))
    for profile, mdict in all_per_cat.items():
        for model, df in mdict.items():
            tag = f"{model}_{profile}"
            for col in ("auc", "ap", "spearman"):
                out[f"{tag}_{col}"] = df[col]
            out[f"{tag}_n_pos"] = df["n_pos"]

    auc_cols = [c for c in out.columns if c.endswith("_auc")]
    out["best_by_auc"] = out[auc_cols].idxmax(axis=1).str.replace("_auc", "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    print(f"\nPer-category aggregated metrics -> {out_path}")

    summary = pd.DataFrame({
        col.replace("_auc", ""): [out[col].mean(skipna=True)]
        for col in auc_cols
    }, index=["macro_AUC"]).T
    print("\nMacro-AUC across IAB-t1 categories:")
    print(summary.round(4).to_string())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> None:
    args = _parse(argv)

    cat_names = ADS16_PRODUCT_CATS

    all_pair: dict[str, dict[str, dict]] = {}
    all_per_cat: dict[str, dict[str, pd.DataFrame]] = {}

    for profile in args.profiles:
        if profile not in PROFILES:
            raise ValueError(
                f"unknown profile {profile!r}; choose from {sorted(PROFILES)}"
            )
        pair_m, per_cat = run_profile(
            profile,
            models_to_train=args.models,
            cv_folds=args.cv_folds,
            cat_names=cat_names,
            save_models=args.save_models,
            models_dir=args.models_dir,
            out_dir=args.out_dir,
        )
        all_pair[profile] = pair_m
        all_per_cat[profile] = per_cat

    _write_pair_metrics(
        all_pair, args.out_dir / "sentient_pair_metrics.csv",
    )
    _write_per_category(
        all_per_cat, args.out_dir / "sentient_per_category_metrics.csv",
        cat_names,
    )


def _parse(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--profiles", nargs="+", default=list(PROFILES),
        help=f"User-side profiles to train. Default: {list(PROFILES)}",
    )
    p.add_argument(
        "--models", nargs="+", default=["lr", "gbm"], choices=["lr", "gbm"],
        help="Estimators to train. Default: both.",
    )
    p.add_argument("--cv-folds", type=int, default=5,
                   help="GroupKFold splits (groups = user_id). Default 5.")
    p.add_argument("--save-models", action="store_true",
                   help="Refit on full data and dump joblib bundles.")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
