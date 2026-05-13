"""Head-to-head comparison: Logistic Regression vs Ridge vs kNN.

All three models predict a continuous *score* per (user, category):
  - LR:    P(y=1)        from LogisticRegression(predict_proba)
  - Ridge: predicted net  from Ridge regression on signed net-like scores
  - kNN:   predicted net  from KNeighborsRegressor on signed net-like scores

Apples-to-apples evaluation via threshold-free ranking metrics on the same
ground truth:
  - AUC      (predicted score vs binary label "user has net positive interest")
  - AP       (precision-recall AUC on the same)
  - Spearman (predicted score vs continuous signed net-like score)

Each model gets its own hyperparameter sweep (selected by macro AUC), then is
evaluated on the held-out folds at its best setting.

Run::

    python -m src.model.compare_models
    python -m src.model.compare_models --min-net-likes 3 --cv-folds 5
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader.agent_processing.categories_t1 import (
    DEFAULT_CATEGORIES_PATH as DEFAULT_T1_PATH,
)
from src.model.train_logistic import (
    DEFAULT_FEATURES,
    DEFAULT_MULTIHOT,
    _filter_features,
    _scorable,
    build_above_mean_labels,
)
from src.data_loader import CORPUS_ROOTS, discover_users


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "Data/model_comparison.csv"
DEFAULT_PREDS_OUT = REPO_ROOT / "Data/model_predictions_sample.csv"
DEFAULT_MODELS_DIR = REPO_ROOT / "Data/models"

# Filename for each saved model bundle. Loader reads these by key.
MODEL_FILES: dict[str, str] = {
    "LR":    "lr_model.joblib",
    "Ridge": "ridge_model.joblib",
    "kNN":   "knn_model.joblib",
}

# How each saved bundle should be invoked at predict time. Stored in the
# bundle so the loader knows whether to call predict_proba or predict.
SCORE_KIND: dict[str, str] = {
    "LR":    "predict_proba",
    "Ridge": "predict",
    "kNN":   "predict",
}

SCORE_UNITS: dict[str, str] = {
    "LR":    "P(net_likes >= min_net_likes), in [0, 1]",
    "Ridge": "predicted net_likes (signed real)",
    "kNN":   "predicted net_likes (signed real)",
}


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #

def _build_logreg(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("model", LogisticRegression(
            C=C, solver="liblinear", class_weight="balanced", max_iter=1000,
        )),
    ])


def _build_ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("model", Ridge(alpha=alpha, solver="svd")),
    ])


def _build_knn(k: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("model", KNeighborsRegressor(
            n_neighbors=k, metric="cosine", weights="distance",
        )),
    ])


# --------------------------------------------------------------------------- #
# Per-category prediction
# --------------------------------------------------------------------------- #

def _predict_logreg(X: np.ndarray, y_bin: np.ndarray, pipe: Pipeline,
                    cv: KFold) -> np.ndarray:
    """Return probability scores in [0, 1]."""
    return cross_val_predict(
        pipe, X, y_bin, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]


def _predict_regressor(X: np.ndarray, y_cont: np.ndarray, pipe: Pipeline,
                       cv: KFold) -> np.ndarray:
    """Return continuous regression score (no proba available)."""
    return cross_val_predict(pipe, X, y_cont, cv=cv, n_jobs=-1)


def _eval_one_category(y_bin: np.ndarray, y_net: np.ndarray,
                       pred: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_bin, pred)
    ap = average_precision_score(y_bin, pred)
    rho, _ = spearmanr(y_net, pred)
    return {"auc": float(auc), "ap": float(ap), "spearman": float(rho)}


def _per_category_scores(
    X: np.ndarray, y_bin: np.ndarray, y_net: np.ndarray,
    cat_names: list[str], cv: KFold,
    *, pipe_factory, hparam: float | int, predict_fn, target_for_predict: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run per-category CV and return per-cat metrics + predictions."""
    rows = []
    preds: dict[str, np.ndarray] = {}
    target = y_bin if target_for_predict == "bin" else y_net
    for k_idx, name in enumerate(cat_names):
        y_b = y_bin[:, k_idx]
        y_n = y_net[:, k_idx]
        if not _scorable(y_b):
            rows.append({"category": name, "auc": np.nan, "ap": np.nan,
                         "spearman": np.nan})
            preds[name] = np.full(len(y_b), np.nan)
            continue
        pipe = pipe_factory(hparam)
        pred = predict_fn(X, target[:, k_idx], pipe, cv)
        preds[name] = pred
        rows.append({"category": name, **_eval_one_category(y_b, y_n, pred)})
    return pd.DataFrame(rows).set_index("category"), preds


# --------------------------------------------------------------------------- #
# Hyperparameter sweep
# --------------------------------------------------------------------------- #

def _sweep(
    name: str, hparams: list, X: np.ndarray, y_bin: np.ndarray, y_net: np.ndarray,
    cat_names: list[str], cv: KFold,
    *, pipe_factory, predict_fn, target_for_predict: str,
) -> tuple[float | int, pd.DataFrame, dict[str, np.ndarray]]:
    print(f"  [{name}] hyperparameter sweep:")
    macros = []
    for h in hparams:
        per_cat, preds = _per_category_scores(
            X, y_bin, y_net, cat_names, cv,
            pipe_factory=pipe_factory, hparam=h,
            predict_fn=predict_fn, target_for_predict=target_for_predict,
        )
        macro_auc = per_cat.auc.mean(skipna=True)
        macros.append((h, macro_auc, per_cat, preds))
        print(f"    {name}={h:>8g}: macro AUC = {macro_auc:.4f}")
    best_h, best_auc, best_per_cat, best_preds = max(macros, key=lambda x: x[1])
    print(f"    -> best {name}={best_h} (macro AUC = {best_auc:.4f})")
    return best_h, best_per_cat, best_preds


# --------------------------------------------------------------------------- #
# Final-fit bundling for export
# --------------------------------------------------------------------------- #

def _fit_per_category_final(
    X: np.ndarray, y: np.ndarray, cat_names: list[str],
    *, pipe_factory: Callable[..., Pipeline], hparam,
    scorable_mask: np.ndarray,
) -> tuple[dict[str, Pipeline], list[str]]:
    """Fit one pipeline per scorable category on the full dataset (no CV).

    ``scorable_mask`` is a boolean array of length ``len(cat_names)`` derived
    from the *binary* labels (``_scorable(y_bin[:, k])``). Passed in
    explicitly so all three saved bundles cover the same set of categories
    regardless of which target each model was trained on.
    """
    models: dict[str, Pipeline] = {}
    skipped: list[str] = []
    for k_idx, cat in enumerate(cat_names):
        if not scorable_mask[k_idx]:
            skipped.append(cat)
            continue
        pipe = pipe_factory(hparam)
        pipe.fit(X, y[:, k_idx])
        models[cat] = pipe
    return models, skipped


def _build_bundle(
    *, model_name: str, hparam, X: np.ndarray, y: np.ndarray,
    cat_names: list[str], pipe_factory: Callable[..., Pipeline],
    feature_names: list[str], min_net_likes: int,
    scorable_mask: np.ndarray,
) -> dict:
    """Refit on all data and package everything the loader needs."""
    models, skipped = _fit_per_category_final(
        X, y, cat_names, pipe_factory=pipe_factory, hparam=hparam,
        scorable_mask=scorable_mask,
    )
    return {
        "model_name": model_name,
        "best_hparam": hparam,
        "feature_names": list(feature_names),  # column order required at predict time
        "category_names": list(cat_names),
        "scorable_categories": list(models.keys()),
        "skipped_categories": skipped,
        "models": models,                      # {cat: fitted Pipeline (StandardScaler + estimator)}
        "score_kind": SCORE_KIND[model_name],  # "predict" or "predict_proba"
        "score_units": SCORE_UNITS[model_name],
        "training_metadata": {
            "n_users": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "min_net_likes": min_net_likes,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "library_versions": {
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }


def _save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run(
    *,
    features_path: Path,
    multihot_path: Path,
    canonical_path: Optional[Path],
    min_net_likes: int,
    Cs: list[float],
    alphas: list[float],
    knn_ks: list[int],
    min_feature_coverage: int,
    cv_folds: int,
    out_path: Optional[Path],
    preds_out_path: Optional[Path],
    preds_users: int,
    save_models_dir: Optional[Path],
    seed: int,
) -> None:
    print("=" * 72)
    print("Stage A: build labels (signed net-like, canonical-filtered)")
    print("=" * 72)
    rating_csv_paths = discover_users(CORPUS_ROOTS)
    print(f"  users: {len(rating_csv_paths)}")
    labels, nets, dropped = build_above_mean_labels(
        rating_csv_paths, multihot_path,
        min_net_likes=min_net_likes,
        canonical_path=canonical_path,
    )
    if dropped:
        print(f"  dropped {len(dropped)} category cols (non-canonical or "
              f"zero exposure): {dropped}")
    print(f"  kept {labels.shape[1]} cats; "
          f"mean positive rate = {labels.values.mean():.3f}")

    print()
    print("=" * 72)
    print("Stage B: align features + filter")
    print("=" * 72)
    feats = pd.read_csv(features_path, index_col="user_id")
    common = sorted(set(labels.index) & set(feats.index))
    print(f"  features: {feats.shape}, common users: {len(common)}")
    X_df, dropped_feats = _filter_features(feats.loc[common], min_feature_coverage)
    if dropped_feats:
        print(f"  dropped {len(dropped_feats)} feature(s) with <"
              f"{min_feature_coverage} non-zero users (kept {X_df.shape[1]})")
    y_bin_df = labels.loc[common]
    y_net_df = nets.loc[common]
    cat_names = list(y_bin_df.columns)
    n_scor = sum(_scorable(y_bin_df.values[:, k]) for k in range(len(cat_names)))
    print(f"  {n_scor}/{len(cat_names)} categories scorable")

    X = X_df.values.astype(np.float64)
    y_bin = y_bin_df.values.astype(np.int64)
    y_net = y_net_df.values.astype(np.float64)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    print()
    print("=" * 72)
    print("Stage C: hyperparameter sweeps (each model on its native target)")
    print("=" * 72)

    best_C, lr_per_cat, lr_preds = _sweep(
        "C", Cs, X, y_bin, y_net, cat_names, cv,
        pipe_factory=_build_logreg, predict_fn=_predict_logreg,
        target_for_predict="bin",
    )
    print()
    best_alpha, ridge_per_cat, ridge_preds = _sweep(
        "alpha", alphas, X, y_bin, y_net, cat_names, cv,
        pipe_factory=_build_ridge, predict_fn=_predict_regressor,
        target_for_predict="net",
    )
    print()
    best_k, knn_per_cat, knn_preds = _sweep(
        "k", knn_ks, X, y_bin, y_net, cat_names, cv,
        pipe_factory=_build_knn, predict_fn=_predict_regressor,
        target_for_predict="net",
    )

    print()
    print("=" * 72)
    print("Stage D: side-by-side comparison")
    print("=" * 72)
    combined = pd.concat({
        "LR":    lr_per_cat,
        "Ridge": ridge_per_cat,
        "kNN":   knn_per_cat,
    }, axis=1)
    print(f"  best LR    C={best_C}")
    print(f"  best Ridge alpha={best_alpha}")
    print(f"  best kNN   k={best_k}")

    valid = combined.dropna(subset=[("LR", "auc")])  # all 3 share scorable cats

    print()
    print("Macro metrics across {} scorable categories:".format(len(valid)))
    print(f"{'metric':<10} {'LR':>10} {'Ridge':>10} {'kNN':>10}")
    print("-" * 42)
    for metric in ("auc", "ap", "spearman"):
        row = " ".join(
            f"{valid[(m, metric)].mean():>10.4f}"
            for m in ("LR", "Ridge", "kNN")
        )
        print(f"{metric:<10} {row}")
    print()
    print("Median metrics:")
    print(f"{'metric':<10} {'LR':>10} {'Ridge':>10} {'kNN':>10}")
    print("-" * 42)
    for metric in ("auc", "ap", "spearman"):
        row = " ".join(
            f"{valid[(m, metric)].median():>10.4f}"
            for m in ("LR", "Ridge", "kNN")
        )
        print(f"{metric:<10} {row}")
    print()
    print("% categories with AUC > 0.60:")
    for m in ("LR", "Ridge", "kNN"):
        pct = (valid[(m, "auc")] > 0.60).mean() * 100
        print(f"  {m:<6} {pct:>5.1f}%")

    print()
    print("Per-category winner (by AUC):")
    auc_only = valid.xs("auc", axis=1, level=1)
    winner = auc_only.idxmax(axis=1)
    win_counts = winner.value_counts()
    for m in ("LR", "Ridge", "kNN"):
        print(f"  {m:<6} won {win_counts.get(m, 0):>2} of {len(valid)} cats")

    print()
    print("Per-category AUC (sorted by best-model AUC):")
    side = pd.DataFrame({
        "LR":    valid[("LR", "auc")],
        "Ridge": valid[("Ridge", "auc")],
        "kNN":   valid[("kNN", "auc")],
    })
    side["winner"] = winner
    side["best_auc"] = side[["LR", "Ridge", "kNN"]].max(axis=1)
    side = side.sort_values("best_auc", ascending=False).drop(columns=["best_auc"])
    print(side.round(4).to_string())

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        flat = combined.copy()
        flat.columns = [f"{m}_{metric}" for m, metric in flat.columns]
        flat["winner_by_auc"] = winner
        flat.to_csv(out_path)
        print(f"\n  wrote {out_path}")

    if preds_out_path:
        n = len(common) if preds_users <= 0 else min(preds_users, len(common))
        sample_users = common[:n]
        preds_df = pd.concat({
            "LR":    pd.DataFrame(lr_preds,    index=common, columns=cat_names),
            "Ridge": pd.DataFrame(ridge_preds, index=common, columns=cat_names),
            "kNN":   pd.DataFrame(knn_preds,   index=common, columns=cat_names),
        }, axis=1).loc[sample_users]
        preds_df.index.name = "user_id"
        preds_out_path.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(preds_out_path)
        print(f"  wrote {preds_out_path}  "
              f"({preds_df.shape[0]} users x {len(cat_names)} cats x 3 models)")
        print(f"    note: these are out-of-fold CV predictions at the best "
              f"hyperparameter per model")
        print(f"    LR   = predicted P(y=1) in [0, 1]")
        print(f"    Ridge/kNN = predicted continuous net-like score "
              f"(real-valued, can be negative)")

    if save_models_dir is not None:
        print()
        print("=" * 72)
        print("Stage E: refit on full data and save model bundles")
        print("=" * 72)
        feature_names = list(X_df.columns)
        scorable_mask = np.array(
            [_scorable(y_bin[:, k]) for k in range(len(cat_names))]
        )
        bundle_specs = [
            ("LR",    {"C": best_C},          y_bin, _build_logreg),
            ("Ridge", {"alpha": best_alpha},  y_net, _build_ridge),
            ("kNN",   {"k": best_k},          y_net, _build_knn),
        ]
        for name, hp, y_for_fit, factory in bundle_specs:
            # _build_* take a scalar (C / alpha / k), not a dict; unwrap.
            (raw_h,) = hp.values()
            bundle = _build_bundle(
                model_name=name,
                hparam=raw_h,
                X=X, y=y_for_fit, cat_names=cat_names,
                pipe_factory=factory,
                feature_names=feature_names,
                min_net_likes=min_net_likes,
                scorable_mask=scorable_mask,
            )
            # Restore the named hparam for the saved metadata.
            bundle["best_hparam"] = hp
            out = save_models_dir / MODEL_FILES[name]
            _save_bundle(bundle, out)
            print(f"  {name:<6} -> {out}  "
                  f"({len(bundle['scorable_categories'])} cats, "
                  f"{X.shape[1]} features, hparam={hp})")
        print(f"  load with: from src.model.loader import load_bundle; "
              f"b = load_bundle('Ridge')")

    print()
    print("=" * 72)
    print("Recommendation")
    print("=" * 72)
    macros = {m: valid[(m, "auc")].mean() for m in ("LR", "Ridge", "kNN")}
    rank = sorted(macros.items(), key=lambda x: -x[1])
    best_model = rank[0][0]
    spread = rank[0][1] - rank[-1][1]
    print(f"  Macro AUC ranking: " +
          ", ".join(f"{m}={s:.4f}" for m, s in rank))
    print(f"  Best by macro AUC: {best_model}  "
          f"(spread top-vs-bottom = {spread:.4f})")
    if spread < 0.01:
        print("  Spread is tiny -- the three models are statistically "
              "indistinguishable on this dataset.")
        print("  Pick LR for interpretability (linear coefficients per cat),")
        print("  Ridge for a smooth continuous score, or kNN for a "
              "non-parametric similarity-based recommender.")
    elif best_model == "LR":
        print("  LR wins -> use it for production: linear coefficients are "
              "interpretable, predict_proba gives calibrated probabilities, "
              "and it's fast at inference.")
    elif best_model == "Ridge":
        print("  Ridge wins -> demographics map roughly linearly to net "
              "interest. Continuous predicted score is useful for ranking.")
    else:
        print("  kNN wins -> demographic similarity matters more than "
              "linear coefficients. Good for cold-start ranking.")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--multihot", type=Path, default=DEFAULT_MULTIHOT)
    p.add_argument("--canonical", type=Path, default=DEFAULT_T1_PATH)
    p.add_argument("--min-net-likes", type=int, default=1)
    p.add_argument("--Cs", nargs="+", type=float,
                   default=[0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    p.add_argument("--knn-ks", nargs="+", type=int,
                   default=[3, 5, 10, 15, 25, 40])
    p.add_argument("--min-feature-coverage", type=int, default=5)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--preds-out", type=Path, default=DEFAULT_PREDS_OUT,
                   help="Where to write the sample CV predictions CSV.")
    p.add_argument("--preds-users", type=int, default=20,
                   help="Number of users to include in the predictions CSV "
                        "(0 = all). Default 20.")
    p.add_argument("--save-models", action="store_true",
                   help="After the sweep, refit each model on ALL data at "
                        "its chosen hyperparameter and write a joblib bundle "
                        "(LR / Ridge / kNN) to --models-dir for downstream "
                        "demo / serving.")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                   help="Directory to write the .joblib model bundles into. "
                        "Only used when --save-models is set.")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run(
        features_path=args.features,
        multihot_path=args.multihot,
        canonical_path=args.canonical if str(args.canonical) else None,
        min_net_likes=args.min_net_likes,
        Cs=list(args.Cs),
        alphas=list(args.alphas),
        knn_ks=list(args.knn_ks),
        min_feature_coverage=args.min_feature_coverage,
        cv_folds=args.cv_folds,
        out_path=None if args.no_save else args.out,
        preds_out_path=None if args.no_save else args.preds_out,
        preds_users=args.preds_users,
        save_models_dir=(
            None if (args.no_save or not args.save_models) else args.models_dir
        ),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
