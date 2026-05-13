"""Ensemble the content IAB model and the sentient pair model on the per-pair grid.

Both models are honestly cross-validated against the SAME 5-fold user-level
split, so their out-of-fold predictions live on the same 36 000 (user, ad)
rows and can be combined apples-to-apples.

Pipeline
--------
1. Build a deterministic 5-fold split over the 120 user ids.
2. **Content IAB model** -- per-fold:
     * Compute net-like labels on TRAIN users only:
         signal = +1 / -1 / 0  (rating > / < / == user_mean)
         net[u, k] = signal @ multihot[:, k]
         y_iab[u, k] = (net[u, k] >= min_net_likes)
     * Fit per-category LogisticRegression on the 136-col user_features matrix.
     * Predict probabilities on TEST users.
   Stack across folds -> ``p_iab_per_cat`` shape (120, n_scorable_cats).
3. **Project IAB to pair grid**: for each (user, ad), average per-cat
   probabilities over the IAB cats the ad is tagged with
        p_iab[u, ad] = mean_{k : multihot[ad, k] = 1} p_iab_per_cat[u, k]
   Ads tagged only with non-scorable cats fall back to the global mean.
4. **Sentient pair model** -- per-fold: fit on TRAIN-user pairs (~28 800 rows),
   predict TEST-user pairs (~7 200 rows). Default model is the
   HistGradientBoostingClassifier on the ``with_demographics`` profile (the
   best single model from ``train_sentient``).
5. Stack the two OOF score arrays and compute several **ensemble strategies**
   on the same 36k pair grid against the pair label
   ``y_pair[u, ad] = (rating[u, ad] > user_mean[u])`` -- the pair-level half
   of the IAB model's signed signal:
       * ``iab_only``           -- IAB projected to pair, no ensemble
       * ``sentient_only``      -- sentient model alone
       * ``mean``               -- 0.5 * p_iab + 0.5 * p_sentient
       * ``weighted (best alpha)`` -- alpha sweep, picks alpha that maxes pair AUC
       * ``stack_lr``           -- LR meta-learner on (p_iab, p_sentient),
                                   itself OOF-trained per fold
       * ``max_confidence``     -- pick whichever per-pair score is further
                                   from 0.5 (the model that's "more decisive")
       * ``oracle``             -- per-pair, pick the model closer to truth
                                   (upper bound, NOT deployable -- shows headroom)

Run::

    python -m src.model.ensemble
    python -m src.model.ensemble --cv-folds 5 --iab-C 1.0
    python -m src.model.ensemble --sentient-profile compact \
        --sentient-model lr --save-predictions
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score,
)

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    CORPUS_ROOTS, IMAGE_ID_FOR, IMAGES_PER_CATEGORY, NUM_CATEGORIES,
    REPO_ROOT, discover_users,
)
from src.data_loader.agent_processing.categories_t1 import (
    DEFAULT_CATEGORIES_PATH as DEFAULT_T1_PATH,
)
from src.model.interest_matrix import load_multihot
from src.model.sentient_dataset import build_pair_dataset
from src.model.train_logistic import (
    DEFAULT_FEATURES, DEFAULT_MULTIHOT,
    _build_logistic, _filter_features, _scorable,
    build_above_mean_labels, build_frac_positive_target,
)
from src.model.train_sentient import (
    PROFILES, _build_gbm, _build_lr,
)


REPO_DATA = REPO_ROOT / "Data"
DEFAULT_OUT = REPO_DATA / "ensemble_metrics.csv"
DEFAULT_PRED_OUT = REPO_DATA / "ensemble_pair_predictions.csv"


# Best sentient hyperparams from the previous train_sentient run (saves us
# repeating that sweep here). Override via CLI if you want to try others.
DEFAULT_SENTIENT_GBM_PARAMS = {
    "learning_rate": 0.1, "max_depth": 6, "max_iter": 200,
}
DEFAULT_SENTIENT_LR_C = 1.0
DEFAULT_IAB_C = 1.0
DEFAULT_MIN_NET_LIKES = 1

# Default IAB config -- updated to the winner of iab_best.py:
# kNN regressor on the frac_positive target, k=25. Pair AUC 0.5929
# (vs 0.5497 for the previous LR-on-binary baseline). Override via CLI.
DEFAULT_IAB_MODEL = "knn"        # in {"lr", "ridge", "knn"}
DEFAULT_IAB_TARGET = "frac_positive"  # in {"binary", "net_likes", "frac_positive"}
DEFAULT_IAB_HPARAM = 25.0        # C for LR, alpha for Ridge, k (int) for kNN


# --------------------------------------------------------------------------- #
# Fold assignment
# --------------------------------------------------------------------------- #
def _make_user_folds(n_users: int, n_folds: int, seed: int) -> np.ndarray:
    """Deterministic per-user fold ids in ``[0, n_folds)``."""
    rng = np.random.default_rng(seed)
    fold = np.tile(np.arange(n_folds), n_users // n_folds + 1)[:n_users]
    rng.shuffle(fold)
    return fold


# --------------------------------------------------------------------------- #
# IAB content model: cross-validated OOF predictions
# --------------------------------------------------------------------------- #
def _build_ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("s", StandardScaler()),
        ("m", Ridge(alpha=alpha, solver="svd")),
    ])


def _build_knn_reg(k: int) -> Pipeline:
    return Pipeline([
        ("s", StandardScaler()),
        ("m", KNeighborsRegressor(n_neighbors=int(k), metric="cosine",
                                  weights="distance")),
    ])


def _resolve_iab(
    model: str, target: str, hparam,
):
    """Return (factory, predict_method, target_kind).

    ``target_kind`` selects which target column matrix the runner passes in:
    ``"binary"`` -> binarised y_iab, ``"net_likes"`` -> signed net_likes,
    ``"frac_positive"`` -> [0, 1] fraction positive.
    """
    if model == "lr":
        if target != "binary":
            raise ValueError("--iab-model=lr requires --iab-target=binary")
        return (lambda h: _build_logistic(float(h)),
                "predict_proba", "binary")
    if model == "ridge":
        if target == "binary":
            raise ValueError("--iab-model=ridge requires a regression target "
                             "(net_likes or frac_positive)")
        return (lambda h: _build_ridge(float(h)), "predict", target)
    if model == "knn":
        if target == "binary":
            raise ValueError("--iab-model=knn requires a regression target "
                             "(net_likes or frac_positive)")
        return (lambda h: _build_knn_reg(int(h)), "predict", target)
    raise ValueError(f"unknown iab model {model!r}")


def _iab_oof_per_user_cat(
    X_user: np.ndarray, y_per_cat: np.ndarray, *,
    user_folds: np.ndarray, scorable_mask: np.ndarray,
    factory, hparam, predict_method: str,
) -> np.ndarray:
    """Per-fold per-cat training -> OOF score ``(n_users, n_cats)``.

    ``predict_method`` is ``"predict_proba"`` (LR -> column 1 -> probability)
    or ``"predict"`` (Ridge / kNN regressor). Non-scorable cats and cats
    whose train fold is single-class (LR only) are filled with NaN.
    """
    n_users, n_cats = y_per_cat.shape
    p = np.full((n_users, n_cats), np.nan, dtype=np.float64)
    for f in np.unique(user_folds):
        train_mask = user_folds != f
        test_mask = user_folds == f
        for k in range(n_cats):
            if not scorable_mask[k]:
                continue
            y_train = y_per_cat[train_mask, k]
            if predict_method == "predict_proba":
                if y_train.sum() == 0 or y_train.sum() == len(y_train):
                    continue
            pipe = factory(hparam)
            pipe.fit(X_user[train_mask], y_train)
            if predict_method == "predict_proba":
                p[test_mask, k] = pipe.predict_proba(X_user[test_mask])[:, 1]
            else:
                p[test_mask, k] = pipe.predict(X_user[test_mask])
    return p


def _project_iab_to_pair(
    p_iab_per_cat: np.ndarray, multihot_300xK: np.ndarray,
) -> np.ndarray:
    """Project (user, cat) scores to (user, ad) by averaging over the cats
    each ad is tagged with.

    Uses NaN in ``p_iab_per_cat`` as a "no opinion" mask: cats the model
    couldn't predict (non-scorable / single-class fold) are excluded from the
    average. Ads with zero remaining scorable cats fall back to the global
    mean of the IAB OOF predictions (so the score stays near the population
    base rate rather than NaN).
    """
    valid = (~np.isnan(p_iab_per_cat)).astype(np.float64)
    p = np.where(np.isnan(p_iab_per_cat), 0.0, p_iab_per_cat)

    # (n_users, n_cats) @ (n_cats, 300) -> (n_users, 300)
    num = (p * valid) @ multihot_300xK.T
    den = valid @ multihot_300xK.T
    fallback = float(np.nanmean(p_iab_per_cat))
    return np.where(den > 0, num / np.maximum(den, 1e-9), fallback)


# --------------------------------------------------------------------------- #
# Sentient pair model: cross-validated OOF predictions
# --------------------------------------------------------------------------- #
def _sentient_oof_pair(
    pair_X: np.ndarray, pair_y: np.ndarray, pair_user_idx: np.ndarray, *,
    user_folds: np.ndarray, factory_call,
) -> np.ndarray:
    """Per-fold sentient training, return OOF P(like) per pair (length n_pairs)."""
    n_pairs = len(pair_y)
    p = np.full(n_pairs, np.nan, dtype=np.float64)
    for f in np.unique(user_folds):
        test_user = np.where(user_folds == f)[0]
        train_user = np.where(user_folds != f)[0]
        train_mask = np.isin(pair_user_idx, train_user)
        test_mask = np.isin(pair_user_idx, test_user)
        est = factory_call()
        est.fit(pair_X[train_mask], pair_y[train_mask])
        p[test_mask] = est.predict_proba(pair_X[test_mask])[:, 1]
    return p


# --------------------------------------------------------------------------- #
# Stacking (also OOF, with the same user folds)
# --------------------------------------------------------------------------- #
def _stack_oof(
    p_iab_pair: np.ndarray, p_sent_pair: np.ndarray, y: np.ndarray,
    pair_user_idx: np.ndarray, user_folds: np.ndarray, *, C: float = 1.0,
) -> np.ndarray:
    """OOF logistic-regression stack on the two pair-level scores."""
    n_pairs = len(y)
    out = np.full(n_pairs, np.nan)
    for f in np.unique(user_folds):
        test_user = np.where(user_folds == f)[0]
        train_user = np.where(user_folds != f)[0]
        train_mask = np.isin(pair_user_idx, train_user)
        test_mask = np.isin(pair_user_idx, test_user)
        Xs_train = np.column_stack([p_iab_pair[train_mask], p_sent_pair[train_mask]])
        Xs_test = np.column_stack([p_iab_pair[test_mask], p_sent_pair[test_mask]])
        meta = LogisticRegression(C=C, solver="liblinear")
        meta.fit(Xs_train, y[train_mask])
        out[test_mask] = meta.predict_proba(Xs_test)[:, 1]
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc":      float(roc_auc_score(y, p)),
        "ap":       float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
    }


def _evaluate_strategies(
    y_pair: np.ndarray, p_iab: np.ndarray, p_sent: np.ndarray,
    p_stack: np.ndarray, alphas: np.ndarray,
) -> pd.DataFrame:
    rows = []
    rows.append({"strategy": "iab_only",      **_binary_metrics(y_pair, p_iab)})
    rows.append({"strategy": "sentient_only", **_binary_metrics(y_pair, p_sent)})
    rows.append({"strategy": "mean",          **_binary_metrics(y_pair, 0.5 * p_iab + 0.5 * p_sent)})

    sweep = []
    for a in alphas:
        p_w = a * p_iab + (1 - a) * p_sent
        sweep.append((a, roc_auc_score(y_pair, p_w)))
    sweep.sort(key=lambda x: x[1], reverse=True)
    best_a, best_auc = sweep[0]
    rows.append({
        "strategy": f"weighted_alpha={best_a:.2f}",
        **_binary_metrics(y_pair, best_a * p_iab + (1 - best_a) * p_sent),
    })

    rows.append({"strategy": "stack_lr_oof", **_binary_metrics(y_pair, p_stack)})

    conf_iab = np.abs(p_iab - 0.5)
    conf_sent = np.abs(p_sent - 0.5)
    pick_conf = np.where(conf_iab >= conf_sent, p_iab, p_sent)
    rows.append({"strategy": "max_confidence", **_binary_metrics(y_pair, pick_conf)})

    err_iab = np.abs(y_pair - p_iab)
    err_sent = np.abs(y_pair - p_sent)
    oracle = np.where(err_iab <= err_sent, p_iab, p_sent)
    rows.append({"strategy": "oracle (upper bound)", **_binary_metrics(y_pair, oracle)})

    return pd.DataFrame(rows).set_index("strategy"), sweep


# --------------------------------------------------------------------------- #
# End-to-end runner
# --------------------------------------------------------------------------- #
def run(
    *,
    cv_folds: int, seed: int,
    iab_model: str, iab_target: str, iab_hparam,
    min_net_likes: int,
    sentient_profile: str, sentient_model: str,
    sentient_lr_C: float, sentient_gbm_params: dict,
    multihot_path: Path, features_path: Path,
    out_path: Path, save_predictions: bool, pred_out_path: Path,
    min_feature_coverage: int = 5,
) -> None:
    print("=" * 72)
    print("Stage 1: build pair dataset (sentient features + pair labels)")
    print("=" * 72)
    pair_ds = build_pair_dataset(**PROFILES[sentient_profile])
    user_ids = pair_ds.user_ids
    n_users = len(user_ids)
    n_ads = len(pair_ds.image_ids)
    print(f"  users:    {n_users}")
    print(f"  ads:      {n_ads}")
    print(f"  pairs:    {len(pair_ds.y)}")
    print(f"  features: {pair_ds.X.shape[1]} ({sentient_profile} profile)")
    print(f"  pair pos rate: {pair_ds.y.mean():.1%}")

    user_folds = _make_user_folds(n_users, cv_folds, seed)
    fold_counts = pd.Series(user_folds).value_counts().sort_index().to_list()
    print(f"  user folds: {cv_folds}-way, sizes per fold = {fold_counts}")

    # ----- Stage 2: IAB labels (computed using ALL users -- only the model
    # is per-fold; labels are deterministic from ratings + multihot). ----- #
    print()
    print("=" * 72)
    print("Stage 2: build IAB content labels and select scorable cats")
    print("=" * 72)
    rt_paths = discover_users(CORPUS_ROOTS)
    rt_paths_ordered = {u: rt_paths[u] for u in user_ids if u in rt_paths}
    if len(rt_paths_ordered) != n_users:
        raise RuntimeError(
            f"missing rating CSVs for "
            f"{set(user_ids) - set(rt_paths_ordered)}; cannot proceed"
        )

    labels_iab, net_iab, dropped_cols = build_above_mean_labels(
        rt_paths_ordered, multihot_path,
        min_net_likes=min_net_likes, canonical_path=DEFAULT_T1_PATH,
    )
    if dropped_cols:
        print(f"  dropped {len(dropped_cols)} non-canonical / zero-exposure cats")
    cat_cols = list(labels_iab.columns)
    y_per_cat_bin = labels_iab.loc[user_ids].values.astype(np.int64)
    y_per_cat_net = net_iab.loc[user_ids].values.astype(np.float64)
    scorable_mask = np.array(
        [_scorable(y_per_cat_bin[:, k]) for k in range(len(cat_cols))]
    )
    n_scorable = int(scorable_mask.sum())
    print(f"  cats kept: {len(cat_cols)} canonical / non-zero-exposure")
    print(f"  scorable:  {n_scorable} (>= 5 pos AND >= 5 neg)")
    print(f"  IAB label pos rate (scorable cats): "
          f"{y_per_cat_bin[:, scorable_mask].mean():.1%}")

    # frac_positive target is computed once, used only if --iab-target=frac_positive
    frac_df, _ = build_frac_positive_target(
        rt_paths_ordered, multihot_path,
        positive_threshold=3, canonical_path=DEFAULT_T1_PATH,
    )
    y_per_cat_frac = frac_df.loc[user_ids, cat_cols].values.astype(np.float64)

    # Aligned multihot for projection: (300, n_cats) in canonical image order.
    M_df, _ = load_multihot(multihot_path)
    image_ids = [
        IMAGE_ID_FOR(c, i)
        for c in range(NUM_CATEGORIES)
        for i in range(IMAGES_PER_CATEGORY)
    ]
    multihot_300xK = M_df.loc[image_ids, cat_cols].values.astype(np.float64)
    # Zero out non-scorable cats so projection naturally ignores them.
    multihot_300xK = multihot_300xK * scorable_mask[None, :]

    # ----- Stage 3: IAB OOF per (user, cat) ----------------------------- #
    print()
    print("=" * 72)
    print(f"Stage 3: IAB OOF per (user, cat) "
          f"[{iab_model}, target={iab_target}, hparam={iab_hparam}]")
    print("=" * 72)
    feats_full = pd.read_csv(features_path, index_col="user_id").loc[user_ids]
    feats, dropped_feats = _filter_features(feats_full, min_feature_coverage)
    if dropped_feats:
        print(f"  dropped {len(dropped_feats)} low-coverage user features "
              f"(<{min_feature_coverage} non-zero users)")
    X_user = feats.values.astype(np.float64)
    print(f"  user feature matrix: {X_user.shape}")

    factory, predict_method, target_kind = _resolve_iab(
        iab_model, iab_target, iab_hparam,
    )
    target_lookup = {
        "binary":        y_per_cat_bin,
        "net_likes":     y_per_cat_net,
        "frac_positive": y_per_cat_frac,
    }
    target = target_lookup[target_kind]

    p_iab_per_cat = _iab_oof_per_user_cat(
        X_user, target,
        user_folds=user_folds, scorable_mask=scorable_mask,
        factory=factory, hparam=iab_hparam, predict_method=predict_method,
    )
    coverage = np.mean(~np.isnan(p_iab_per_cat))
    print(f"  OOF coverage: {coverage:.1%} of (user, cat) cells filled")

    # ----- Stage 4: project IAB scores to per-(user, ad) -------------- #
    print()
    print("=" * 72)
    print("Stage 4: project IAB (user, cat) scores -> (user, ad)")
    print("=" * 72)
    p_iab_pair = _project_iab_to_pair(p_iab_per_cat, multihot_300xK)
    if predict_method == "predict":
        # Regression scores aren't on a probability scale -- min-max scale them
        # to [0, 1] so they can be combined with the sentient probability via
        # weighted mean / stacking. (Rank-preserving for AUC; only the absolute
        # scale changes for the weighted ensemble.)
        lo, hi = float(np.nanmin(p_iab_pair)), float(np.nanmax(p_iab_pair))
        if hi - lo > 1e-9:
            p_iab_pair = (p_iab_pair - lo) / (hi - lo)
            print(f"  scaled regression scores from [{lo:+.3f}, {hi:+.3f}] -> [0, 1]")
    print(f"  p_iab pair shape: {p_iab_pair.shape}")
    print(f"  p_iab range: [{p_iab_pair.min():.3f}, {p_iab_pair.max():.3f}], "
          f"mean: {p_iab_pair.mean():.3f}")

    # ----- Stage 5: sentient OOF per pair ----------------------------- #
    print()
    print("=" * 72)
    print(f"Stage 5: sentient OOF per pair [{sentient_model}, "
          f"{sentient_profile}]")
    print("=" * 72)
    if sentient_model == "lr":
        factory_call = lambda: _build_lr(sentient_lr_C)
        hparam_str = f"C={sentient_lr_C}"
    elif sentient_model == "gbm":
        factory_call = lambda: _build_gbm(sentient_gbm_params)
        hparam_str = ", ".join(f"{k}={v}" for k, v in sentient_gbm_params.items())
    else:
        raise ValueError(f"unknown sentient_model {sentient_model!r}")
    print(f"  hparams: {hparam_str}")

    p_sent_pair_flat = _sentient_oof_pair(
        pair_ds.X, pair_ds.y, pair_ds.groups,
        user_folds=user_folds, factory_call=factory_call,
    )
    p_sent_pair = p_sent_pair_flat.reshape(n_users, n_ads)
    print(f"  p_sent pair shape: {p_sent_pair.shape}")
    print(f"  p_sent range: [{p_sent_pair.min():.3f}, {p_sent_pair.max():.3f}], "
          f"mean: {p_sent_pair.mean():.3f}")

    # ----- Stage 6: pair labels (broadcast user means) --------------- #
    y_pair = pair_ds.y.reshape(n_users, n_ads)

    # ----- Stage 7: stacking (also OOF) ------------------------------ #
    print()
    print("=" * 72)
    print("Stage 6: OOF stacking + ensemble strategies")
    print("=" * 72)
    p_stack_flat = _stack_oof(
        p_iab_pair.ravel(), p_sent_pair.ravel(), y_pair.ravel(),
        pair_ds.groups, user_folds, C=1.0,
    )
    p_stack = p_stack_flat.reshape(n_users, n_ads)

    alphas = np.round(np.linspace(0.0, 1.0, 21), 2)
    metrics_df, sweep = _evaluate_strategies(
        y_pair.ravel(), p_iab_pair.ravel(), p_sent_pair.ravel(),
        p_stack.ravel(), alphas,
    )

    metrics_df["delta_vs_sentient_alone"] = metrics_df["auc"] - metrics_df.loc["sentient_only", "auc"]
    metrics_df["delta_vs_iab_alone"] = metrics_df["auc"] - metrics_df.loc["iab_only", "auc"]

    print("\nPair-level metrics (CV-honest):")
    print(metrics_df.round(4).to_string())

    print("\nWeighted ensemble alpha sweep (alpha = weight on IAB):")
    sweep_df = pd.DataFrame(sweep, columns=["alpha", "auc"]).set_index("alpha").sort_index()
    print(sweep_df.round(4).to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path)
    print(f"\nMetrics -> {out_path}")

    # ----- Stage 8: optionally dump pair-level predictions ---------- #
    if save_predictions:
        rows = pd.DataFrame({
            "user_id":  np.repeat(np.array(user_ids), n_ads),
            "image_id": np.tile(np.array(pair_ds.image_ids), n_users),
            "rating":   pair_ds.ratings,
            "y_pair":   y_pair.ravel(),
            "p_iab":    p_iab_pair.ravel(),
            "p_sent":   p_sent_pair.ravel(),
            "p_stack":  p_stack.ravel(),
        })
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(pred_out_path, index=False)
        print(f"Pair predictions -> {pred_out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    # IAB content model
    p.add_argument("--iab-model", default=DEFAULT_IAB_MODEL,
                   choices=["lr", "ridge", "knn"],
                   help="LR (binary), Ridge or kNN (regression).")
    p.add_argument("--iab-target", default=DEFAULT_IAB_TARGET,
                   choices=["binary", "net_likes", "frac_positive"],
                   help="binary required for lr; net_likes/frac_positive for "
                        "ridge/knn. frac_positive is the iab_best winner.")
    p.add_argument("--iab-hparam", type=float, default=DEFAULT_IAB_HPARAM,
                   help="C for lr, alpha for ridge, k for knn (cast to int).")
    p.add_argument("--min-net-likes", type=int, default=DEFAULT_MIN_NET_LIKES)
    p.add_argument("--min-feature-coverage", type=int, default=5,
                   help="Drop user-feature columns with fewer than N "
                        "non-zero users (default 5).")
    # Sentient pair model
    p.add_argument("--sentient-profile", default="with_demographics",
                   choices=list(PROFILES))
    p.add_argument("--sentient-model", default="gbm", choices=["lr", "gbm"])
    p.add_argument("--sentient-lr-C", type=float, default=DEFAULT_SENTIENT_LR_C)
    # Paths / output
    p.add_argument("--multihot", type=Path, default=DEFAULT_MULTIHOT)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--pred-out", type=Path, default=DEFAULT_PRED_OUT)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    # kNN expects integer k.
    iab_hparam = int(args.iab_hparam) if args.iab_model == "knn" else args.iab_hparam
    run(
        cv_folds=args.cv_folds, seed=args.seed,
        iab_model=args.iab_model, iab_target=args.iab_target,
        iab_hparam=iab_hparam,
        min_net_likes=args.min_net_likes,
        min_feature_coverage=args.min_feature_coverage,
        sentient_profile=args.sentient_profile,
        sentient_model=args.sentient_model,
        sentient_lr_C=args.sentient_lr_C,
        sentient_gbm_params=DEFAULT_SENTIENT_GBM_PARAMS,
        multihot_path=args.multihot,
        features_path=args.features,
        out_path=args.out,
        save_predictions=args.save_predictions,
        pred_out_path=args.pred_out,
    )


if __name__ == "__main__":
    main()
