"""6-base-model majority-voting ensemble on the per-(user, ad) grid.

Six base models are trained out-of-fold on the SAME 5-fold user-level split,
so their predictions can be combined honestly per pair:

  IAB side (trained on the 136-col user_features matrix; per-IAB-cat then
            projected to (user, ad) by averaging over the cats each ad is
            tagged with):
    * iab_lr     -- LogisticRegression (classifier)   target = binary y_iab
    * iab_ridge  -- Ridge (regressor)                 target = signed net_likes
    * iab_knn    -- KNeighborsRegressor               target = signed net_likes

  Sentient side (trained on the 225-col pair feature matrix directly):
    * sent_lr    -- LogisticRegression (classifier)   target = binary y_pair
    * sent_ridge -- Ridge (regressor)                 target = signed (rating - mu)
    * sent_knn   -- KNeighborsRegressor               target = signed (rating - mu)

The structural parallel: both sides use Ridge/kNN as regressors on a SIGNED
continuous signal (per-cat ``net_likes`` on the IAB side, per-pair
``rating - user_mean`` on the sentient side), and LR as a classifier on the
binarized version of that signal. The regressors vote ``like`` when their
predicted score is > 0; the classifiers vote ``like`` when their predicted
probability is >= 0.5.

Voting strategies evaluated on the 36k pair grid against the same pair label
``y_pair = (rating > user_mean)``:

    * vote_count           -- 0..6 vote tally, used as a continuous score
                              for AUC / AP (the natural ranking signal)
    * majority_6 (>= 4)    -- hard majority across all 6 base models
    * majority_iab (>= 2)  -- IAB side alone, 2 of 3 vote yes
    * majority_sent (>= 2) -- sentient side alone, 2 of 3 vote yes
    * unanimous_6          -- all 6 vote yes (precision-heavy)
    * any_6                -- any 1 of 6 votes yes (recall-heavy)

Run::

    python -m src.model.voting
    python -m src.model.voting --save-predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    CORPUS_ROOTS, IMAGE_ID_FOR, IMAGES_PER_CATEGORY, NUM_CATEGORIES,
    REPO_ROOT, ADS16DataProcessor, discover_users,
)
from src.data_loader.agent_processing.categories_t1 import (
    DEFAULT_CATEGORIES_PATH as DEFAULT_T1_PATH,
)
from src.model.ensemble import _make_user_folds, _project_iab_to_pair
from src.model.interest_matrix import load_multihot
from src.model.sentient_dataset import build_pair_dataset
from src.model.train_logistic import (
    DEFAULT_FEATURES, DEFAULT_MULTIHOT, _scorable, build_above_mean_labels,
)
from src.model.train_sentient import PROFILES


REPO_DATA = REPO_ROOT / "Data"
DEFAULT_OUT = REPO_DATA / "voting_metrics.csv"
DEFAULT_PRED_OUT = REPO_DATA / "voting_pair_predictions.csv"

# Hardcoded hparams. Picked as reasonable middles -- not from a fresh sweep.
# Override via CLI if you want to tune.
DEFAULT_HPARAMS = {
    "iab_lr_C":      1.0,
    "iab_ridge_a":   100.0,
    "iab_knn_k":     10,
    "sent_lr_C":     1.0,
    "sent_ridge_a":  100.0,
    "sent_knn_k":    25,        # bigger k on the 28k-row sentient train set
}

DEFAULT_MIN_NET_LIKES = 1


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def _lr_pipe(C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(
            C=C, solver="liblinear", class_weight="balanced", max_iter=2000,
        )),
    ])


def _ridge_pipe(alpha: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("reg", Ridge(alpha=alpha, solver="svd")),
    ])


def _knn_pipe(k: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("reg", KNeighborsRegressor(
            n_neighbors=k, metric="cosine", weights="distance",
        )),
    ])


# --------------------------------------------------------------------------- #
# IAB side: per-fold per-cat training -> OOF (user, cat) -> project to pair
# --------------------------------------------------------------------------- #
def _iab_oof_per_cat(
    X_user: np.ndarray, y: np.ndarray, *,
    user_folds: np.ndarray, scorable_mask: np.ndarray,
    factory: Callable[..., Pipeline], hparam, predict_method: str,
) -> np.ndarray:
    """Generic per-fold, per-cat training. ``predict_method`` is either
    ``"predict_proba"`` (LR) or ``"predict"`` (Ridge / kNN regressor)."""
    n_users, n_cats = y.shape
    out = np.full((n_users, n_cats), np.nan, dtype=np.float64)
    for f in np.unique(user_folds):
        train = user_folds != f
        test = user_folds == f
        for k in range(n_cats):
            if not scorable_mask[k]:
                continue
            y_train = y[train, k]
            if predict_method == "predict_proba":
                # LR can't fit a single-class fold for cat k.
                if y_train.sum() == 0 or y_train.sum() == len(y_train):
                    continue
            pipe = factory(hparam)
            pipe.fit(X_user[train], y_train)
            if predict_method == "predict_proba":
                out[test, k] = pipe.predict_proba(X_user[test])[:, 1]
            else:
                out[test, k] = pipe.predict(X_user[test])
    return out


# --------------------------------------------------------------------------- #
# Sentient side: per-fold pair training -> OOF (n_pairs,)
# --------------------------------------------------------------------------- #
def _sent_oof_pair(
    pair_X: np.ndarray, target_y: np.ndarray, pair_user_idx: np.ndarray, *,
    user_folds: np.ndarray, factory: Callable[..., Pipeline], hparam,
    predict_method: str,
) -> np.ndarray:
    """Per-fold sentient training. Returns OOF score per pair."""
    n_pairs = len(target_y)
    out = np.full(n_pairs, np.nan, dtype=np.float64)
    for f in np.unique(user_folds):
        test_user = np.where(user_folds == f)[0]
        train_user = np.where(user_folds != f)[0]
        train_mask = np.isin(pair_user_idx, train_user)
        test_mask = np.isin(pair_user_idx, test_user)
        pipe = factory(hparam)
        pipe.fit(pair_X[train_mask], target_y[train_mask])
        if predict_method == "predict_proba":
            out[test_mask] = pipe.predict_proba(pair_X[test_mask])[:, 1]
        else:
            out[test_mask] = pipe.predict(pair_X[test_mask])
    return out


# --------------------------------------------------------------------------- #
# Vote thresholding
# --------------------------------------------------------------------------- #
def _to_vote(score: np.ndarray, *, kind: str) -> np.ndarray:
    """Threshold a per-pair score into a 0/1 ``like`` vote.

    ``kind`` is ``"prob"`` (LR -> >= 0.5) or ``"signed"`` (regressor -> > 0).
    """
    if kind == "prob":
        return (score >= 0.5).astype(np.int8)
    if kind == "signed":
        return (score > 0).astype(np.int8)
    raise ValueError(kind)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Pair-level binary classification metrics."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_pos_rate": float(y_pred.mean()),
    }


def _ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, score)),
        "ap":  float(average_precision_score(y_true, score)),
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(
    *, cv_folds: int, seed: int, min_net_likes: int,
    sentient_profile: str, hparams: dict,
    multihot_path: Path, features_path: Path,
    out_path: Path, save_predictions: bool, pred_out_path: Path,
) -> None:
    print("=" * 72)
    print("Stage 1: build pair dataset (sentient features + signed pair signal)")
    print("=" * 72)
    pair_ds = build_pair_dataset(**PROFILES[sentient_profile])
    user_ids = pair_ds.user_ids
    n_users = len(user_ids)
    n_ads = len(pair_ds.image_ids)
    n_pairs = len(pair_ds.y)

    # Continuous signed analogue of y_pair: rating - user's own mean.
    user_means_per_pair = pair_ds.user_means[pair_ds.groups]
    y_signed = pair_ds.ratings.astype(np.float64) - user_means_per_pair
    y_pair = pair_ds.y.astype(np.int64)

    user_folds = _make_user_folds(n_users, cv_folds, seed)
    print(f"  users:    {n_users}")
    print(f"  ads:      {n_ads}")
    print(f"  pairs:    {n_pairs}")
    print(f"  features: {pair_ds.X.shape[1]} ({sentient_profile} profile)")
    print(f"  pos rate (y_pair): {y_pair.mean():.1%}")
    print(f"  signed signal range: "
          f"[{y_signed.min():.2f}, {y_signed.max():.2f}], "
          f"mean: {y_signed.mean():+.3f}")

    # ----- Stage 2: IAB labels + multihot ----------------------------- #
    print()
    print("=" * 72)
    print("Stage 2: build IAB labels (binary + signed net_likes)")
    print("=" * 72)
    rt_paths = discover_users(CORPUS_ROOTS)
    rt_paths_ordered = {u: rt_paths[u] for u in user_ids if u in rt_paths}
    labels_iab, net_iab, dropped = build_above_mean_labels(
        rt_paths_ordered, multihot_path,
        min_net_likes=min_net_likes, canonical_path=DEFAULT_T1_PATH,
    )
    cat_cols = list(labels_iab.columns)
    y_per_cat_bin = labels_iab.loc[user_ids].values.astype(np.int64)
    y_per_cat_net = net_iab.loc[user_ids].values.astype(np.float64)
    scorable_mask = np.array(
        [_scorable(y_per_cat_bin[:, k]) for k in range(len(cat_cols))]
    )
    print(f"  cats kept:  {len(cat_cols)}  (scorable: {int(scorable_mask.sum())})")

    M_df, _ = load_multihot(multihot_path)
    image_ids = [
        IMAGE_ID_FOR(c, i)
        for c in range(NUM_CATEGORIES)
        for i in range(IMAGES_PER_CATEGORY)
    ]
    multihot_300xK = M_df.loc[image_ids, cat_cols].values.astype(np.float64)
    multihot_300xK *= scorable_mask[None, :]   # zero out non-scorable cats

    feats = pd.read_csv(features_path, index_col="user_id").loc[user_ids]
    X_user = feats.values.astype(np.float64)
    print(f"  user features matrix: {X_user.shape}")

    # ----- Stage 3: train all 6 base models OOF ----------------------- #
    print()
    print("=" * 72)
    print("Stage 3: train 6 base models OOF (same user fold split)")
    print("=" * 72)

    # --- IAB-LR
    print("  iab_lr     (LogReg on user features, target=binary y_iab)")
    iab_lr_per_cat = _iab_oof_per_cat(
        X_user, y_per_cat_bin,
        user_folds=user_folds, scorable_mask=scorable_mask,
        factory=_lr_pipe, hparam=hparams["iab_lr_C"],
        predict_method="predict_proba",
    )
    iab_lr_pair = _project_iab_to_pair(iab_lr_per_cat, multihot_300xK)
    iab_lr_score = iab_lr_pair.ravel()
    iab_lr_vote = _to_vote(iab_lr_score, kind="prob")

    # --- IAB-Ridge
    print("  iab_ridge  (Ridge on user features, target=signed net_likes)")
    iab_ridge_per_cat = _iab_oof_per_cat(
        X_user, y_per_cat_net,
        user_folds=user_folds, scorable_mask=scorable_mask,
        factory=_ridge_pipe, hparam=hparams["iab_ridge_a"],
        predict_method="predict",
    )
    iab_ridge_pair = _project_iab_to_pair(iab_ridge_per_cat, multihot_300xK)
    iab_ridge_score = iab_ridge_pair.ravel()
    iab_ridge_vote = _to_vote(iab_ridge_score, kind="signed")

    # --- IAB-kNN
    print("  iab_knn    (kNN on user features, target=signed net_likes)")
    iab_knn_per_cat = _iab_oof_per_cat(
        X_user, y_per_cat_net,
        user_folds=user_folds, scorable_mask=scorable_mask,
        factory=_knn_pipe, hparam=hparams["iab_knn_k"],
        predict_method="predict",
    )
    iab_knn_pair = _project_iab_to_pair(iab_knn_per_cat, multihot_300xK)
    iab_knn_score = iab_knn_pair.ravel()
    iab_knn_vote = _to_vote(iab_knn_score, kind="signed")

    # --- Sentient-LR
    print("  sent_lr    (LogReg on pair features, target=binary y_pair)")
    sent_lr_score = _sent_oof_pair(
        pair_ds.X, y_pair, pair_ds.groups,
        user_folds=user_folds, factory=_lr_pipe,
        hparam=hparams["sent_lr_C"], predict_method="predict_proba",
    )
    sent_lr_vote = _to_vote(sent_lr_score, kind="prob")

    # --- Sentient-Ridge
    print("  sent_ridge (Ridge on pair features, target=signed rating-user_mean)")
    sent_ridge_score = _sent_oof_pair(
        pair_ds.X, y_signed, pair_ds.groups,
        user_folds=user_folds, factory=_ridge_pipe,
        hparam=hparams["sent_ridge_a"], predict_method="predict",
    )
    sent_ridge_vote = _to_vote(sent_ridge_score, kind="signed")

    # --- Sentient-kNN
    print("  sent_knn   (kNN on pair features, target=signed rating-user_mean)")
    print("    (this is the slow one ~ a couple minutes)")
    sent_knn_score = _sent_oof_pair(
        pair_ds.X, y_signed, pair_ds.groups,
        user_folds=user_folds, factory=_knn_pipe,
        hparam=hparams["sent_knn_k"], predict_method="predict",
    )
    sent_knn_vote = _to_vote(sent_knn_score, kind="signed")

    # ----- Stage 4: per-base-model metrics ---------------------------- #
    base_models = {
        "iab_lr":     (iab_lr_score,     iab_lr_vote),
        "iab_ridge":  (iab_ridge_score,  iab_ridge_vote),
        "iab_knn":    (iab_knn_score,    iab_knn_vote),
        "sent_lr":    (sent_lr_score,    sent_lr_vote),
        "sent_ridge": (sent_ridge_score, sent_ridge_vote),
        "sent_knn":   (sent_knn_score,   sent_knn_vote),
    }

    base_rows = []
    for name, (score, vote) in base_models.items():
        # Scores are continuous (real or [0, 1]); both work for AUC ranking.
        base_rows.append({
            "model": name, "kind": "base",
            **_ranking_metrics(y_pair, score),
            **_binary_metrics(y_pair, vote),
        })
    base_df = pd.DataFrame(base_rows)

    # ----- Stage 5: voting strategies --------------------------------- #
    iab_vote_sum = iab_lr_vote + iab_ridge_vote + iab_knn_vote     # 0..3
    sent_vote_sum = sent_lr_vote + sent_ridge_vote + sent_knn_vote  # 0..3
    total_vote = iab_vote_sum + sent_vote_sum                      # 0..6

    voting_rows = []

    voting_rows.append({
        "model": "vote_count (0..6)", "kind": "ensemble",
        **_ranking_metrics(y_pair, total_vote.astype(np.float64)),
        **_binary_metrics(y_pair, (total_vote >= 4).astype(np.int8)),
    })
    voting_rows.append({
        "model": "majority_6 (>=4)", "kind": "ensemble",
        "auc": float(roc_auc_score(y_pair, total_vote.astype(np.float64))),
        "ap":  float(average_precision_score(y_pair, total_vote.astype(np.float64))),
        **_binary_metrics(y_pair, (total_vote >= 4).astype(np.int8)),
    })
    voting_rows.append({
        "model": "majority_iab (>=2 of 3)", "kind": "ensemble",
        "auc": float(roc_auc_score(y_pair, iab_vote_sum.astype(np.float64))),
        "ap":  float(average_precision_score(y_pair, iab_vote_sum.astype(np.float64))),
        **_binary_metrics(y_pair, (iab_vote_sum >= 2).astype(np.int8)),
    })
    voting_rows.append({
        "model": "majority_sent (>=2 of 3)", "kind": "ensemble",
        "auc": float(roc_auc_score(y_pair, sent_vote_sum.astype(np.float64))),
        "ap":  float(average_precision_score(y_pair, sent_vote_sum.astype(np.float64))),
        **_binary_metrics(y_pair, (sent_vote_sum >= 2).astype(np.int8)),
    })
    voting_rows.append({
        "model": "unanimous_6 (==6)", "kind": "ensemble",
        "auc": float(roc_auc_score(y_pair, total_vote.astype(np.float64))),
        "ap":  float(average_precision_score(y_pair, total_vote.astype(np.float64))),
        **_binary_metrics(y_pair, (total_vote == 6).astype(np.int8)),
    })
    voting_rows.append({
        "model": "any_6 (>=1)", "kind": "ensemble",
        "auc": float(roc_auc_score(y_pair, total_vote.astype(np.float64))),
        "ap":  float(average_precision_score(y_pair, total_vote.astype(np.float64))),
        **_binary_metrics(y_pair, (total_vote >= 1).astype(np.int8)),
    })

    voting_df = pd.DataFrame(voting_rows)
    full_df = pd.concat([base_df, voting_df], ignore_index=True)

    cols_order = [
        "model", "kind", "auc", "ap",
        "accuracy", "precision", "recall", "f1", "predicted_pos_rate",
    ]
    full_df = full_df[cols_order]

    # Vote-count distribution (info-only)
    print()
    print("Vote-count distribution across the 36k pairs:")
    print(pd.Series(total_vote).value_counts().sort_index().to_string())

    print()
    print("Per-base-model + voting metrics:")
    print(full_df.round(4).to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\nMetrics -> {out_path}")

    if save_predictions:
        rows = pd.DataFrame({
            "user_id":    np.repeat(np.array(user_ids), n_ads),
            "image_id":   np.tile(np.array(pair_ds.image_ids), n_users),
            "rating":     pair_ds.ratings,
            "y_pair":     y_pair,
            "iab_lr":     iab_lr_vote,
            "iab_ridge":  iab_ridge_vote,
            "iab_knn":    iab_knn_vote,
            "sent_lr":    sent_lr_vote,
            "sent_ridge": sent_ridge_vote,
            "sent_knn":   sent_knn_vote,
            "vote_count": total_vote,
        })
        pred_out_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(pred_out_path, index=False)
        print(f"Pair votes -> {pred_out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-net-likes", type=int, default=DEFAULT_MIN_NET_LIKES)
    p.add_argument("--sentient-profile", default="with_demographics",
                   choices=list(PROFILES))
    p.add_argument("--iab-lr-C", type=float, default=DEFAULT_HPARAMS["iab_lr_C"])
    p.add_argument("--iab-ridge-a", type=float, default=DEFAULT_HPARAMS["iab_ridge_a"])
    p.add_argument("--iab-knn-k", type=int, default=DEFAULT_HPARAMS["iab_knn_k"])
    p.add_argument("--sent-lr-C", type=float, default=DEFAULT_HPARAMS["sent_lr_C"])
    p.add_argument("--sent-ridge-a", type=float, default=DEFAULT_HPARAMS["sent_ridge_a"])
    p.add_argument("--sent-knn-k", type=int, default=DEFAULT_HPARAMS["sent_knn_k"])
    p.add_argument("--multihot", type=Path, default=DEFAULT_MULTIHOT)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--pred-out", type=Path, default=DEFAULT_PRED_OUT)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    hparams = {
        "iab_lr_C":     args.iab_lr_C,
        "iab_ridge_a":  args.iab_ridge_a,
        "iab_knn_k":    args.iab_knn_k,
        "sent_lr_C":    args.sent_lr_C,
        "sent_ridge_a": args.sent_ridge_a,
        "sent_knn_k":   args.sent_knn_k,
    }
    run(
        cv_folds=args.cv_folds, seed=args.seed,
        min_net_likes=args.min_net_likes,
        sentient_profile=args.sentient_profile, hparams=hparams,
        multihot_path=args.multihot, features_path=args.features,
        out_path=args.out, save_predictions=args.save_predictions,
        pred_out_path=args.pred_out,
    )


if __name__ == "__main__":
    main()
