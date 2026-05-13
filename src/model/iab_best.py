"""Find the best IAB content model for use inside the ensemble.

The original IAB sweep in ``compare_models.py`` evaluates per-`(user, IAB-cat)`
metrics. The ensemble in ``ensemble.py`` projects those per-cat scores to the
per-`(user, ad)` grid via the multi-hot. **A higher per-cat AUC does not
guarantee a higher pair AUC after projection** -- categories with bad
predictions but lots of ad coverage can drag the projection down.

So this script picks the IAB winner by the metric the ensemble actually cares
about: pair-level AUC after projection, evaluated against the same
``y_pair = (rating > user_mean)`` label the sentient model uses.

Configurations tried (3 model types x 2 regression targets):

  * LR (binary classifier)       - target = binary y_iab           (one config: best C)
  * Ridge (regressor)            - target = signed net_likes       (alpha sweep)
  * Ridge (regressor)            - target = frac_positive [0,1]    (alpha sweep)
  * kNN  (regressor)             - target = signed net_likes       (k sweep)
  * kNN  (regressor)             - target = frac_positive [0,1]    (k sweep)

Per-config: per-cat OOF predictions on the same 5-fold user split as
``ensemble.py``, projected to (user, ad) by averaging over the cats each ad
is tagged with, then pair AUC against the binary pair label.

Outputs:
  * ``Data/iab_config_search.csv`` with per-config (per_cat_macro_auc,
    pair_auc, pair_ap) so you can pick the winner.
  * Prints the winning config + a one-liner suggesting the matching CLI
    flags for ``ensemble.py``.

Run::

    python -m src.model.iab_best
    python -m src.model.iab_best --alphas 10 25 50 100 --ks 3 5 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
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
from src.model.ensemble import _make_user_folds, _project_iab_to_pair
from src.model.interest_matrix import load_multihot
from src.model.sentient_dataset import build_pair_dataset
from src.model.train_logistic import (
    DEFAULT_FEATURES, DEFAULT_MULTIHOT,
    _filter_features, _scorable,
    build_above_mean_labels, build_frac_positive_target,
)
from src.model.train_sentient import PROFILES
from src.model.voting import _iab_oof_per_cat


REPO_DATA = REPO_ROOT / "Data"
DEFAULT_OUT = REPO_DATA / "iab_config_search.csv"


def _lr(C):
    return Pipeline([
        ("s", StandardScaler()),
        ("m", LogisticRegression(C=C, solver="liblinear",
                                 class_weight="balanced", max_iter=2000)),
    ])


def _ridge(alpha):
    return Pipeline([
        ("s", StandardScaler()),
        ("m", Ridge(alpha=alpha, solver="svd")),
    ])


def _knn(k):
    return Pipeline([
        ("s", StandardScaler()),
        ("m", KNeighborsRegressor(n_neighbors=k, metric="cosine",
                                  weights="distance")),
    ])


def _percat_macro_auc(p_per_cat: np.ndarray, y_bin_per_cat: np.ndarray,
                      scorable_mask: np.ndarray) -> float:
    """Mean AUC over scorable cats where prediction is non-NaN."""
    aucs = []
    for k in range(y_bin_per_cat.shape[1]):
        if not scorable_mask[k]:
            continue
        col = p_per_cat[:, k]
        if np.isnan(col).any():
            continue
        aucs.append(roc_auc_score(y_bin_per_cat[:, k], col))
    return float(np.mean(aucs)) if aucs else float("nan")


def _eval_one(
    name: str, factory: Callable[..., Pipeline], hparam,
    target_kind: str,
    *, X_user: np.ndarray, target: np.ndarray, y_bin: np.ndarray,
    user_folds: np.ndarray, scorable_mask: np.ndarray,
    multihot_300xK: np.ndarray, y_pair_flat: np.ndarray,
) -> dict:
    """Train one IAB config OOF, project to pair, return all metrics.

    ``target_kind`` is ``"prob"`` (LR) or ``"reg"`` (Ridge / kNN).
    """
    p_per_cat = _iab_oof_per_cat(
        X_user, target,
        user_folds=user_folds, scorable_mask=scorable_mask,
        factory=factory, hparam=hparam,
        predict_method="predict_proba" if target_kind == "prob" else "predict",
    )
    per_cat_auc = _percat_macro_auc(p_per_cat, y_bin, scorable_mask)
    p_pair = _project_iab_to_pair(p_per_cat, multihot_300xK).ravel()
    return {
        "config": name,
        "hparam": hparam,
        "per_cat_macro_auc": per_cat_auc,
        "pair_auc": float(roc_auc_score(y_pair_flat, p_pair)),
        "pair_ap":  float(average_precision_score(y_pair_flat, p_pair)),
    }


def run(
    *, alphas: list[float], ks: list[int], lr_C: float,
    cv_folds: int, seed: int,
    multihot_path: Path, features_path: Path,
    out_path: Path, sentient_profile: str,
) -> None:
    print("=" * 72)
    print("Stage 1: build pair dataset (for the SAME user fold split + pair labels)")
    print("=" * 72)
    pair_ds = build_pair_dataset(**PROFILES[sentient_profile])
    user_ids = pair_ds.user_ids
    n_users = len(user_ids)
    user_folds = _make_user_folds(n_users, cv_folds, seed)
    y_pair_flat = pair_ds.y.astype(np.int64)

    print(f"  users: {n_users}, ads: {len(pair_ds.image_ids)}, "
          f"pairs: {len(y_pair_flat)}")
    print(f"  fold sizes: {pd.Series(user_folds).value_counts().sort_index().tolist()}")

    print()
    print("=" * 72)
    print("Stage 2: build IAB targets (binary y_iab + signed net_likes + frac_positive)")
    print("=" * 72)
    rt_paths = discover_users(CORPUS_ROOTS)
    rt_paths_ordered = {u: rt_paths[u] for u in user_ids if u in rt_paths}

    labels_iab, net_iab, dropped_cols = build_above_mean_labels(
        rt_paths_ordered, multihot_path,
        min_net_likes=1, canonical_path=DEFAULT_T1_PATH,
    )
    cat_cols = list(labels_iab.columns)
    y_per_cat_bin = labels_iab.loc[user_ids].values.astype(np.int64)
    y_per_cat_net = net_iab.loc[user_ids].values.astype(np.float64)

    frac_df, frac_cols = build_frac_positive_target(
        rt_paths_ordered, multihot_path,
        positive_threshold=3, canonical_path=DEFAULT_T1_PATH,
    )
    if list(frac_cols) != cat_cols:
        raise RuntimeError(
            "frac_positive cat ordering disagrees with net_likes cat ordering"
        )
    y_per_cat_frac = frac_df.loc[user_ids].values.astype(np.float64)

    scorable_mask = np.array(
        [_scorable(y_per_cat_bin[:, k]) for k in range(len(cat_cols))]
    )
    print(f"  cats kept: {len(cat_cols)}  (scorable: {int(scorable_mask.sum())})")
    print(f"  net_likes range:    "
          f"[{y_per_cat_net.min():+.1f}, {y_per_cat_net.max():+.1f}], "
          f"mean: {y_per_cat_net.mean():+.3f}")
    print(f"  frac_positive range: "
          f"[{y_per_cat_frac.min():.2f}, {y_per_cat_frac.max():.2f}], "
          f"mean: {y_per_cat_frac.mean():.3f}")
    if dropped_cols:
        print(f"  dropped {len(dropped_cols)} non-canonical/zero-exposure cats")

    M_df, _ = load_multihot(multihot_path)
    image_ids = [
        IMAGE_ID_FOR(c, i)
        for c in range(NUM_CATEGORIES)
        for i in range(IMAGES_PER_CATEGORY)
    ]
    multihot_300xK = M_df.loc[image_ids, cat_cols].values.astype(np.float64)
    multihot_300xK *= scorable_mask[None, :]

    feats_full = pd.read_csv(features_path, index_col="user_id").loc[user_ids]
    X_user_df, dropped_feats = _filter_features(feats_full, min_coverage=5)
    print(f"  user features after coverage filter: {X_user_df.shape} "
          f"(dropped {len(dropped_feats)})")
    X_user = X_user_df.values.astype(np.float64)

    print()
    print("=" * 72)
    print("Stage 3: try each (model, target, hparam) and report pair AUC")
    print("=" * 72)
    rows: list[dict] = []

    print(f"\n  -- LR (binary y_iab, C={lr_C}) --")
    rows.append(_eval_one(
        f"lr_bin (C={lr_C})", _lr, lr_C, "prob",
        X_user=X_user, target=y_per_cat_bin, y_bin=y_per_cat_bin,
        user_folds=user_folds, scorable_mask=scorable_mask,
        multihot_300xK=multihot_300xK, y_pair_flat=y_pair_flat,
    ))
    print(f"    pair AUC = {rows[-1]['pair_auc']:.4f}, "
          f"per-cat macro = {rows[-1]['per_cat_macro_auc']:.4f}")

    print("\n  -- Ridge (target = signed net_likes) --")
    for a in alphas:
        rows.append(_eval_one(
            f"ridge_net (alpha={a:g})", _ridge, a, "reg",
            X_user=X_user, target=y_per_cat_net, y_bin=y_per_cat_bin,
            user_folds=user_folds, scorable_mask=scorable_mask,
            multihot_300xK=multihot_300xK, y_pair_flat=y_pair_flat,
        ))
        print(f"    alpha={a:>6g}  per-cat={rows[-1]['per_cat_macro_auc']:.4f}  "
              f"pair AUC={rows[-1]['pair_auc']:.4f}")

    print("\n  -- Ridge (target = frac_positive in [0, 1]) --")
    for a in alphas:
        rows.append(_eval_one(
            f"ridge_frac (alpha={a:g})", _ridge, a, "reg",
            X_user=X_user, target=y_per_cat_frac, y_bin=y_per_cat_bin,
            user_folds=user_folds, scorable_mask=scorable_mask,
            multihot_300xK=multihot_300xK, y_pair_flat=y_pair_flat,
        ))
        print(f"    alpha={a:>6g}  per-cat={rows[-1]['per_cat_macro_auc']:.4f}  "
              f"pair AUC={rows[-1]['pair_auc']:.4f}")

    print("\n  -- kNN (target = signed net_likes) --")
    for k in ks:
        rows.append(_eval_one(
            f"knn_net (k={k})", _knn, k, "reg",
            X_user=X_user, target=y_per_cat_net, y_bin=y_per_cat_bin,
            user_folds=user_folds, scorable_mask=scorable_mask,
            multihot_300xK=multihot_300xK, y_pair_flat=y_pair_flat,
        ))
        print(f"    k={k:>4d}    per-cat={rows[-1]['per_cat_macro_auc']:.4f}  "
              f"pair AUC={rows[-1]['pair_auc']:.4f}")

    print("\n  -- kNN (target = frac_positive in [0, 1]) --")
    for k in ks:
        rows.append(_eval_one(
            f"knn_frac (k={k})", _knn, k, "reg",
            X_user=X_user, target=y_per_cat_frac, y_bin=y_per_cat_bin,
            user_folds=user_folds, scorable_mask=scorable_mask,
            multihot_300xK=multihot_300xK, y_pair_flat=y_pair_flat,
        ))
        print(f"    k={k:>4d}    per-cat={rows[-1]['per_cat_macro_auc']:.4f}  "
              f"pair AUC={rows[-1]['pair_auc']:.4f}")

    df = pd.DataFrame(rows).sort_values("pair_auc", ascending=False)

    print()
    print("=" * 72)
    print("Summary (sorted by pair AUC):")
    print("=" * 72)
    print(df.round(4).to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  -> {out_path}")

    winner = df.iloc[0]
    print()
    print("=" * 72)
    print(f"WINNER: {winner['config']}")
    print(f"  pair AUC: {winner['pair_auc']:.4f}, pair AP: {winner['pair_ap']:.4f}")
    print(f"  per-cat macro AUC: {winner['per_cat_macro_auc']:.4f}")
    print("=" * 72)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[10, 25, 50, 100, 200])
    p.add_argument("--ks", nargs="+", type=int, default=[3, 5, 10, 25])
    p.add_argument("--lr-C", type=float, default=0.1)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--multihot", type=Path, default=DEFAULT_MULTIHOT)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--sentient-profile", default="with_demographics",
                   choices=list(PROFILES))
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    a = parse_args(argv)
    run(
        alphas=list(a.alphas), ks=list(a.ks), lr_C=a.lr_C,
        cv_folds=a.cv_folds, seed=a.seed,
        multihot_path=a.multihot, features_path=a.features,
        out_path=a.out, sentient_profile=a.sentient_profile,
    )


if __name__ == "__main__":
    main()
