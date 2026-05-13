"""Build the (user, ad) pair-level dataset for the "sentient" model.

This is the data layer for the design/sentiment model. It materialises one row
per ``(user, ad)`` pair (~120 users x 300 ads = ~36k rows) joining:

User-side X (compact, 60 dim by default)
  * **50 cluster multihot** from ``Data/sentiment_multihot_clusters.csv`` --
    each user's revealed visual taste over 50 image clusters. Values in
    ``{-1, 0, +1}`` (sign carries preference direction).
  * **10 raw Big-5 answers** from each user's ``UXXXX-B5.csv``.
  * Optional ``include_demographics=True`` adds the 136 cols of
    ``Data/user_features.csv`` (gender / age / income / favourite-sport
    one-hots etc.) for an ablation run.

Ad-side X (~30 dim by default)
  * **20 LLM-extracted design/sentiment fields** from
    ``Data/ads16_design_features.csv``, numerically encoded:
    ordinal int for the 4 ordered enums, one-hot for the 3 unordered enums,
    0/1 for the 5 booleans, raw passthrough for the 8 ints.
  * Optional ``include_iab=True`` appends the IAB-t1 multi-hot for ablation.

Label y (pair-level)
  * ``1 if rating > user_mean else 0`` -- the same "+1 like" half of the
    signed signal used by the content IAB model
    (``src/model/train_logistic.py``). Each user's mean is computed across
    all 300 of their ratings, so the label is per-user-baseline-corrected and
    directly comparable to the content models' labels.

The output of :func:`build_pair_dataset` is everything ``train_sentient`` needs
for a GroupKFold-by-user evaluation:

    PairDataset(X, y, groups, user_ids, image_ids, categories, feature_names)

Run as a script for a quick sanity dump::

    python -m src.model.sentient_dataset --head 5
    python -m src.model.sentient_dataset --include-demographics --head 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.data_loader import (
    ADS16DataProcessor,
    CORPUS_ROOTS,
    IMAGE_ID_FOR,
    IMAGES_PER_CATEGORY,
    NUM_CATEGORIES,
    REPO_ROOT,
    discover_users,
)


DEFAULT_DESIGN = REPO_ROOT / "Data/ads16_design_features.csv"
DEFAULT_USER_FEATURES = REPO_ROOT / "Data/user_features.csv"
DEFAULT_IAB_MULTIHOT = REPO_ROOT / "Data/ads16_multihot_t1.csv"
DEFAULT_CLUSTERS = REPO_ROOT / "Data/sentiment_multihot_clusters.csv"

# IAB t1 multi-hot has 6 metadata columns (image_id, file path, etc.) we never
# want to feed in as features. Same list train_logistic.py uses.
IAB_META_COLS = {
    "image_id", "image_path", "category", "category_index",
    "image_index", "raw_response",
}

# Ordinal mappings for the 4 enum design fields where the levels have a
# natural ordering. Encoding them as ints (0..k) keeps the feature dimension
# small while preserving the rank semantics for linear models.
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "design_quality":         {"low": 0, "medium": 1, "high": 2},
    "perceived_credibility":  {"low": 0, "medium": 1, "high": 2},
    "spamminess":             {"low": 0, "medium": 1, "high": 2},
    "word_count_bin":         {"0": 0, "1-5": 1, "6-15": 2, "16-40": 3, "40+": 4},
}

# Unordered enums get one-hot encoded.
ONEHOT_COLS: list[str] = [
    "primary_subject_type",   # product / person / scene / text-only / mixed
    "usage_context",          # in-use / standalone / lifestyle / abstract / none
    "emotion_valence",        # positive / neutral / negative
]

BOOL_COLS: list[str] = [
    "human_presence", "logo_present",
    "value_proposition_present", "cta_present", "offer_present",
]

# Numeric int 1-10 design fields are passed through unchanged.
PASSTHROUGH_COLS: list[str] = [
    "visual_clutter", "focal_point_presence", "contrast_level",
    "visual_saliency_score", "product_visibility", "brand_prominence",
    "text_density", "readability",
]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_design_features(path: Path = DEFAULT_DESIGN) -> pd.DataFrame:
    """Load the per-ad design feature CSV indexed by ``image_id``."""
    df = pd.read_csv(path, index_col="image_id", dtype={"image_id": str})
    df.index = df.index.astype(str)
    return df


def encode_design_features(design: pd.DataFrame) -> pd.DataFrame:
    """Numeric encoding of the 20-field design schema.

    * ordered enums  -> ordinal int (see ``ORDINAL_MAPS``)
    * bools          -> 0/1
    * unordered enums-> one-hot (one column per level, ``col__level``)
    * 1-10 ints      -> passthrough

    Returns a frame whose every column is numeric and whose index is the
    original ``image_id``. Unknown enum values raise -- we want a hard error
    rather than silent NaN if the LLM ever emits a level outside the schema.
    """
    out = pd.DataFrame(index=design.index)

    for col, mapping in ORDINAL_MAPS.items():
        if col not in design.columns:
            raise KeyError(f"missing design column {col!r}")
        encoded = design[col].astype(str).map(mapping)
        if encoded.isna().any():
            bad = sorted(set(design.loc[encoded.isna(), col].astype(str).unique()))
            raise ValueError(
                f"unknown level(s) for {col!r}: {bad}; "
                f"expected one of {sorted(mapping)}"
            )
        out[col] = encoded.astype(np.int64)

    for col in BOOL_COLS:
        if col not in design.columns:
            raise KeyError(f"missing design column {col!r}")
        out[col] = design[col].astype(int)

    for col in PASSTHROUGH_COLS:
        if col not in design.columns:
            raise KeyError(f"missing design column {col!r}")
        out[col] = design[col].astype(np.int64)

    onehot = pd.get_dummies(
        design[ONEHOT_COLS].astype(str),
        prefix=ONEHOT_COLS, prefix_sep="__", dtype=int,
    )
    out = pd.concat([out, onehot], axis=1)

    out = out.add_prefix("ad__")
    return out


def load_user_features(path: Path = DEFAULT_USER_FEATURES) -> pd.DataFrame:
    """Load the per-user demographic / preference matrix indexed by user_id."""
    return pd.read_csv(path, index_col="user_id")


def load_clusters(path: Path = DEFAULT_CLUSTERS) -> pd.DataFrame:
    """Load the 50-cluster pos/neg multi-hot indexed by user_id.

    Source: ``Data/sentiment_multihot_clusters.csv`` (built by clustering
    each user's 5 IM-POS + 5 IM-NEG images by visual embedding, then
    encoding cluster membership as +1 for pos / -1 for neg / 0 for none).
    """
    df = pd.read_csv(path, index_col="user_id")
    df.columns = [
        f"cluster__{c.split('_', 1)[-1]}" if c.startswith("cluster_") else f"cluster__{c}"
        for c in df.columns
    ]
    return df.astype(np.int64)


def load_b5(corpus_roots: Iterable[Path] = CORPUS_ROOTS) -> pd.DataFrame:
    """Per-user TIPI-style Big-5 raw scores -> ``(n_users, 10)`` DataFrame.

    Each ``UXXXX-B5.csv`` is ``;``-delimited with header ``Question#;Answer``
    and 10 rows of integer answers. We expose the 10 raw answers as columns
    ``b5_q1`` .. ``b5_q10`` rather than computing trait composites (which
    would require knowing which scale variant ADS-16 used for sure).
    """
    rows: list[dict] = []
    for root in corpus_roots:
        if not Path(root).is_dir():
            continue
        for u_dir in sorted(Path(root).iterdir()):
            if not u_dir.is_dir():
                continue
            b5_path = u_dir / f"{u_dir.name}-B5.csv"
            if not b5_path.is_file():
                continue
            df = pd.read_csv(b5_path, sep=";", quotechar='"')
            # Defensive: expect 10 ordered question rows.
            if len(df) != 10 or "Answer" not in df.columns:
                raise ValueError(
                    f"unexpected B5 layout in {b5_path}: "
                    f"shape={df.shape} cols={list(df.columns)}"
                )
            answers = df["Answer"].astype(int).tolist()
            row = {f"b5__q{i + 1}": v for i, v in enumerate(answers)}
            row["user_id"] = u_dir.name
            rows.append(row)
    return pd.DataFrame(rows).set_index("user_id").sort_index()


def load_iab_multihot(path: Path = DEFAULT_IAB_MULTIHOT) -> pd.DataFrame:
    """Load the IAB-t1 multi-hot CSV, drop metadata cols, prefix with ``iab__``."""
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"{path}: missing image_id column")
    df = df.drop_duplicates(subset="image_id", keep="last").set_index("image_id")
    df.index = df.index.astype(str)
    keep = [c for c in df.columns if c not in IAB_META_COLS]
    df = df[keep].astype(int).add_prefix("iab__")
    return df


def load_user_ratings(
    user_paths: dict[str, Path],
) -> tuple[np.ndarray, list[str]]:
    """Stack all users' rating vectors into ``(n_users, 300)`` int8.

    Order is the canonical ``IMAGE_ID_FOR`` order ("1_1" .. "20_15") and
    matches what ``ADS16DataProcessor.load_ratings`` returns.
    """
    user_ids = sorted(user_paths)
    rows = []
    for uid in user_ids:
        proc = ADS16DataProcessor(
            rating_csv_path=user_paths[uid],
            multihot_csv_path=DEFAULT_IAB_MULTIHOT,  # only needed for compute_user_vector; we just call load_ratings
            image_id_for=IMAGE_ID_FOR,
        )
        rows.append(proc.load_ratings())
    return np.vstack(rows), user_ids


# --------------------------------------------------------------------------- #
# Pair dataset
# --------------------------------------------------------------------------- #
@dataclass
class PairDataset:
    """One materialised (user, ad) pair-level dataset.

    Attributes
    ----------
    X : np.ndarray shape (n_pairs, n_features)
        Feature matrix in canonical row order: user-major, ad-minor. Row
        ``u * 300 + i`` is ``(user_ids[u], image_ids[i])``.
    y : np.ndarray shape (n_pairs,)
        Binary label ``1 if rating > user_mean else 0``.
    groups : np.ndarray shape (n_pairs,)
        Integer user index in ``[0, n_users)``. Pass to GroupKFold so all 300
        rows for a given user stay in the same fold.
    ratings : np.ndarray shape (n_pairs,)
        Raw 0..5 rating, kept around for diagnostics / alt label experiments.
    user_means : np.ndarray shape (n_users,)
        Per-user mean rating across all 300 ads.
    user_ids : list[str]
    image_ids : list[str]
        Length 300, in canonical category order.
    categories : np.ndarray shape (n_pairs,)
        IAB t1 category index in ``[0, 20)`` for each row's ad. Used for the
        per-category aggregation that lets us compare side-by-side with the
        content IAB models.
    feature_names : list[str]
        Column names of ``X`` (``user__*``, ``b5__*``, ``ad__*``, optionally ``iab__*``).
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    ratings: np.ndarray
    user_means: np.ndarray
    user_ids: list[str]
    image_ids: list[str]
    categories: np.ndarray
    feature_names: list[str]


def build_pair_dataset(
    *,
    clusters_path: Path = DEFAULT_CLUSTERS,
    user_features_path: Path = DEFAULT_USER_FEATURES,
    design_path: Path = DEFAULT_DESIGN,
    iab_multihot_path: Path = DEFAULT_IAB_MULTIHOT,
    corpus_roots: Iterable[Path] = CORPUS_ROOTS,
    include_clusters: bool = True,
    include_b5: bool = True,
    include_demographics: bool = False,
    include_iab: bool = False,
    user_ids: Optional[list[str]] = None,
) -> PairDataset:
    """Materialise the pair dataset described in the module docstring.

    Parameters
    ----------
    include_clusters
        Append the 50-cluster pos/neg multi-hot to user features. Default on.
    include_b5
        Append per-user TIPI-style Big-5 raw answers (10 cols). Default on.
    include_demographics
        Append the 136 demographic / preference cols of ``user_features.csv``.
        Default OFF -- we keep the user side compact (60 dim) for the small
        n_user=120 regime. Turn on for an ablation run.
    include_iab
        Append the IAB-t1 multi-hot (~34 cols) to ad features for ablation.
        Default off so the ad side is clean "design/sentiment only".
    user_ids
        Optional subset of user ids (for smoke tests). Default: every user
        with all of {cluster row (if requested), B5 file, rating CSV}.
    """
    # ----- ad features (300 rows) ----------------------------------------- #
    design_raw = load_design_features(design_path)
    ad_features = encode_design_features(design_raw)
    if include_iab:
        iab = load_iab_multihot(iab_multihot_path)
        ad_features = ad_features.join(iab, how="left")
        if ad_features.isna().any().any():
            missing = ad_features.index[ad_features.isna().any(axis=1)].tolist()
            raise ValueError(
                f"{len(missing)} ad(s) missing IAB multi-hot rows: "
                f"{missing[:5]}..."
            )

    # ----- user features (n_users rows) ----------------------------------- #
    blocks: list[pd.DataFrame] = []
    if include_clusters:
        blocks.append(load_clusters(clusters_path))
    if include_b5:
        blocks.append(load_b5(corpus_roots))
    if include_demographics:
        blocks.append(load_user_features(user_features_path).add_prefix("user__"))
    if not blocks:
        raise ValueError(
            "all user-side feature blocks disabled; enable at least one of "
            "include_clusters / include_b5 / include_demographics"
        )
    # Inner-join so we only keep users present in every requested block.
    user_features = blocks[0]
    for blk in blocks[1:]:
        user_features = user_features.join(blk, how="inner")

    # ----- ratings (n_users x 300) ---------------------------------------- #
    available_rt = discover_users(corpus_roots)
    candidate_users = [
        u for u in user_features.index if u in available_rt
    ]
    if user_ids is not None:
        candidate_users = [u for u in candidate_users if u in set(user_ids)]
    if not candidate_users:
        raise ValueError("no users with both a feature row and a rating CSV")
    candidate_users = sorted(candidate_users)
    user_features = user_features.loc[candidate_users]

    rating_paths = {u: available_rt[u] for u in candidate_users}
    ratings, ordered_users = load_user_ratings(rating_paths)
    assert ordered_users == candidate_users  # load_user_ratings sorts too

    # Canonical 300-image order.
    image_ids = [
        IMAGE_ID_FOR(c, i)
        for c in range(NUM_CATEGORIES)
        for i in range(IMAGES_PER_CATEGORY)
    ]
    missing_ad = [iid for iid in image_ids if iid not in ad_features.index]
    if missing_ad:
        raise KeyError(
            f"design features missing {len(missing_ad)} image(s): "
            f"{missing_ad[:5]}... (rerun src.ad_design.extract)"
        )
    ad_features_aligned = ad_features.loc[image_ids]

    n_users = len(candidate_users)
    n_ads = len(image_ids)
    n_pairs = n_users * n_ads

    user_means = ratings.mean(axis=1)
    y = (ratings > user_means[:, None]).astype(np.int8).reshape(n_pairs)

    # Build X by tiling user features across ads and ad features across users
    # without ever materialising a Python loop.
    user_block = np.repeat(
        user_features.to_numpy(np.float32), repeats=n_ads, axis=0,
    )                                                # (n_pairs, n_user_feat)
    ad_block = np.tile(
        ad_features_aligned.to_numpy(np.float32), reps=(n_users, 1),
    )                                                # (n_pairs, n_ad_feat)
    X = np.concatenate([user_block, ad_block], axis=1)

    feature_names = (
        list(user_features.columns) + list(ad_features_aligned.columns)
    )

    groups = np.repeat(np.arange(n_users, dtype=np.int32), n_ads)
    flat_ratings = ratings.reshape(n_pairs).astype(np.int8)
    cat_per_ad = np.repeat(np.arange(NUM_CATEGORIES, dtype=np.int32),
                           IMAGES_PER_CATEGORY)
    categories = np.tile(cat_per_ad, n_users)

    return PairDataset(
        X=X, y=y, groups=groups,
        ratings=flat_ratings, user_means=user_means.astype(np.float32),
        user_ids=candidate_users, image_ids=image_ids,
        categories=categories, feature_names=feature_names,
    )


# --------------------------------------------------------------------------- #
# CLI sanity dump
# --------------------------------------------------------------------------- #
def _summary(ds: PairDataset, head: int) -> None:
    print(f"users:        {len(ds.user_ids)}")
    print(f"ads:          {len(ds.image_ids)}")
    print(f"pairs:        {len(ds.y)}")
    print(f"features:     {len(ds.feature_names)}")
    print(f"  cluster__:  {sum(c.startswith('cluster__') for c in ds.feature_names)}")
    print(f"  b5__:       {sum(c.startswith('b5__')      for c in ds.feature_names)}")
    print(f"  user__:     {sum(c.startswith('user__')    for c in ds.feature_names)}")
    print(f"  ad__:       {sum(c.startswith('ad__')      for c in ds.feature_names)}")
    print(f"  iab__:      {sum(c.startswith('iab__')     for c in ds.feature_names)}")
    pos = int(ds.y.sum())
    print(f"label balance: pos={pos}/{len(ds.y)} ({pos / len(ds.y):.1%})")

    print(f"\nFirst {head} feature names: {ds.feature_names[:head]}")
    print(f"Last  {head} feature names: {ds.feature_names[-head:]}")

    print(f"\nFirst {head} pairs:")
    head_df = pd.DataFrame({
        "user_id":   [ds.user_ids[g] for g in ds.groups[:head]],
        "image_id":  [ds.image_ids[i % len(ds.image_ids)] for i in range(head)],
        "category":  ds.categories[:head],
        "rating":    ds.ratings[:head],
        "user_mean": [ds.user_means[g] for g in ds.groups[:head]],
        "label":     ds.y[:head],
    })
    print(head_df.to_string(index=False))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-demographics", action="store_true",
                        help="Append the 136-col demographic block to user side.")
    parser.add_argument("--no-clusters", action="store_true",
                        help="Drop the 50 cluster cols from user side.")
    parser.add_argument("--no-b5", action="store_true",
                        help="Drop the 10 Big-5 cols from user side.")
    parser.add_argument("--include-iab", action="store_true",
                        help="Append IAB-t1 multi-hot to ad features.")
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args(argv)
    ds = build_pair_dataset(
        include_clusters=not args.no_clusters,
        include_b5=not args.no_b5,
        include_demographics=args.include_demographics,
        include_iab=args.include_iab,
    )
    _summary(ds, args.head)


if __name__ == "__main__":
    main()
