"""Per-(user, category) interest matrix from ratings + multi-hot ad features.

The standard ``user_vector`` formula treats the rating as a *weight* on each
ad's category multi-hot - so categories with more tagged ads accumulate
larger sums regardless of preference (popularity / exposure bias).

This module produces an *interest matrix* instead: for each (user, category)
pair, the value summarizes "given an ad has this category, how does the user
feel about it?" - which directly expresses preference, not exposure.

Default formulation (``rating_norm="center"``, ``aggregate="mean"``)::

    residual_i      = rating[u, i] - mean(rating[u, :])
    interest[u, k]  = mean over {ads i where multihot[i, k] = 1} of residual_i

So ``interest[u, k]`` is the user's average above/below-baseline rating on
ads tagged with ``k``. Signed, exposure-corrected, comparable across users.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.data_loader import ADS16DataProcessor


DEFAULT_META_COLS: set[str] = {
    "image_id", "category", "image_index", "path",
    "raw_text", "n_matched", "n_unmatched",
}

RATING_NORMS = {"none", "center", "zscore"}
AGGREGATES = {"mean", "sum", "frac_positive", "like_dislike"}


def load_multihot(multihot_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load per-image multi-hot, returning ``(image_id-indexed DataFrame, cat_cols)``."""
    mh = pd.read_csv(multihot_csv)
    cat_cols = [c for c in mh.columns if c not in DEFAULT_META_COLS]
    if not cat_cols:
        raise ValueError(f"No category columns found in {multihot_csv}")
    return mh.set_index("image_id")[cat_cols].astype(np.float64), cat_cols


def _normalize_ratings(r_raw: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return r_raw
    if mode == "center":
        return r_raw - r_raw.mean()
    if mode == "zscore":
        std = r_raw.std()
        return (r_raw - r_raw.mean()) / std if std > 0 else r_raw - r_raw.mean()
    raise ValueError(f"unknown rating_norm={mode!r}; expected one of {RATING_NORMS}")


def _aggregate(
    r_raw: np.ndarray,
    r_norm: np.ndarray,
    M: np.ndarray,
    coverage_safe: np.ndarray,
    mode: str,
    *,
    positive_threshold: int,
    neutral_value: float,
) -> np.ndarray:
    if mode == "mean":
        # Average residual across ads tagged with cat-k.
        return (r_norm @ M) / coverage_safe
    if mode == "sum":
        # The original "rating-as-weight" formula. Kept for parity / debugging.
        return r_norm @ M
    if mode == "frac_positive":
        # Fraction of cat-k ads the user rated >= threshold (raw scale).
        pos = (r_raw >= positive_threshold).astype(np.float64)
        return (pos @ M) / coverage_safe
    if mode == "like_dislike":
        # Signed deviation from a fixed neutral value, averaged across cat-k ads.
        # 1->-2, 2->-1, 3->0, 4->+1, 5->+2 when neutral=3. Captures "like vs
        # dislike" with magnitude, independent of user's own mean.
        pref = r_raw - neutral_value
        return (pref @ M) / coverage_safe
    raise ValueError(f"unknown aggregate={mode!r}; expected one of {AGGREGATES}")


def build_interest_matrix(
    rating_csv_paths: Mapping[str, Path],
    multihot_csv: Path,
    *,
    image_id_for: Callable[[int, int], str],
    num_categories: int = 20,
    images_per_category: int = 15,
    rating_norm: str = "center",
    aggregate: str = "mean",
    positive_threshold: int = 3,
    neutral_value: float = 3.0,
) -> pd.DataFrame:
    """Compute the (n_users, n_categories) interest matrix.

    Parameters
    ----------
    rating_csv_paths:
        ``{user_id: rating_csv_path}`` mapping.
    multihot_csv:
        Path to the per-image multi-hot CSV (e.g. ``ads16_multihot_t1.csv``).
    image_id_for:
        Maps ``(category_index, image_index)`` -> the image id used as the
        ``image_id`` column of the multi-hot CSV (e.g. ``lambda c, i:
        f"{c+1}_{i+1}"``).
    num_categories, images_per_category:
        ADS-16 layout (default 20 x 15 = 300 images).
    rating_norm:
        ``"none"`` (raw 1-5), ``"center"`` (subtract per-user mean - default),
        or ``"zscore"``.
    aggregate:
        ``"mean"`` (default), ``"sum"``, ``"frac_positive"``, or
        ``"like_dislike"``. See ``_aggregate`` for math.
    positive_threshold:
        Used by ``aggregate="frac_positive"`` (rating >= threshold counts as
        a like).
    neutral_value:
        Used by ``aggregate="like_dislike"`` (preference = rating - neutral).
    """
    M_df, cat_cols = load_multihot(multihot_csv)
    image_ids = [
        image_id_for(c, i)
        for c in range(num_categories)
        for i in range(images_per_category)
    ]
    missing = [iid for iid in image_ids if iid not in M_df.index]
    if missing:
        raise KeyError(
            f"multi-hot CSV is missing {len(missing)} expected image_id(s); "
            f"first few: {missing[:5]}"
        )
    M = M_df.loc[image_ids].values
    coverage = M.sum(axis=0)
    coverage_safe = np.maximum(coverage, 1.0)

    rows: dict[str, np.ndarray] = {}
    for uid, rt_path in rating_csv_paths.items():
        proc = ADS16DataProcessor(
            rating_csv_path=rt_path,
            multihot_csv_path=multihot_csv,
            image_id_for=image_id_for,
            feature_columns=cat_cols,
        )
        r_raw = proc.load_ratings().astype(np.float64)
        r_norm = _normalize_ratings(r_raw, rating_norm)
        rows[uid] = _aggregate(
            r_raw, r_norm, M, coverage_safe, aggregate,
            positive_threshold=positive_threshold,
            neutral_value=neutral_value,
        )

    return pd.DataFrame(rows, index=cat_cols).T  # (n_users, n_cats)
