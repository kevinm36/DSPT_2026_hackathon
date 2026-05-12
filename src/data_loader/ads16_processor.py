"""ADS-16 per-user data processo (Cher Wang @chwang2 with Cursor)

Combines a user's per-image ratings with the multi-hot feature vectors of the
ADS-16 image corpus to produce a single weighted profile vector per user.

Inputs
------
1. Rating CSV (e.g. ``U0001-RT.csv``)
   * ``;``-delimited, quoted fields
   * Row 0: 20 column headers ``Cat0`` .. ``Cat19``
   * Row 1: 20 human-readable category names (e.g. "Clothing & Shoes")
   * Row 2: 20 cells, each a ``,``-separated string of 15 integer ratings in
     ``[0, 5]`` corresponding to the 15 images of that category.

2. Multi-hot feature CSV
   * One row per image
   * Contains an image-id column (default ``image_id``) plus one column per
     binary feature dimension.

Output
------
``user_vector = sum_i  rating_i * multihot_i``  over all 300 images of the
corpus (20 categories x 15 images).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


NUM_CATEGORIES: int = 20
IMAGES_PER_CATEGORY: int = 15
NUM_IMAGES: int = NUM_CATEGORIES * IMAGES_PER_CATEGORY  # 300
MIN_RATING: int = 0
MAX_RATING: int = 5


@dataclass
class UserProfile:
    """The output bundle returned by :meth:`ADS16DataProcessor.process`."""

    user_id: str
    ratings: np.ndarray  # shape (300,), int8, values in [0, 5]
    image_ids: list[str]  # length 300, in rating order
    feature_columns: list[str]  # names of the multi-hot feature columns
    user_vector: np.ndarray  # shape (feature_dim,), float64


class ADS16DataProcessor:
    """Build a single weighted multi-hot profile vector for one ADS-16 user.

    Parameters
    ----------
    rating_csv_path:
        Path to the user's ``*-RT.csv`` rating file.
    multihot_csv_path:
        Path to the per-image multi-hot CSV.
    image_id_column:
        Name of the id column in the multi-hot CSV. Default ``"image_id"``.
    image_id_for:
        Callable mapping ``(category_index, image_index_within_category)`` to
        the matching image id used in the multi-hot CSV. ``category_index`` is
        ``0..19`` and ``image_index_within_category`` is ``0..14``. Defaults to
        ``lambda c, i: f"Cat{c}_{i + 1}"``.
    feature_columns:
        Optional explicit list of feature columns to use from the multi-hot
        CSV. If ``None`` every column other than ``image_id_column`` is used.
    rating_sep / rating_inner_sep / multihot_sep:
        Delimiters for the two CSVs. The defaults match ADS-16.
    """

    def __init__(
        self,
        rating_csv_path: str | Path,
        multihot_csv_path: str | Path,
        *,
        image_id_column: str = "image_id",
        image_id_for: Optional[Callable[[int, int], str]] = None,
        feature_columns: Optional[Sequence[str]] = None,
        rating_sep: str = ";",
        rating_inner_sep: str = ",",
        multihot_sep: str = ",",
    ) -> None:
        self.rating_csv_path = Path(rating_csv_path)
        self.multihot_csv_path = Path(multihot_csv_path)
        self.image_id_column = image_id_column
        self.image_id_for = image_id_for or (lambda c, i: f"Cat{c}_{i + 1}")
        self.feature_columns: Optional[list[str]] = (
            list(feature_columns) if feature_columns is not None else None
        )
        self.rating_sep = rating_sep
        self.rating_inner_sep = rating_inner_sep
        self.multihot_sep = multihot_sep

        self._ratings: Optional[np.ndarray] = None
        self._multihot_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def load_ratings(self) -> np.ndarray:
        """Parse the rating CSV and return a flat ``(300,)`` int8 vector.

        The 20 categories are concatenated in column order
        (``Cat0`` image 1..15, ``Cat1`` image 1..15, ... ``Cat19`` image 1..15)
        so the result is aligned with the canonical image ordering produced by
        :attr:`image_id_for`.
        """
        if not self.rating_csv_path.is_file():
            raise FileNotFoundError(f"Rating CSV not found: {self.rating_csv_path}")

        df = pd.read_csv(self.rating_csv_path, sep=self.rating_sep, dtype=str)

        if df.shape[1] != NUM_CATEGORIES:
            raise ValueError(
                f"Expected {NUM_CATEGORIES} category columns in "
                f"{self.rating_csv_path.name}, got {df.shape[1]}."
            )
        if df.shape[0] < 2:
            raise ValueError(
                f"Expected at least 2 data rows (category names + ratings) in "
                f"{self.rating_csv_path.name}, got {df.shape[0]}."
            )

        rating_row = df.iloc[1].tolist()

        ratings = np.zeros(NUM_IMAGES, dtype=np.int8)
        for c, cell in enumerate(rating_row):
            tokens = [tok.strip() for tok in str(cell).split(self.rating_inner_sep)]
            if len(tokens) != IMAGES_PER_CATEGORY:
                raise ValueError(
                    f"Category column {c} has {len(tokens)} ratings, "
                    f"expected {IMAGES_PER_CATEGORY}."
                )
            try:
                values = [int(t) for t in tokens]
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer rating in category column {c}: {tokens}"
                ) from exc
            for v in values:
                if not MIN_RATING <= v <= MAX_RATING:
                    raise ValueError(
                        f"Rating {v} out of range [{MIN_RATING}, {MAX_RATING}] "
                        f"in category column {c}."
                    )
            start = c * IMAGES_PER_CATEGORY
            ratings[start : start + IMAGES_PER_CATEGORY] = values

        self._ratings = ratings
        return ratings

    def load_multihot(self) -> pd.DataFrame:
        """Load the per-image multi-hot CSV indexed by image id."""
        if not self.multihot_csv_path.is_file():
            raise FileNotFoundError(
                f"Multi-hot CSV not found: {self.multihot_csv_path}"
            )

        df = pd.read_csv(self.multihot_csv_path, sep=self.multihot_sep)
        if self.image_id_column not in df.columns:
            raise ValueError(
                f"Multi-hot CSV {self.multihot_csv_path.name} is missing id "
                f"column {self.image_id_column!r}. "
                f"Found columns: {list(df.columns)}"
            )

        if df[self.image_id_column].duplicated().any():
            dups = df.loc[df[self.image_id_column].duplicated(), self.image_id_column]
            raise ValueError(
                f"Duplicate image ids in multi-hot CSV: "
                f"{dups.tolist()[:5]} ({len(dups)} total)."
            )

        df = df.set_index(self.image_id_column)

        if self.feature_columns is not None:
            missing = [c for c in self.feature_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Feature columns not present in multi-hot CSV: {missing}"
                )
            df = df[list(self.feature_columns)]

        self._multihot_df = df
        return df

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute_user_vector(self) -> np.ndarray:
        """Return ``sum_i rating_i * multihot_i`` over all 300 images."""
        if self._ratings is None:
            self.load_ratings()
        if self._multihot_df is None:
            self.load_multihot()
        assert self._ratings is not None and self._multihot_df is not None

        # Build the list of image ids in rating order (Cat0_1 .. Cat19_15).
        image_ids = [
            self.image_id_for(c, i)
            for c in range(NUM_CATEGORIES)
            for i in range(IMAGES_PER_CATEGORY)
        ]

        missing = [iid for iid in image_ids if iid not in self._multihot_df.index]
        if missing:
            raise KeyError(
                f"Multi-hot CSV is missing rows for {len(missing)} image id(s); "
                f"first few: {missing[:5]}. "
                f"Override `image_id_for` to match your id scheme."
            )

        # Reindex the multi-hot frame to align rows with the flat rating order
        # so we can do the weighted sum in one vectorised step.
        aligned = self._multihot_df.loc[image_ids].to_numpy(dtype=np.float64)
        user_vector = self._ratings.astype(np.float64) @ aligned
        return user_vector

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def process(self, user_id: Optional[str] = None) -> UserProfile:
        """Run the full pipeline and return a :class:`UserProfile`."""
        ratings = self.load_ratings()
        multihot = self.load_multihot()
        user_vector = self.compute_user_vector()

        image_ids = [
            self.image_id_for(c, i)
            for c in range(NUM_CATEGORIES)
            for i in range(IMAGES_PER_CATEGORY)
        ]
        if user_id is None:
            # ``U0001-RT.csv`` -> ``U0001``
            user_id = self.rating_csv_path.stem.split("-")[0]

        return UserProfile(
            user_id=user_id,
            ratings=ratings,
            image_ids=image_ids,
            feature_columns=list(multihot.columns),
            user_vector=user_vector,
        )
