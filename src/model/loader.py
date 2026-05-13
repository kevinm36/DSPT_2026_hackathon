"""Load saved model bundles and predict on new user features.

Bundles are produced by ``compare_models --save-models`` and live in
``Data/models/`` by default.

The serving API is intentionally tiny:

    >>> from src.model.loader import load_bundle
    >>> b = load_bundle("Ridge")
    >>> b.feature_names           # the 112 columns the model expects, in order
    ['inf__age', 'inf__gender_male', ...]
    >>> b.scorable_categories     # the 26 IAB t1 cats the model emits scores for
    ['Automotive', 'Books and Literature', ...]
    >>> scores = b.predict({"inf__age": 35, "inf__gender_male": 1, ...})
    >>> scores
    {'Automotive': -8.34, 'Books and Literature': -1.21, 'Business and Finance': ...}

Inputs accepted by ``predict``:
    * dict[str, value]            -- missing keys default to 0 (handy for web forms)
    * pandas.Series               -- indexed by feature name
    * pandas.DataFrame            -- batch (rows = users), columns indexed by name
    * numpy.ndarray               -- 1D vector or 2D matrix in feature_names order

Output:
    * for dict / Series / 1D array  -> dict {category_name: score}
    * for DataFrame / 2D array      -> DataFrame indexed like input, columns = cats

Score units (also stored in ``bundle.score_units``):
    * LR    -> P(net_likes >= min_net_likes), in [0, 1]
    * Ridge -> predicted net_likes (signed real - more negative = more "rejector")
    * kNN   -> predicted net_likes (signed real - same units as Ridge)
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = REPO_ROOT / "saved_models"

MODEL_FILES: dict[str, str] = {
    "LR":    "lr_model.joblib",
    "Ridge": "ridge_model.joblib",
    "kNN":   "knn_model.joblib",
}

InputType = Union[Mapping[str, float], pd.Series, pd.DataFrame, np.ndarray]


class ModelBundle:
    """Thin wrapper around a saved bundle with a uniform predict API."""

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.model_name: str = raw["model_name"]
        self.best_hparam: dict = raw["best_hparam"]
        self.feature_names: list[str] = raw["feature_names"]
        self.category_names: list[str] = raw["category_names"]
        self.scorable_categories: list[str] = raw["scorable_categories"]
        self.skipped_categories: list[str] = raw["skipped_categories"]
        self.models = raw["models"]                # {cat: fitted Pipeline}
        self.score_kind: str = raw["score_kind"]   # "predict" or "predict_proba"
        self.score_units: str = raw["score_units"]
        self.training_metadata: dict = raw["training_metadata"]

    # ------------------------------------------------------------------ #
    # Input coercion
    # ------------------------------------------------------------------ #
    def _to_matrix(self, user: InputType) -> tuple[np.ndarray, "pd.Index | None"]:
        """Return ``(X, row_index_or_None)``.

        ``row_index_or_None`` is a pandas Index when the input was a DataFrame
        (used to label the output DataFrame) and ``None`` for single-row
        inputs (which return a dict).
        """
        n_feat = len(self.feature_names)

        if isinstance(user, pd.DataFrame):
            X = user.reindex(columns=self.feature_names).fillna(0.0).to_numpy(np.float64)
            return X, user.index

        if isinstance(user, pd.Series):
            X = user.reindex(self.feature_names).fillna(0.0).to_numpy(np.float64)
            return X.reshape(1, -1), None

        if isinstance(user, Mapping):
            X = np.array(
                [[float(user.get(name, 0.0)) for name in self.feature_names]],
                dtype=np.float64,
            )
            return X, None

        if isinstance(user, np.ndarray):
            X = np.asarray(user, dtype=np.float64)
            if X.ndim == 1:
                if X.size != n_feat:
                    raise ValueError(
                        f"1D array has length {X.size}; expected {n_feat} "
                        f"(see bundle.feature_names for the order)."
                    )
                return X.reshape(1, -1), None
            if X.ndim == 2:
                if X.shape[1] != n_feat:
                    raise ValueError(
                        f"2D array has {X.shape[1]} cols; expected {n_feat}."
                    )
                return X, None
            raise ValueError(f"ndarray must be 1D or 2D, got {X.ndim}D")

        raise TypeError(
            f"Unsupported input type {type(user).__name__}; "
            f"pass a dict, pandas Series/DataFrame, or numpy array."
        )

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def _score_one(self, pipe, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of length n_samples - the per-sample category score."""
        if self.score_kind == "predict_proba":
            return pipe.predict_proba(X)[:, 1]
        return pipe.predict(X)

    def predict(self, user: InputType):
        """Score ``user`` across every scorable category.

        Returns a dict for single-row inputs, a DataFrame for batch inputs.
        See module docstring for input-type details.
        """
        X, row_index = self._to_matrix(user)
        is_batch = X.shape[0] > 1 or row_index is not None

        scores: dict[str, np.ndarray] = {}
        for cat, pipe in self.models.items():
            scores[cat] = self._score_one(pipe, X)

        if not is_batch:
            return {cat: float(arr[0]) for cat, arr in scores.items()}

        df = pd.DataFrame(scores, index=row_index)
        df.columns.name = "category"
        return df

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        meta = self.training_metadata
        return (
            f"<ModelBundle {self.model_name!r} "
            f"hparam={self.best_hparam} "
            f"n_features={meta['n_features']} "
            f"n_users={meta['n_users']} "
            f"cats={len(self.scorable_categories)}>"
        )


def load_bundle(
    model_name: str,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> ModelBundle:
    """Load a saved bundle by name. ``model_name`` in {'LR', 'Ridge', 'kNN'}."""
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of {list(MODEL_FILES)}"
        )
    path = models_dir / MODEL_FILES[model_name]
    if not path.is_file():
        raise FileNotFoundError(
            f"No saved bundle at {path}. Run "
            f"`python -m src.model.compare_models --save-models` first."
        )
    return ModelBundle(joblib.load(path))


def load_all(models_dir: Path = DEFAULT_MODELS_DIR) -> dict[str, ModelBundle]:
    """Load all three bundles. Useful for side-by-side demo."""
    return {name: load_bundle(name, models_dir) for name in MODEL_FILES}
