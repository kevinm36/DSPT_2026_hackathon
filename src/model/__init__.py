"""Ridge baseline for ADS-16 user-interest modeling."""

from .interest_matrix import (
    DEFAULT_META_COLS,
    build_interest_matrix,
    load_multihot,
)

__all__ = ["build_interest_matrix", "load_multihot", "DEFAULT_META_COLS"]
