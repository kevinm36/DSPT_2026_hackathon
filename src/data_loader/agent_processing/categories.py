"""Canonical IAB Tier 2 category list and the categorization prompt template.

Both ``batch_invoke_ads`` (caller side) and ``multihot_from_responses``
(post-processor) need the exact same canonical list, so it lives here once.

The on-disk file ``IAB-t2.csv`` is one category per line (CSV-quoted when the
name itself contains commas).
"""

from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_CATEGORIES_PATH: Path = Path(__file__).parent / "IAB-t2.csv"

PROMPT_INSTRUCTION: str = (
    "Assign IAB tier2 categories to the image. Return a list of categories "
    "separated by commas. Do not provide any reasoning or words in addition. "
    "Do not invent new categories."
)


def load_categories(path: Path = DEFAULT_CATEGORIES_PATH) -> list[str]:
    """Return the canonical IAB Tier 2 category list, in file order.

    Handles CSV-quoted lines (some category names contain commas, e.g.
    ``"Death, Injury, or Military Conflict"``). Blank lines are skipped.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Categories file not found: {path}")

    cats: list[str] = []
    seen: set[str] = set()
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            name = row[0].strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            cats.append(name)
    return cats


def build_categorization_prompt(
    instruction: str = PROMPT_INSTRUCTION,
    categories: list[str] | None = None,
    *,
    categories_path: Path = DEFAULT_CATEGORIES_PATH,
) -> str:
    """Build the full prompt: instruction + the category list appended."""
    if categories is None:
        categories = load_categories(categories_path)
    cat_block = "\n".join(categories)
    return f"{instruction}\nFull list of categories\n{cat_block}"
