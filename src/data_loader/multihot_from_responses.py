"""Convert agent JSONL responses into a per-image multi-hot category matrix.

Given the JSONL produced by ``batch_invoke_ads`` - one JSON record per image
with a ``text`` field containing a comma-separated list of IAB tier 2 category
names - this module:

1. Loads the canonical category list from ``IAB-t2.csv``.
2. Parses each record's ``text`` into a list of candidate category strings.
3. Matches them against the canonical list (case- and whitespace-insensitive).
4. Emits a CSV / DataFrame with one row per image and one column per canonical
   category (binary 0/1), plus identifying columns (``image_id``, ``category``,
   ``path``).

The schema is compatible with
:class:`src.data_loader.ads16_processor.ADS16DataProcessor` provided you pass a
matching ``image_id_for`` callable when constructing it.

Run as a script::

    python -m src.data_loader.multihot_from_responses \
        --responses Data/ads16_agent_responses.jsonl \
        --output Data/ads16_multihot.csv

Or import:

    from src.data_loader.multihot_from_responses import responses_to_multihot
    df, unmatched = responses_to_multihot(Path("Data/ads16_agent_responses.jsonl"))
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .agent_processing.categories import DEFAULT_CATEGORIES_PATH, load_categories


DEFAULT_RESPONSES_PATH = Path("Data/ads16_agent_responses.jsonl")
DEFAULT_OUTPUT_PATH = Path("Data/ads16_multihot.csv")


_NORMALIZE_WS = re.compile(r"\s+")
_STRIP_CHARS = " \t\n\r\"'`*-•·.;:[](){}"
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•·])\s+")
_NOISE_TOKENS = {"", "and", "or", "the", "a"}


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation - for fuzzy matching."""
    cleaned = text.strip(_STRIP_CHARS).lower()
    cleaned = _LIST_MARKER.sub("", cleaned)
    return _NORMALIZE_WS.sub(" ", cleaned).strip()


def _build_lookup(categories: list[str]) -> dict[str, str]:
    """Map normalized category name -> canonical name."""
    return {_normalize(c): c for c in categories}


def _build_canonical_patterns(categories: list[str]) -> list[tuple[str, "re.Pattern[str]"]]:
    """Return ``[(canonical_name, compiled_regex), ...]`` sorted longest first.

    The regex matches the canonical name as a whole token (case-insensitive)
    so we can extract categories whose names contain commas before we split
    on commas - and avoid false matches like ``Music`` inside ``Music Video``.
    """
    pairs: list[tuple[str, "re.Pattern[str]"]] = []
    for canon in sorted(categories, key=len, reverse=True):
        # Boundaries that treat letters/digits/' as part of the token, so
        # "Beauty" inside "Beauty Tips" matches but inside "Beautyish" doesn't.
        pattern = re.compile(
            rf"(?<![A-Za-z0-9'&]){re.escape(canon)}(?![A-Za-z0-9'&])",
            re.IGNORECASE,
        )
        pairs.append((canon, pattern))
    return pairs


def parse_response_text(text: str) -> list[str]:
    """Split a model response into raw category-token candidates.

    The agent is prompted to emit a comma-separated list, but we tolerate:
      - leading "Categories:" / "Output:" headers
      - bullets, numbers, quotes around tokens
      - tokens spread across multiple lines
    """
    if not text:
        return []
    # Drop a leading "Categories:" style header if present.
    text = re.sub(r"^[^:\n]{0,40}:\s*", "", text.strip(), count=1)
    raw = re.split(r"[,\n]", text)
    out: list[str] = []
    for tok in raw:
        tok = _LIST_MARKER.sub("", tok).strip(_STRIP_CHARS)
        if tok:
            out.append(tok)
    return out


def assign_categories(
    text: str,
    lookup: dict[str, str],
    *,
    canonical_patterns: Optional[list[tuple[str, "re.Pattern[str]"]]] = None,
) -> tuple[list[str], list[str]]:
    """Return ``(matched canonical categories, unmatched raw tokens)``.

    Two-pass strategy:
      1. Greedy whole-token regex search for every canonical name (longest
         first). Catches names containing commas and skips list markers.
      2. Whatever remains is comma/newline-split and normalized; anything not
         already matched is reported as unmatched.
    """
    if not text:
        return [], []

    if canonical_patterns is None:
        canonical_patterns = _build_canonical_patterns(list(lookup.values()))

    matched: list[str] = []
    seen: set[str] = set()
    remaining = text

    for canon, pattern in canonical_patterns:
        if canon in seen:
            continue
        if pattern.search(remaining):
            seen.add(canon)
            matched.append(canon)
            # Blank out matches so leftover comma-splitting doesn't see them.
            remaining = pattern.sub(" ", remaining)

    unmatched: list[str] = []
    for tok in parse_response_text(remaining):
        norm = _normalize(tok)
        if not norm or norm in _NOISE_TOKENS:
            continue
        canon = lookup.get(norm)
        if canon is not None and canon not in seen:
            seen.add(canon)
            matched.append(canon)
        elif canon is None:
            unmatched.append(tok)
    return matched, unmatched


def _iter_records(responses_path: Path) -> Iterable[dict]:
    if not responses_path.is_file():
        raise FileNotFoundError(f"Responses JSONL not found: {responses_path}")
    with responses_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Bad JSON on line {line_no} of {responses_path}: {exc}"
                ) from exc


def responses_to_multihot(
    responses_path: Path = DEFAULT_RESPONSES_PATH,
    *,
    categories_path: Path = DEFAULT_CATEGORIES_PATH,
    image_id_format: str = "{category}_{image_id}",
    include_error_rows: bool = False,
) -> tuple[pd.DataFrame, Counter[str]]:
    """Build the multi-hot DataFrame from a JSONL of agent responses.

    Parameters
    ----------
    responses_path:
        Path to the JSONL file produced by ``batch_invoke_ads``.
    categories_path:
        Canonical category list (one per line).
    image_id_format:
        Format string used to synthesize the ``image_id`` column from each
        record's ``category`` and ``image_id`` fields. Default
        ``"{category}_{image_id}"`` (e.g. ``"11_3"``).
    include_error_rows:
        If ``True``, include records with an ``error`` field as all-zero rows.
        Default ``False`` (errored records are skipped entirely).

    Returns
    -------
    (DataFrame, Counter)
        - DataFrame with columns ``image_id``, ``category``, ``image_index``,
          ``path``, ``raw_text``, ``n_matched``, ``n_unmatched`` followed by
          one binary column per canonical category, in canonical order.
        - Counter of every unmatched raw token across the corpus, for QA.
    """
    categories = load_categories(categories_path)
    lookup = _build_lookup(categories)
    canonical_patterns = _build_canonical_patterns(categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    rows: list[dict] = []
    multihot_blocks: list[np.ndarray] = []
    unmatched_counter: Counter[str] = Counter()

    for record in _iter_records(responses_path):
        if "error" in record and not include_error_rows:
            continue

        category = record.get("category", "")
        image_id_raw = record.get("image_id", "")
        path = record.get("path", "")
        text = record.get("text", "") or ""

        matched, unmatched = assign_categories(
            text, lookup, canonical_patterns=canonical_patterns,
        )
        unmatched_counter.update(unmatched)

        vec = np.zeros(len(categories), dtype=np.int8)
        for canon in matched:
            vec[cat_index[canon]] = 1
        multihot_blocks.append(vec)

        rows.append(
            {
                "image_id": image_id_format.format(
                    category=category, image_id=image_id_raw
                ),
                "category": category,
                "image_index": image_id_raw,
                "path": path,
                "raw_text": text,
                "n_matched": len(matched),
                "n_unmatched": len(unmatched),
            }
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "image_id", "category", "image_index", "path",
                "raw_text", "n_matched", "n_unmatched", *categories,
            ]
        )
        return empty, unmatched_counter

    meta_df = pd.DataFrame(rows)
    multihot_df = pd.DataFrame(np.vstack(multihot_blocks), columns=categories)
    df = pd.concat([meta_df, multihot_df], axis=1)
    return df, unmatched_counter


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--categories-file", type=Path, default=DEFAULT_CATEGORIES_PATH,
    )
    parser.add_argument(
        "--image-id-format", default="{category}_{image_id}",
        help="Format string with {category} and {image_id} placeholders.",
    )
    parser.add_argument(
        "--include-error-rows", action="store_true",
        help="Include errored records as all-zero rows (default: skip).",
    )
    parser.add_argument(
        "--unmatched-report", type=Path, default=None,
        help="Optional path to write the unmatched-token report as CSV.",
    )
    args = parser.parse_args(argv)

    df, unmatched = responses_to_multihot(
        args.responses,
        categories_path=args.categories_file,
        image_id_format=args.image_id_format,
        include_error_rows=args.include_error_rows,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {args.output}")

    if unmatched:
        top = unmatched.most_common(10)
        print(f"Unmatched tokens: {sum(unmatched.values())} occurrences "
              f"across {len(unmatched)} unique strings. Top 10:")
        for tok, n in top:
            print(f"  {n:>4}  {tok!r}")
        if args.unmatched_report:
            args.unmatched_report.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(unmatched.most_common(), columns=["token", "count"]) \
                .to_csv(args.unmatched_report, index=False)
            print(f"Unmatched report written to {args.unmatched_report}")
    else:
        print("All response tokens matched the canonical list.")


if __name__ == "__main__":
    main()
