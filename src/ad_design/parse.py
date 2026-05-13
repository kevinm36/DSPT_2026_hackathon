"""Parse design-feature JSONL responses into a flat per-ad DataFrame.

Reads the JSONL produced by ``extract.batch_invoke`` and emits one DataFrame
row per ad image with all schema fields as columns. Records that:

  * have an ``error`` field (network / agent-side failures), OR
  * fail JSON parsing (the model returned non-JSON text), OR
  * fail schema validation (wrong types, out-of-range ints, unknown enum vals)

are reported as part of the parser's stats but not included in the output
DataFrame by default.

Run as a script::

    python -m src.ad_design.parse                 # default in/out paths
    python -m src.ad_design.parse --include-errors  # keep partial rows too
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import FIELD_NAMES, validate_response


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSES_PATH = REPO_ROOT / "Data/ads16_design_features.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "Data/ads16_design_features.csv"


_JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.DOTALL | re.IGNORECASE)


def extract_json_payload(text: str) -> Optional[str]:
    """Pull the JSON object out of a model response.

    Tries (in order):
      1. ``<json>...</json>`` tagged block (the contract the prompt asks for)
      2. The first top-level ``{...}`` block in the text (fallback)

    Returns the JSON string, or None if neither pattern matched.
    """
    m = _JSON_TAG_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback - find first balanced top-level object
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_record(record: dict) -> tuple[Optional[dict], list[str]]:
    """Parse one JSONL record. Returns ``(parsed_dict_or_None, issues)``.

    ``parsed_dict_or_None`` is the schema-validated dict (or None if parsing
    failed). ``issues`` is a non-empty list iff something went wrong - useful
    for the per-row error report even when the row is dropped from the
    output DataFrame.
    """
    if "error" in record:
        return None, [f"agent error: {record['error']}"]
    text = record.get("text")
    if not isinstance(text, str):
        return None, ["missing 'text' field in record"]
    payload = extract_json_payload(text)
    if payload is None:
        return None, [f"no JSON found in response (first 80 chars: {text[:80]!r})"]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as e:
        return None, [f"json decode error: {e.msg} at pos {e.pos}"]
    if not isinstance(parsed, dict):
        return None, [f"top-level JSON is {type(parsed).__name__}, expected object"]
    issues = validate_response(parsed)
    if issues:
        return None, issues
    return parsed, []


def responses_to_features(
    responses_path: Path = DEFAULT_RESPONSES_PATH,
    *,
    include_errors: bool = False,
    image_id_format: str = "{category}_{image_id}",
) -> tuple[pd.DataFrame, Counter[str]]:
    """Convert the response JSONL into a per-ad feature DataFrame.

    Parameters
    ----------
    responses_path:
        JSONL file written by ``extract.batch_invoke``.
    include_errors:
        If True, also emit rows for records that failed parsing/validation,
        with all schema columns set to NaN. Useful when you want a row per
        attempted ad regardless of success.
    image_id_format:
        Template combining ``category`` (sub-folder name) and ``image_id``
        (file stem) into the row's ``image_id`` value. Default
        ``"{category}_{image_id}"`` matches the convention used elsewhere in
        the repo (e.g. ``"1_1"`` for ``Ads/1/1.png``).

    Returns
    -------
    df, issues_counter
        ``df`` is indexed by ``image_id`` and has one column per schema field.
        ``issues_counter`` maps issue-message-prefix -> count, useful for
        quickly seeing why responses are getting dropped.
    """
    if not responses_path.is_file():
        raise FileNotFoundError(responses_path)

    rows: list[dict] = []
    issues_counter: Counter[str] = Counter()
    n_ok = 0
    n_err = 0
    n_total = 0

    # If a path was retried, keep the latest successful record. Records are
    # appended in arrival order in the JSONL, so iterate to the end and use
    # last-write-wins.
    seen_paths: dict[str, dict] = {}
    with responses_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            key = rec.get("path") or rec.get("image_id") or ""
            if "error" in rec and key in seen_paths and "error" not in seen_paths[key]:
                # Don't downgrade an earlier success with a later error.
                continue
            seen_paths[key] = rec

    for rec in seen_paths.values():
        parsed, issues = parse_record(rec)
        image_id = image_id_format.format(
            category=rec.get("category", ""),
            image_id=rec.get("image_id", ""),
        )
        if parsed is None:
            n_err += 1
            for issue in issues:
                # Bucket by first 60 chars so similar errors aggregate.
                issues_counter[issue[:60]] += 1
            if include_errors:
                rows.append({"image_id": image_id, **{f: None for f in FIELD_NAMES}})
            continue
        n_ok += 1
        rows.append({"image_id": image_id, **{f: parsed[f] for f in FIELD_NAMES}})

    print(
        f"Parsed {n_total} record(s) from {responses_path.name}: "
        f"{n_ok} ok, {n_err} failed."
    )
    if issues_counter:
        print("Top issue patterns:")
        for issue, count in issues_counter.most_common(5):
            print(f"  {count:>3}x  {issue}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("image_id")
    return df, issues_counter


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--include-errors", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    df, _ = responses_to_features(args.responses, include_errors=args.include_errors)
    if df.empty:
        print("No rows produced; nothing to write.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output)
    print(f"Wrote {df.shape[0]} rows x {df.shape[1]} cols to {args.output}")


if __name__ == "__main__":
    main()
