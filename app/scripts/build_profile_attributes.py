#!/usr/bin/env python3
"""Regenerate profile_attributes.json from user_features_manifest.json."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def _human_label(attr_id: str) -> str:
    if "__" not in attr_id:
        return attr_id.replace("_", " ").strip().title()
    rest = attr_id.split("__", 1)[1]
    return rest.replace("_", " ").strip().title()


def main() -> None:
    app_dir = Path(__file__).resolve().parents[1]
    manifest_path = app_dir / "user_features_manifest.json"
    out_path = app_dir / "profile_attributes.json"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    numeric: list[str] = list(data["numeric_columns"])
    categorical: list[str] = list(data["categorical_columns"])

    attributes: list[dict[str, object]] = []

    for col in numeric:
        prefix = col.split("__", 1)[0] if "__" in col else ""
        if prefix == "inf":
            kind = "information"
        elif prefix == "pref":
            kind = "preference"
        else:
            kind = "information"
        attributes.append(
            {
                "id": col,
                "kind": kind,
                "value_type": "numerical",
                "label": _human_label(col),
                "options": [],
            }
        )

    group_options: dict[str, list[str]] = defaultdict(list)
    group_order: list[str] = []
    for col in categorical:
        parts = col.split("__", 2)
        if len(parts) != 3:
            raise ValueError(f"Expected format type__name__value, got: {col!r}")
        prefix, _aname, _val = parts
        group_id = f"{prefix}__{_aname}"
        if group_id not in group_options:
            group_order.append(group_id)
        group_options[group_id].append(col)

    inf_groups = [g for g in group_order if g.startswith("inf__")]
    pref_groups = [g for g in group_order if g.startswith("pref__")]

    for group_id in inf_groups + pref_groups:
        prefix = group_id.split("__", 1)[0]
        kind = "information" if prefix == "inf" else "preference"
        opts = group_options[group_id]
        attributes.append(
            {
                "id": group_id,
                "kind": kind,
                "value_type": "categorical",
                "label": _human_label(group_id),
                "options": opts,
            }
        )

    out_path.write_text(
        json.dumps({"attributes": attributes}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(attributes)} attributes to {out_path}")

    sample_path = app_dir / "sample_valid_profile.csv"
    ids = [str(a["id"]) for a in attributes]
    row: list[str] = []
    for a in attributes:
        if a["value_type"] == "numerical":
            row.append("0")
        else:
            row.append(str(a["options"][0]))
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(ids)
        w.writerow(row)
    print(f"Wrote sample CSV to {sample_path}")


if __name__ == "__main__":
    main()
