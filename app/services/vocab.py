from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttributeSpec:
    id: str
    label: str
    kind: str
    value_type: str
    options: tuple[str, ...]


def load_profile_vocab(path: Path) -> tuple[AttributeSpec, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_attrs = data.get("attributes")
    if not isinstance(raw_attrs, list) or len(raw_attrs) < 1:
        raise ValueError("profile_attributes.json must contain a non-empty attributes array")

    specs: list[AttributeSpec] = []
    for item in raw_attrs:
        if not isinstance(item, dict):
            raise ValueError("Each attribute must be an object")
        attr_id = item.get("id")
        label = item.get("label")
        kind = item.get("kind")
        value_type = item.get("value_type")
        options = item.get("options", [])
        if not isinstance(attr_id, str) or not attr_id.strip():
            raise ValueError("Each attribute needs a non-empty string id")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Attribute {attr_id!r} needs a label")
        if kind not in ("information", "preference"):
            raise ValueError(f"Attribute {attr_id!r} needs kind 'information' or 'preference'")
        if value_type not in ("numerical", "categorical"):
            raise ValueError(f"Attribute {attr_id!r} needs value_type 'numerical' or 'categorical'")
        if not isinstance(options, list):
            raise ValueError(f"Attribute {attr_id!r} options must be a list")
        str_options: list[str] = []
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                raise ValueError(f"Invalid option for {attr_id!r}")
            str_options.append(opt)
        if value_type == "categorical" and not str_options:
            raise ValueError(f"Categorical attribute {attr_id!r} needs a non-empty options list")
        if value_type == "numerical" and str_options:
            raise ValueError(f"Numerical attribute {attr_id!r} must have an empty options list")
        specs.append(
            AttributeSpec(
                id=attr_id,
                label=label,
                kind=kind,
                value_type=value_type,
                options=tuple(str_options),
            )
        )

    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise ValueError(f"Duplicate attribute id: {spec.id}")
        seen.add(spec.id)

    return tuple(specs)


def validate_profile(
    profile: dict[str, str], vocab: tuple[AttributeSpec, ...]
) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in vocab:
        raw = profile.get(spec.id)
        if raw is None:
            raw = ""
        val = raw.strip() if isinstance(raw, str) else str(raw).strip()

        if spec.value_type == "categorical":
            if val == "":
                raise ValueError(f"Missing value for {spec.label} ({spec.id})")
            if val not in set(spec.options):
                raise ValueError(f"Invalid value for {spec.label}: {val!r}")
            out[spec.id] = val
            continue

        if val == "":
            raise ValueError(f"Missing value for {spec.label} ({spec.id})")
        try:
            num = float(val)
        except ValueError as exc:
            raise ValueError(f"{spec.label} must be a number, got {val!r}") from exc
        if not math.isfinite(num):
            raise ValueError(f"{spec.label} must be a finite number")
        out[spec.id] = val

    return out
