from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttributeSpec:
    id: str
    label: str
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
        options = item.get("options")
        if not isinstance(attr_id, str) or not attr_id.strip():
            raise ValueError("Each attribute needs a non-empty string id")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Attribute {attr_id!r} needs a label")
        if not isinstance(options, list) or not options:
            raise ValueError(f"Attribute {attr_id!r} needs a non-empty options list")
        str_options: list[str] = []
        for opt in options:
            if not isinstance(opt, str) or not opt.strip():
                raise ValueError(f"Invalid option for {attr_id!r}")
            str_options.append(opt)
        specs.append(AttributeSpec(id=attr_id, label=label, options=tuple(str_options)))

    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise ValueError(f"Duplicate attribute id: {spec.id}")
        seen.add(spec.id)

    return tuple(specs)


def validate_profile(
    profile: dict[str, str], vocab: tuple[AttributeSpec, ...]
) -> dict[str, str]:
    allowed = {spec.id: set(spec.options) for spec in vocab}
    out: dict[str, str] = {}
    for spec in vocab:
        val = profile.get(spec.id)
        if val is None or val == "":
            raise ValueError(f"Missing value for {spec.label} ({spec.id})")
        if val not in allowed[spec.id]:
            raise ValueError(f"Invalid value for {spec.label}: {val!r}")
        out[spec.id] = val
    return out
