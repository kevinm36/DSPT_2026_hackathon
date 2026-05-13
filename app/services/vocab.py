from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# UI-only sentinel for categoricals (not stored in profile_attributes.json options).
INVALID_CATEGORICAL_PLACEHOLDER = "invalid"


@dataclass(frozen=True)
class CategoricalOption:
    """Maps a stored attribute token (e.g. multihot column name) to a user-facing label."""

    value: str
    label: str


@dataclass(frozen=True)
class AttributeSpec:
    id: str
    label: str
    kind: str
    value_type: str
    options: tuple[CategoricalOption, ...]


def _default_label_from_token(token: str) -> str:
    """Fallback display label when JSON lists options as plain strings (legacy)."""
    if token.count("__") >= 2:
        part = token.split("__", 2)[-1]
    else:
        part = token
    return part.replace("_", " ").strip()


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

        parsed: list[CategoricalOption] = []
        for opt in options:
            if isinstance(opt, str):
                v = opt.strip()
                if not v:
                    raise ValueError(f"Invalid option string for {attr_id!r}")
                if v == INVALID_CATEGORICAL_PLACEHOLDER:
                    raise ValueError(
                        f"Option value {INVALID_CATEGORICAL_PLACEHOLDER!r} is reserved for the UI; "
                        f"choose a different token for {attr_id!r}"
                    )
                parsed.append(CategoricalOption(value=v, label=_default_label_from_token(v)))
            elif isinstance(opt, dict):
                v = opt.get("value")
                lb = opt.get("label")
                if not isinstance(v, str) or not v.strip():
                    raise ValueError(f"Each categorical option for {attr_id!r} needs a non-empty string value")
                if v.strip() == INVALID_CATEGORICAL_PLACEHOLDER:
                    raise ValueError(
                        f"Option value {INVALID_CATEGORICAL_PLACEHOLDER!r} is reserved for the UI; "
                        f"choose a different token for {attr_id!r}"
                    )
                if not isinstance(lb, str) or not lb.strip():
                    raise ValueError(f"Each categorical option for {attr_id!r} needs a non-empty string label")
                parsed.append(CategoricalOption(value=v.strip(), label=lb.strip()))
            else:
                raise ValueError(f"Each option for {attr_id!r} must be a string or an object with value and label")

        if value_type == "categorical" and not parsed:
            raise ValueError(f"Categorical attribute {attr_id!r} needs a non-empty options list")
        if value_type == "numerical" and parsed:
            raise ValueError(f"Numerical attribute {attr_id!r} must have an empty options list")

        seen_vals: set[str] = set()
        for o in parsed:
            if o.value in seen_vals:
                raise ValueError(f"Duplicate option value {o.value!r} for attribute {attr_id!r}")
            seen_vals.add(o.value)

        specs.append(
            AttributeSpec(
                id=attr_id,
                label=label,
                kind=kind,
                value_type=value_type,
                options=tuple(parsed),
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
            if val == INVALID_CATEGORICAL_PLACEHOLDER:
                out[spec.id] = val
                continue
            allowed = {o.value for o in spec.options}
            if val not in allowed:
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
        if num < 0:
            raise ValueError(f"{spec.label} must be zero or greater (non-negative)")
        out[spec.id] = val

    return out
