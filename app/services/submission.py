from __future__ import annotations

import base64
import csv
import io
from typing import Any

from fastapi import UploadFile

from app.services.vocab import (
    INVALID_CATEGORICAL_PLACEHOLDER,
    AttributeSpec,
    validate_profile,
)

_MAX_IMAGES = 5
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_ALLOWED_IMAGE_CT = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def parse_profile_csv(content: str, vocab: tuple[AttributeSpec, ...]) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")
    expected = [spec.id for spec in vocab]
    header = [h.strip() if h else h for h in reader.fieldnames]
    if list(header) != expected:
        raise ValueError(
            "CSV headers must match the template exactly: " + ", ".join(expected)
        )
    rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain one data row")
    row = rows[0]
    out = {spec.id: (row.get(spec.id) or "").strip() for spec in vocab}
    for spec in vocab:
        if spec.value_type != "categorical":
            continue
        v = out[spec.id]
        if not v:
            continue
        allowed = {o.value for o in spec.options}
        if v == INVALID_CATEGORICAL_PLACEHOLDER:
            continue
        if v not in allowed:
            out[spec.id] = INVALID_CATEGORICAL_PLACEHOLDER
    return out


def field_as_str(form_data: dict[str, Any], key: str) -> str:
    val = form_data.get(key)
    if isinstance(val, str):
        return val.strip()
    return ""


def merge_profile_from_form(form_data: dict[str, Any], vocab: tuple[AttributeSpec, ...]) -> dict[str, str]:
    return {spec.id: field_as_str(form_data, spec.id) for spec in vocab}


def data_url_for_image(content_type: str, raw: bytes) -> str:
    ct = (content_type or "image/jpeg").split(";")[0].strip().lower()
    if ct not in _ALLOWED_IMAGE_CT:
        ct = "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{ct};base64,{b64}"


async def collect_submission(
    *,
    images: list[UploadFile],
    profile_csv: UploadFile | None,
    form_map: dict[str, Any],
    vocab: tuple[AttributeSpec, ...],
) -> tuple[list[dict[str, Any]], dict[str, str]] | str:
    """
    Returns (image_rows, profile) on success, where each image row has
    filename, raw bytes, content_type. On failure returns an error message string.
    """
    non_empty_images = [f for f in images if f.filename]
    if not non_empty_images:
        return "Please attach at least one image (up to five)."

    if len(non_empty_images) > _MAX_IMAGES:
        return f"At most {_MAX_IMAGES} images are allowed."

    image_rows: list[dict[str, Any]] = []
    for upload in non_empty_images[:_MAX_IMAGES]:
        raw = await upload.read()
        if len(raw) > _MAX_IMAGE_BYTES:
            return (
                f"Image {upload.filename!r} is too large (max {_MAX_IMAGE_BYTES // (1024 * 1024)} MB per file)."
            )
        ct = (upload.content_type or "").split(";")[0].strip().lower()
        if ct and ct not in _ALLOWED_IMAGE_CT:
            return (
                f"Unsupported type for {upload.filename!r}. Use JPEG, PNG, WebP, or GIF."
            )
        name = upload.filename or "image"
        image_rows.append({"filename": name, "raw": raw, "content_type": ct or "image/jpeg"})

    profile_raw: dict[str, str]
    try:
        if profile_csv and profile_csv.filename:
            text = (await profile_csv.read()).decode("utf-8")
            profile_raw = parse_profile_csv(text, vocab)
        else:
            profile_raw = merge_profile_from_form(form_map, vocab)
        profile = validate_profile(profile_raw, vocab)
    except ValueError as exc:
        return str(exc)
    except UnicodeDecodeError:
        return "Profile CSV must be UTF-8 encoded text."

    invalid_labels = sorted(
        spec.label
        for spec in vocab
        if spec.value_type == "categorical"
        and profile.get(spec.id) == INVALID_CATEGORICAL_PLACEHOLDER
    )
    if invalid_labels:
        joined = ", ".join(invalid_labels)
        return f"Some profile fields are invalid. Invalid fields: {joined}."

    return image_rows, profile
