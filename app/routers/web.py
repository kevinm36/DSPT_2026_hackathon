from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from app.services import model_service
from app.services.vocab import AttributeSpec, load_profile_vocab, validate_profile

router = APIRouter()

_APP_DIR = Path(__file__).resolve().parents[1]
_TEMPLATES = Jinja2Templates(directory=str(_APP_DIR / "templates"))
_VOCAB_PATH = _APP_DIR / "profile_attributes.json"
_VOCAB: tuple[AttributeSpec, ...] = load_profile_vocab(_VOCAB_PATH)

_MAX_IMAGES = 5
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_ALLOWED_IMAGE_CT = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _error_partial(request: Request, message: str) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="partials/results_error.html",
        context={"message": message},
        status_code=200,
    )


def _parse_profile_csv(content: str) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")
    expected = [spec.id for spec in _VOCAB]
    header = [h.strip() if h else h for h in reader.fieldnames]
    if list(header) != expected:
        raise ValueError(
            "CSV headers must match the template exactly: " + ", ".join(expected)
        )
    rows = list(reader)
    if not rows:
        raise ValueError("CSV must contain one data row")
    row = rows[0]
    return {spec.id: (row.get(spec.id) or "").strip() for spec in _VOCAB}


def _field_as_str(form_data: dict[str, Any], key: str) -> str:
    val = form_data.get(key)
    if isinstance(val, str):
        return val.strip()
    return ""


def _merge_profile_from_form(form_data: dict[str, Any]) -> dict[str, str]:
    return {spec.id: _field_as_str(form_data, spec.id) for spec in _VOCAB}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={"attributes": _VOCAB},
    )


@router.get("/profile/csv-template", response_class=PlainTextResponse)
async def profile_csv_template() -> PlainTextResponse:
    header = ",".join(spec.id for spec in _VOCAB)
    body = ",".join(spec.options[0] for spec in _VOCAB)
    text = header + "\n" + body + "\n"
    return PlainTextResponse(
        text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="profile_template.csv"'},
    )


@router.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    images: list[UploadFile] = File(default_factory=list),
    profile_csv: UploadFile | None = File(None),
) -> HTMLResponse:
    form = await request.form()
    form_map = dict(form)

    non_empty_images = [f for f in images if f.filename]
    if not non_empty_images:
        return _error_partial(request, "Please attach at least one image (up to five).")

    if len(non_empty_images) > _MAX_IMAGES:
        return _error_partial(request, f"At most {_MAX_IMAGES} images are allowed.")

    image_payloads: list[tuple[str, bytes]] = []
    for upload in non_empty_images[:_MAX_IMAGES]:
        raw = await upload.read()
        if len(raw) > _MAX_IMAGE_BYTES:
            return _error_partial(
                request,
                f"Image {upload.filename!r} is too large (max {_MAX_IMAGE_BYTES // (1024 * 1024)} MB per file).",
            )
        ct = (upload.content_type or "").split(";")[0].strip().lower()
        if ct and ct not in _ALLOWED_IMAGE_CT:
            return _error_partial(
                request,
                f"Unsupported type for {upload.filename!r}. Use JPEG, PNG, WebP, or GIF.",
            )
        name = upload.filename or "image"
        image_payloads.append((name, raw))

    profile_raw: dict[str, str]
    try:
        if profile_csv and profile_csv.filename:
            text = (await profile_csv.read()).decode("utf-8")
            profile_raw = _parse_profile_csv(text)
        else:
            profile_raw = _merge_profile_from_form(form_map)
        profile = validate_profile(profile_raw, _VOCAB)
    except ValueError as exc:
        return _error_partial(request, str(exc))
    except UnicodeDecodeError:
        return _error_partial(request, "Profile CSV must be UTF-8 encoded text.")

    predictions = model_service.stub_predict(image_payloads, profile)
    ranked: list[dict[str, Any]] = []
    for rank, pred in enumerate(predictions, start=1):
        ranked.append(
            {
                "rank": rank,
                "filename": pred.filename,
                "affinity": pred.affinity,
                "reason": pred.reason,
            }
        )

    return _TEMPLATES.TemplateResponse(
        request=request,
        name="partials/results.html",
        context={"rows": ranked},
        status_code=200,
    )
