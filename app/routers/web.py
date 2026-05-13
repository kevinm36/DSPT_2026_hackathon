from __future__ import annotations

import csv
import io
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.services import model_service, submission
from app.services.results_cache import get as cache_get
from app.services.results_cache import put as cache_put
from app.services.vocab import AttributeSpec, load_profile_vocab

router = APIRouter()

_APP_DIR = Path(__file__).resolve().parents[1]
_TEMPLATES = Jinja2Templates(directory=str(_APP_DIR / "templates"))
_VOCAB_PATH = _APP_DIR / "profile_attributes.json"
_VOCAB: tuple[AttributeSpec, ...] = load_profile_vocab(_VOCAB_PATH)


def _valid_rid(rid: str) -> bool:
    try:
        uuid.UUID(rid)
        return True
    except ValueError:
        return False


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    info = tuple(a for a in _VOCAB if a.kind == "information")
    pref = tuple(a for a in _VOCAB if a.kind == "preference")
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "attributes_information": info,
            "attributes_preference": pref,
        },
    )


def _csv_template_default_cell(spec: AttributeSpec) -> str:
    if spec.value_type == "categorical":
        return spec.options[0].value
    return "0"


@router.get("/profile/csv-template", response_class=PlainTextResponse)
async def profile_csv_template() -> PlainTextResponse:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([spec.id for spec in _VOCAB])
    writer.writerow([_csv_template_default_cell(spec) for spec in _VOCAB])
    text = buf.getvalue()
    return PlainTextResponse(
        text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="profile_template.csv"'},
    )


@router.post("/results", response_model=None)
async def submit_results(
    request: Request,
    images: list[UploadFile] = File(default_factory=list),
    profile_csv: UploadFile | None = File(None),
) -> RedirectResponse | HTMLResponse:
    form = await request.form()
    form_map = dict(form)

    outcome = await submission.collect_submission(
        images=images,
        profile_csv=profile_csv,
        form_map=form_map,
        vocab=_VOCAB,
    )
    if isinstance(outcome, str):
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="submit_error.html",
            context={"message": outcome},
            status_code=200,
        )

    image_rows, profile = outcome
    tuples = [(r["filename"], r["raw"]) for r in image_rows]
    predictions = model_service.stub_predict(tuples, profile)

    items: list[dict[str, Any]] = []
    for rank, pred in enumerate(predictions, start=1):
        row = image_rows[pred.slot_index]
        items.append(
            {
                "slot": pred.slot_index,
                "rank": rank,
                "filename": pred.filename,
                "score": pred.affinity,
                "thumbnail_src": submission.data_url_for_image(row["content_type"], row["raw"]),
                "reason": pred.reason,
                "image_attributes": pred.image_attributes,
            }
        )

    rid = cache_put({"items": items})
    return RedirectResponse(url=f"/results/view?rid={rid}", status_code=303)


@router.get("/results/view", response_class=HTMLResponse)
async def results_view(request: Request, rid: str) -> HTMLResponse:
    if not _valid_rid(rid):
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="submit_error.html",
            context={"message": "Invalid results link."},
            status_code=200,
        )
    data = cache_get(rid)
    if data is None:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="submit_error.html",
            context={
                "message": "Results expired or not found. Submit the form again from the home page.",
            },
            status_code=200,
        )
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="results.html",
        context={"rid": rid, "items": data["items"]},
    )


@router.get("/results/partials/detail/{rid}/{slot}", response_class=HTMLResponse)
async def results_detail_partial(
    request: Request,
    rid: str,
    slot: int,
) -> HTMLResponse:
    if not _valid_rid(rid):
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="partials/image_detail.html",
            context={
                "filename": "",
                "reason": "",
                "image_attributes": {},
                "error": "Invalid results link.",
            },
        )
    data = cache_get(rid)
    if data is None:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="partials/image_detail.html",
            context={
                "filename": "",
                "reason": "",
                "image_attributes": {},
                "error": "Results expired or not found.",
            },
        )
    for it in data["items"]:
        if it["slot"] == slot:
            return _TEMPLATES.TemplateResponse(
                request=request,
                name="partials/image_detail.html",
                context={
                    "filename": it["filename"],
                    "reason": it["reason"],
                    "image_attributes": it["image_attributes"],
                    "error": None,
                },
            )
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="partials/image_detail.html",
        context={
            "filename": "",
            "reason": "",
            "image_attributes": {},
            "error": "No detail for that image.",
        },
    )
