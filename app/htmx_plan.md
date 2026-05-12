# Web UI plan: FastAPI + Jinja2 + HTMX

## Goal

Build a Python web app where the user uploads up to five images and supplies a categorical customer profile (one dropdown per configured attribute). The backend runs a supervised model (stubbed until the real artifact is wired) and returns an affinity score and short reason per image, ranked for display.

## Stack

- **FastAPI** — routing, multipart uploads, validation.
- **Jinja2** — HTML shell, tabs, and **HTML partials** returned to HTMX.
- **HTMX** — `POST /predict` swaps the `#results` region with a fragment (no SPA build step).

Django remains a valid alternative; this package uses FastAPI for a smaller, hackathon-friendly surface.

## Layout

1. **Start page** (`GET /`) — one form wrapping both tabs so a single submit serializes images and profile fields.
2. **Tab: Images** — up to five file inputs (`Browse`); server enforces at most five files and allowed image MIME types.
3. **Tab: Customer profile** — one `<select>` per categorical attribute. Optional one-row CSV upload whose columns match the configured attribute ids. If both CSV and dropdowns are present, **CSV wins** after successful parse; otherwise dropdown values are used.
4. **Predict** — submits the form via HTMX to `POST /predict`; response is **only** the inner HTML for `#results` (success table or error alert).
5. **Results** — per image: rank, filename, affinity (float), reason (text). Rows sorted by affinity descending.

## Data contract

- **Multipart form** — files under field name `images` (repeatable, max five). Profile fields named by each attribute `id` in `profile_attributes.json`, or `profile_csv` file.
- **CSV** — header row with column names exactly matching attribute ids in `profile_attributes.json`; one body row (first row used if multiple).
- **Vocabulary** — `profile_attributes.json` lists each attribute’s `id`, human `label`, and allowed `options`. Replace this file when real names and categories are ready; no code change required if the schema stays the same.

## Backend modules

| Path | Role |
|------|------|
| `main.py` | App factory, static files, templates, router include, lifespan hooks if needed later for model load. |
| `routers/web.py` | `GET /`, `POST /predict`, `GET /profile/csv-template` (downloadable template). |
| `services/vocab.py` | Load and validate profile against JSON vocabulary. |
| `services/model_service.py` | `predict(...)` — currently a **stub** returning deterministic placeholder scores and reasons. |
| `templates/` | `base.html`, `index.html`, `partials/results.html`, `partials/results_error.html`. |
| `static/style.css` | Layout and tabs. |

## HTMX behavior

- Root `<form>` uses `hx-post="/predict"`, `hx-target="#results"`, `hx-swap="innerHTML"`, `enctype="multipart/form-data"`.
- Predict control is `type="submit"` inside the form so files and fields post together.
- Validation and stub errors return **HTTP 200** with `partials/results_error.html` so HTMX always swaps `#results` without extra `hx-on::` error handling.

## Security notes

- Server-side validation of every categorical value against the vocabulary file.
- Escape all dynamic text in Jinja (`{{ x | e }}` is default for variables).
- Enforce upload size limits in FastAPI / Starlette configuration when deploying.

## Run locally (Conda)

Dependencies are declared only in `environment.yml` in this folder (conda-forge, no pip section).

From the **repository root** (parent of this `app/` directory):

```bash
cd /path/to/DSPT_2026_hackathon
conda env create -f app/environment.yml
conda activate dspt-affinity-ui
uvicorn app.main:app --reload --app-dir .
```

After you edit `app/environment.yml`, update the existing environment:

```bash
conda activate dspt-affinity-ui
conda env update -f app/environment.yml --prune
```

Open `http://127.0.0.1:8000/`.

## Replacing placeholder profile metadata

Edit `profile_attributes.json`: set real `label` strings, `id` keys (used as form field names and CSV headers), and full `options` lists per attribute. Keep the `attributes` array in sync with your model features; length is not hard-coded in Python beyond requiring at least one attribute.

## Replacing the stub model

Implement loading your trained artifact inside `services/model_service.py` (for example load the artifact in a FastAPI lifespan handler and store it on the application instance’s `.state`) and replace `stub_predict` with calls into your pipeline. Keep the return shape: list of objects with `filename`, `affinity`, `reason`, sorted by descending affinity.
