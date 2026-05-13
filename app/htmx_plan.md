# Web UI plan: FastAPI + Jinja2 + HTMX

## Goal

Build a Python web app where the user uploads up to five images and supplies a categorical customer profile (one dropdown per configured attribute). The backend runs a supervised model (stubbed until the real artifact is wired) and returns ranked scores plus per-image attributes and reasoning.

## Stack

- **FastAPI** — routing, multipart uploads, validation.
- **Jinja2** — HTML shell and partials.
- **HTMX** — on the **results page**, each thumbnail uses `hx-get` to load a detail fragment (image attributes + reasoning) into `#detail-slot`.

## Layout

1. **Start page** (`GET /`) — one classic HTML form (`POST /results`, multipart) wrapping both tabs.
2. **Tab: Images** — up to five file inputs; server enforces at most five files and allowed image MIME types.
3. **Tab: Customer profile** — one `<select>` per categorical attribute, or optional one-row CSV (**CSV overrides** dropdowns when valid).
4. **Predict** — full-page submit to `POST /results`; on success the server **redirects** (`303`) to `GET /results/view?rid=…` with ranked thumbnails. On validation failure, `submit_error.html` is returned (`200`).
5. **Results page** (`GET /results/view`) — thumbnails sorted by score (rank #1 first). Clicking a thumbnail loads `GET /results/partials/detail/{rid}/{slot}` into `#detail-slot` (HTMX partial): model image attributes + prediction reason.

## Data contract

- **Multipart form** — files under `images` (repeatable, max five). Profile fields named by each attribute `id` in `profile_attributes.json`, or `profile_csv` file.
- **CSV** — header row exactly matching attribute ids; one body row (first row used if multiple).
- **Vocabulary** — `profile_attributes.json` lists each attribute’s `id`, `label`, and `options`.
- **Results cache** — after a successful run, payload is stored in memory (`services/results_cache.py`) under a UUID `rid` (TTL ~1 hour, capped entries). Thumbnails use `data:` URLs built at submit time.

## Backend modules

| Path | Role |
|------|------|
| `main.py` | App factory, static files, router include. |
| `routers/web.py` | `GET /`, `POST /results`, `GET /results/view`, `GET /results/partials/detail/{rid}/{slot}`, `GET /profile/csv-template`. |
| `services/submission.py` | Shared parsing/validation for images + profile. |
| `services/results_cache.py` | In-memory store for result payloads keyed by `rid`. |
| `services/vocab.py` | Load and validate profile against JSON vocabulary. |
| `services/model_service.py` | `stub_predict` — scores, reasons, and stub **image attributes** per slot. |
| `templates/results.html` | Ranked gallery + `#detail-slot`. |
| `templates/partials/image_detail.html` | HTMX fragment for one image. |
| `static/style.css` | Gallery, active thumb, detail panel. |

## HTMX behavior

- **Results page** only: each `.result-thumb` has `hx-get`, `hx-target="#detail-slot"`, `hx-swap="innerHTML"`.
- Small script marks `.is-active` on the selected thumb on `htmx:beforeRequest`.

## Security notes

- Server-side validation of every categorical value against the vocabulary file.
- Escape dynamic text in Jinja (default for `{{ }}`).
- Result cache is process-local and time-bounded; do not rely on it for sensitive persistence.

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

Edit `profile_attributes.json`: set real `label` strings, `id` keys, and `options` lists. Keep the `attributes` array in sync with your model features.

## Replacing the stub model

Extend `services/model_service.py` so the real model returns, per image slot: `affinity`, `reason`, and `image_attributes` (dict of categorical outputs). `routers/web.py` maps those into the cache payload consumed by `results.html` and `partials/image_detail.html`.
