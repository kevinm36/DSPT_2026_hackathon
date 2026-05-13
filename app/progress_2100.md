# Progress summary — Image affinity web app (`app/`)

This document summarizes what the **Image affinity ranker** web application does, how it is built, and how end users operate it. All implementation lives under the `app/` directory unless noted.

---

## Purpose

Users upload **one to five** images and provide a **customer profile** (numerical and categorical fields derived from the ADS16-style feature manifest). The backend runs a **supervised model** (currently a **deterministic stub**) that assigns an **affinity score** and **reasoning text** per image. Results appear on a **dedicated results page**: thumbnails **ranked by score**, with **HTMX-driven detail panels** per image (stub image attributes + prediction reason).

---

## Technology stack

| Layer | Choice |
|--------|--------|
| Web framework | **FastAPI** |
| Templates | **Jinja2** |
| Interactivity | **HTMX** on the results page (partial HTML loads for per-image detail) |
| Styling | `app/static/style.css` (layout, tabs, gallery, detail slot) |
| Dependencies | Declared in `app/environment.yml` (**conda-forge** only) |
| Python | 3.11+ (per environment file) |

---

## HTTP routes and behavior

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Home: two-tab form (Images + Customer profile), multipart submit to `/results`. |
| `POST` | `/results` | Validates images + profile from **multipart form fields**; runs model; stores payload in memory cache; **303 redirect** to `/results/view?rid=…`. Uses `response_model=None` because the handler returns either `RedirectResponse` or an HTML error page. |
| `GET` | `/results/view?rid=…` | Full-page ranked gallery; thumbnails trigger HTMX loads into `#detail-slot`. |
| `GET` | `/results/partials/detail/{rid}/{slot}` | HTML fragment: model image attributes + reasoning for one slot. |
| `GET` | `/profile/csv-template` | Downloadable UTF-8 CSV: header row = all profile attribute ids, second row = defaults (`0` for numerical, first categorical **value** per group). |

**Errors:** Failed validation on `POST /results` returns `submit_error.html` (200) with a message and link back to `/`.

---

## Feature list (functional)

1. **Tabbed home UI** — **Images** tab (up to five file inputs, same field name `images`) and **Customer profile** tab (form fields + optional **local-only** CSV picker that fills the form in the browser).
2. **Image rules** — At least one image required; max five; MIME checks (JPEG, PNG, WebP, GIF); per-file size cap (see `submission` constants).
3. **Profile schema** — Loaded from `app/profile_attributes.json`:
   - **Information** vs **Preference** sections (`kind`: `information` for `inf__…`, `preference` for `pref__…`).
   - **Numerical** attributes: free numeric input, **non-negative** enforced in UI (`min="0"`) and server-side (`validate_profile`); blank or invalid values are rejected before predict with an aggregate error (information fields listed before preference fields).
   - **Categorical** attributes: one dropdown per **grouped** manifest feature (`type__attribute_name`); each option is a **`value` + `label`** pair (`value` = full multihot token; `label` = arbitrary display string editable in JSON). Includes a blank-looking **`invalid`** sentinel for bad CSV tokens until the user fixes the value.
   - **Legacy support:** categorical `options` may still be a list of strings; labels are auto-derived.
4. **CSV helper (browser)** — Optional file read **only in the client** to populate the form from a one-row UTF-8 CSV whose headers match attribute `id` list **exactly** (order-sensitive). **Predict** always submits the **current form**; the CSV file itself is not uploaded.
5. **Sample assets** — `app/sample_valid_profile.csv` (valid example row); regeneratable with the build script.
6. **Results flow** — In-memory **UUID-keyed cache** (`results_cache.py`) with TTL and max entry cap; holds ranked items with **base64 data URL** thumbnails for display after redirect.
7. **Ranked gallery** — Thumbnails ordered by descending score; rank badge and filename/score metadata.
8. **HTMX detail** — Clicking a thumbnail loads a partial with **image-level stub attributes** and **prediction reason** text.
9. **Navigation** — “New prediction” returns to `/`; template download and inline hints link to `/profile/csv-template`.
10. **Stub model** — `model_service.AgentModel.predict` on **`default_agent_model`**: deterministic scores from hashes + profile digest; returns `slot_index`, `image_attributes` dict, and `reason` per image (replace with real trained pipeline later). `stub_predict(...)` remains a thin delegate for compatibility.

---

## Key files (under `app/`)

| Path | Role |
|------|------|
| `main.py` | FastAPI app, static mount, router include. |
| `routers/web.py` | All routes above; CSV template writer; wires vocab + submission + cache. |
| `services/submission.py` | Multipart image handling; profile from **form fields** via `merge_profile_from_form`; `parse_profile_csv` kept for tooling/tests; `data_url_for_image`. |
| `services/vocab.py` | `AttributeSpec`, `CategoricalOption`, `load_profile_vocab`, `validate_profile`. |
| `services/results_cache.py` | UUID store for result payloads between redirect and HTMX partials. |
| `services/model_service.py` | `AgentModel` + `default_agent_model`; stub `predict`; `ImagePrediction` output type. |
| `profile_attributes.json` | Authoritative profile field definitions (often rebuilt from manifest). |
| `user_features_manifest.json` | Source list of `numeric_columns` and `categorical_columns` feature names. |
| `scripts/build_profile_attributes.py` | Regenerates `profile_attributes.json` + `sample_valid_profile.csv` from the manifest. |
| `templates/base.html` | Shell, HTMX CDN, global styles. |
| `templates/index.html` | Home form, tabs, profile sections, numerical + categorical controls. |
| `templates/results.html` | Results gallery + `#detail-slot` + thumb active-state script. |
| `templates/submit_error.html` | Validation / bad-link errors. |
| `templates/partials/image_detail.html` | HTMX fragment for one image’s detail. |
| `static/style.css` | Visual design for cards, tabs, gallery, tables, errors. |
| `environment.yml` | Conda environment `dspt-affinity-ui` (conda-forge packages). |
| `htmx_plan.md` | Technical plan / architecture notes for maintainers. |
| `user_manual.md` | End-user oriented instructions (longer form than this summary). |

---

## User instructions (condensed)

1. **Run the app** (from repository root, with Conda):  
   `conda activate dspt-affinity-ui`  
   `uvicorn app.main:app --reload --app-dir .`  
   Open `http://127.0.0.1:8000/`.

2. **Images tab** — Attach one to five images (allowed types only); unused slots can stay empty.

3. **Customer profile tab**  
   - Fill **Information** (`inf__…`) and **Preference** (`pref__…`) sections.  
   - **Numbers:** must be ≥ 0 (defaults start at 0).  
   - **Dropdowns:** show **labels** from JSON; submitted values are the underlying **`value`** tokens. First option in each list is the default selection.  
   - **Optional local CSV:** choose a one-row UTF-8 file whose headers match `profile_attributes.json` ids **in order**; the browser fills the form from that row. Edit anything before **Predict**; the server uses **only** the form values.

4. **Predict** — Submits images + profile fields; on success you are redirected to the **results** page.

5. **Results page** — Inspect rank order; click a thumbnail to load **attributes + reasoning** below. Use **← New prediction** to start over.

6. **Templates / samples** — Use **“this template”** on the form or `/profile/csv-template` for a fresh CSV shell to load **locally** into the form; use `app/sample_valid_profile.csv` as a known-good example row.

7. **After changing the manifest** — Run `python3 app/scripts/build_profile_attributes.py`, then re-apply any **custom categorical `label`** edits in `profile_attributes.json` if the script overwrote them.

For step-by-step prose and troubleshooting, see **`app/user_manual.md`**.

---

## Integration notes (future work)

- Replace **`AgentModel.predict`** on **`default_agent_model`** (or swap in your own `AgentModel`) in `services/model_service.py` for the real artifact; keep return shape compatible with `routers/web.py` (per-slot filename, score, reason, image-side attributes). `stub_predict` delegates to the same implementation if you still call it from tests.
- If the production model expects a different profile vectorization (e.g. full multihot width), map the submitted `dict[str, str]` profile into that representation inside the model service or a dedicated adapter.
- The results cache is **process-local** and **ephemeral**; it is suitable for demos and single-instance deployment, not durable multi-server state.

---

*Document generated for milestone tracking (`progress_2100`).*
