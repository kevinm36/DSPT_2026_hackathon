# Model constructor — where to plug in a trained model

This document lists the **exact call site**, **recommended implementation module**, and the **input / output variables** the web app expects today. Replace the stub with your trained pipeline while keeping the same outward contract so `routers/web.py` and the results UI stay unchanged.

---

## 1. Where the model is invoked (call site)

**File:** `app/routers/web.py`  
**Handler:** `submit_results` (inside `POST /results`), immediately after a successful `collect_submission`.

**Call today:**

```89:91:app/routers/web.py
    image_rows, profile = outcome
    tuples = [(r["filename"], r["raw"]) for r in image_rows]
    predictions = model_service.default_agent_model.predict(tuples, profile)
```

**What to change:** keep the two lines that build `tuples` and `profile`; implement your pipeline on **`CustomInferenceInterface.predict`** (or replace **`default_agent_model`** with your own `CustomInferenceInterface` subclass / instance). The thin wrapper **`stub_predict`** still delegates to **`default_agent_model.predict`** if you prefer to keep a function-style entry point.

The loop below **must** still receive something iterable like the current `predictions` list, with the same per-element fields described in section 4.

```93:106:app/routers/web.py
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
```

- **`rank`** is assigned here in **descending score order** as produced by your model function (the stub sorts by `affinity` before return; your implementation should return rows already sorted by score descending, or sort before returning).

---

## 2. Where to implement loading + inference (recommended)

**Primary module:** `app/services/model_service.py`

| Location | Responsibility |
|----------|----------------|
| Top of file / new helpers | `joblib.load`, `torch.load`, `onnxruntime`, path constants, feature preprocessing. |
| **`CustomInferenceInterface.predict`** (or a subclass) | Map `(image_payloads, profile)` → `list[ImagePrediction]`. |
| **`default_agent_model`** | Module-level instance used by `web.py`; replace or rebind after loading a real artifact. |
| `ImagePrediction` dataclass | Extend only if you also update the mapping in `web.py` (section 4). |

Current stub and output type definition (abbreviated):

```7:91:app/services/model_service.py
@dataclass(frozen=True)
class ImagePrediction:
    slot_index: int
    filename: str
    affinity: float
    reason: str
    image_attributes: dict[str, str]


class CustomInferenceInterface:
    def predict(
        self,
        image_payloads: list[tuple[str, bytes]],
        profile: dict[str, str],
    ) -> list[ImagePrediction]:
        ...
        scored.sort(key=lambda p: p.affinity, reverse=True)
        return scored


default_agent_model = CustomInferenceInterface()


def stub_predict(
    image_payloads: list[tuple[str, bytes]],
    profile: dict[str, str],
) -> list[ImagePrediction]:
    return default_agent_model.predict(image_payloads, profile)
```

**Optional:** load the artifact once in **`app/main.py`** using a FastAPI `lifespan` context manager, attach the loaded object to `app.state`, and assign **`model_service.default_agent_model`** (or inject your `CustomInferenceInterface` into the router—avoid circular imports).

---

## 3. Input variables (what the router passes into the model)

### 3.1 `tuples` — images

**Variable name in router:** `tuples`  
**Python type:** `list[tuple[str, bytes]]`  
**Construction:**

```python
tuples = [(r["filename"], r["raw"]) for r in image_rows]
```

| Component | Meaning |
|-----------|--------|
| `str` (first element of each tuple) | Original upload filename (for display and `ImagePrediction.filename`). |
| `bytes` (second element) | Raw file bytes of the image (decode / tensorize / save to temp as your model requires). |

**Order:** index `0` is the first uploaded image slot, `1` the second, etc. **`slot_index` in the output must refer back to this same index** so `image_rows[pred.slot_index]` resolves to the correct thumbnail bytes.

**Source chain:** `image_rows` comes from `submission.collect_submission` → each dict has keys `filename`, `raw`, `content_type`.

### 3.2 `profile` — customer profile

**Variable name in router:** `profile`  
**Python type:** `dict[str, str]`  
**Construction:** returned by `validate_profile` inside `collect_submission`; keys and string values match **`app/profile_attributes.json`** after validation:

- **Numerical** attributes: string representation of a finite, **non-negative** number (e.g. `"42"`, `"0.5"`).
- **Categorical** attributes: string equal to the chosen option **`value`** (full multihot token), not the human `label`.

Invalid profiles (e.g. blank numerical fields, categorical **`invalid`** sentinel) are rejected in **`collect_submission`** before **`predict`** runs; the model always receives a fully valid `profile` dict.

Your model should map this dict into whatever feature vector or tensor your training pipeline expects (e.g. one-hot / multihot alignment with `user_features_manifest.json`).

---

## 4. Output variables (what the router expects from the model)

**`CustomInferenceInterface.predict`** must return **`list[ImagePrediction]`** (or an iterable of objects with the **same attributes**), one entry **per uploaded image** (same length as `tuples` before any sorting).

### 4.1 `ImagePrediction` fields

| Field | Type | Used as |
|-------|------|--------|
| `slot_index` | `int` | Index into `image_rows` / `tuples` for thumbnails and cache slot key. |
| `filename` | `str` | Display name; usually mirror the input filename for that slot. |
| `affinity` | `float` | **`score`** in the results cache; higher = better rank. |
| `reason` | `str` | Shown as prediction reasoning in `partials/image_detail.html` (plain text). |
| `image_attributes` | `dict[str, str]` | Shown as “model output” attributes for that image in the detail partial (string key-value rows). |

### 4.2 Cached payload per item (`items` list)

Each dict stored in `results_cache` and read by `results.html` / detail partials:

| Key | Source |
|-----|--------|
| `slot` | `pred.slot_index` |
| `rank` | 1-based position after sorting by descending `affinity` |
| `filename` | `pred.filename` |
| `score` | `pred.affinity` |
| `thumbnail_src` | Built in router from `image_rows[pred.slot_index]` (not from model). |
| `reason` | `pred.reason` |
| `image_attributes` | `pred.image_attributes` |

Do **not** remove `slot` / `score` / `reason` / `image_attributes` without updating templates and `results_detail_partial` in `web.py`.

---

## 5. Downstream consumers (read-only for model integration)

These files **consume** the cached structure; they do not call the model:

- `app/templates/results.html` — `item.rank`, `item.filename`, `item.score`, `item.thumbnail_src`, HTMX URL uses `item.slot`.
- `app/templates/partials/image_detail.html` — `filename`, `reason`, `image_attributes`.
- `app/routers/web.py` — `results_detail_partial` reads `data["items"]` by `slot`.

---

## 6. Minimal integration checklist

1. Implement inference on **`CustomInferenceInterface.predict`** in **`app/services/model_service.py`** (or replace **`default_agent_model`**); the call site in **`app/routers/web.py`** is **`default_agent_model.predict(tuples, profile)`** (around the `image_rows, profile = outcome` block).
2. Accept **`image_payloads: list[tuple[str, bytes]]`** and **`profile: dict[str, str]`**.
3. Return **`list[ImagePrediction]`** with **`slot_index`** aligned to input order, **`affinity`** sortable descending, and string **`reason`** / **`image_attributes`** for the UI.
4. Keep **`web.py`** mapping loop (section 1) unchanged unless you intentionally extend the UI contract.

---

*For product-level behavior and CSV rules, see `app/user_manual.md` and `app/progress_2100.md`.*
