# Design Document: Agentic Image Classification Pipeline

## Overview

An autonomous pipeline that classifies all images in the ADS-16 dataset into IAB Content Taxonomy Tier 2 categories, producing multi-hot vectors. These vectors feed into the existing `ads16_processor.py` and a downstream Logistic Regression model.

The pipeline processes two image sets:
1. **Ad images** (300) — produces the multi-hot CSV that `ads16_processor.py` already expects
2. **User personal images** (~1,200) — 5 positive + 5 negative per user, aggregated into per-user preference profiles in the same IAB space

## Project Structure

```
├── app/                          # FastAPI web UI (image ranking demo)
│   ├── main.py                   #   App setup
│   ├── routers/web.py            #   HTTP endpoints
│   ├── services/model_service.py #   Canonical model interface (CustomInferenceInterface)
│   ├── services/submission.py    #   Form/CSV parsing, image validation
│   └── templates/                #   Jinja2 + HTMX templates
│
├── agent_model/                  # Runtime inference bridge (AgentModel → AgentCore)
│   ├── agent_model.py            #   Calls deployed ranking agent
│   └── model_service.py          #   Legacy (superseded by app/services/model_service.py)
│
├── src/
│   ├── data_loader/              # Classification pipeline
│   │   ├── agent_processing/     #   Batch invoke + IAB categories
│   │   ├── multihot_from_responses.py
│   │   ├── ads16_processor.py    #   Ratings × multihot → user vectors
│   │   └── profile_builder.py
│   ├── model/                    # Supervised ML (LR, Ridge, kNN)
│   ├── ad_design/                # Visual design feature extraction (20 fields)
│   └── agentcore/                # Deployment utility (deploy/invoke/delete)
│
├── scripts/                      # Deployment scripts
│   ├── deploy_hello_world.py     #   Validates AgentCore setup
│   └── deploy_image_ranking.py   #   Deploys the ranking agent
│
├── config/                       # Environment config
│   ├── agentcore.env             #   AWS region, bucket, role, model ID
│   └── agentcore.env.example
│
├── tests/                        # Integration tests
│   ├── test_deploy.py            #   Deploy utility test
│   ├── test_deploy_image_ranking.py
│   └── test_deploy_feature_extraction.py
│
└── archive/                      # Raw ADS-16 dataset (300 ads × 20 categories)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Inputs                                                     │
│  • 300 ad PNGs (archive/Ads/)                               │
│  • ~1200 personal PNGs (archive/Corpus/U*/IM-POS & IM-NEG)  │
│  • IAB Tier 2 category list (from IAB-t2.csv)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Agent Orchestrator                                         │
│                                                             │
│  1. Image Discovery — scan archive, build manifest          │
│  2. Batch Submission — send images to classification API    │
│  3. Validation — check responses against IAB schema         │
│  4. Assembly — build multi-hot CSVs from results            │
│  5. Aggregation — collapse user images into profiles        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Classification Backend (AgentCore)                         │
│  AgentCore runtime (boto3 invoke_agent_runtime)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Outputs                                                    │
│  • ad_multihot.csv (300 rows × ~30 IAB columns)             │
│  • user_profiles.csv (120 rows × pos/neg/diff profiles)     │
│  • ad_design_features.csv (300 rows × 20 design features)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Downstream                                                 │
│  • ads16_processor.py → user_profile (ratings × multihot)   │
│  • LR model refines profile with B5 + personal image signal │
│  • Image Ranking Agent ranks ads using the profile          │
│  • Web UI (app/) displays ranked results to users           │
└─────────────────────────────────────────────────────────────┘
```

## Full System Architecture (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OFFLINE / TRAINING PIPELINE                                            │
│                                                                         │
│  archive/ images ──► batch_invoke_ads.py ──► multihot_from_responses.py │
│                  ──► ad_design/extract.py ──► ad_design_features.csv    │
│                                                                         │
│  multihot + ratings ──► src/model/ ──► trained LR/Ridge/kNN models      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DEPLOYMENT LAYER (src/agentcore/)                                      │
│                                                                         │
│  deploy.py: deploy_agent() / invoke_agent() / delete_agent()            │
│  scripts/deploy_hello_world.py — validates setup                        │
│  scripts/deploy_image_ranking.py — deploys the ranking agent            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ONLINE / INFERENCE                                                     │
│                                                                         │
│  app/ (FastAPI + HTMX)                                                  │
│    ├── User uploads images + profile CSV                                │
│    ├── Calls agent_model/AgentModel.predict()                           │
│    │     └── invoke_agent_runtime(image_ranking_agent)                  │
│    └── Displays ranked results with reasoning                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Image Discovery

Scans the archive and builds a manifest of all images with canonical IDs:
- Ad images: `Cat{0-19}_{1-15}` (matches `ads16_processor.py` convention)
- User images: `U{XXXX}_{POS|NEG}_{1-5}`

### 2. Classification Client

Sends images to the vision-language model with a prompt containing the IAB category list. Returns a list of matching categories per image.

### 3. Validator

Checks that returned categories are valid IAB category names. Strips invalid entries, normalizes casing. If an image gets zero valid categories, it's retried once with a stricter prompt.

### 4. CSV Assembler

Converts validated category lists into binary multi-hot vectors and writes the output CSVs. Ensures row ordering and ID format match what `ads16_processor.py` expects.

### 5. Profile Aggregator

Groups user image classifications by user and sentiment, then sums:
- `pos_profile` = sum of IAB vectors for 5 positive images
- `neg_profile` = sum of IAB vectors for 5 negative images
- `diff_profile` = pos - neg (optional preference direction signal)

## Implemented Modules (`src/data_loader/`)

### `agent_processing/categories.py`

Canonical IAB Tier 2 category list and prompt template. Single source of truth for both the batch invoker and the post-processor.

- **`load_categories(path)`** — Reads `IAB-t2.csv` (one category per line, CSV-quoted for names containing commas). Returns a deduplicated list in file order.
- **`build_categorization_prompt(instruction, categories)`** — Assembles the full agent prompt: a terse instruction ("Assign IAB tier2 categories… return comma-separated list") followed by the complete category list.
- **`PROMPT_INSTRUCTION`** — Default instruction constant used by the batch invoker.
- **`DEFAULT_CATEGORIES_PATH`** — Points to `IAB-t2.csv` co-located with this module.

### `agent_processing/batch_invoke_ads.py`

Batch invocation of the deployed AgentCore image-classification agent over the ADS-16 ad corpus.

- **`batch_invoke(ads_root, output_path, ...)`** — Walks `ads_root` recursively for `.png` files, base64-encodes each image, and calls `boto3.client("bedrock-agentcore").invoke_agent_runtime(...)` in parallel via `ThreadPoolExecutor` (default 8 workers). Results are appended to a JSONL file one record per image.
- **Resume support** — On restart, already-processed paths (successful records in the JSONL) are skipped automatically.
- **`invoke_one(image_path, ...)`** — Single-image invocation. Builds the JSON payload (`prompt`, `image_base64`, `image_format`), calls the agent runtime, and extracts the `text` field from the response.
- **CLI** — Runnable as `python -m src.data_loader.agent_processing.batch_invoke_ads` with flags for `--ads-root`, `--output`, `--agent-arn`, `--max-workers`, `--limit`, `--no-resume`.

### `multihot_from_responses.py`

Post-processor that converts agent JSONL responses into a per-image multi-hot category matrix CSV.

- **`responses_to_multihot(responses_path, ...)`** — Main entry point. Loads the canonical category list, iterates JSONL records, matches each record's `text` field against canonical names, and returns a DataFrame (rows = images, columns = metadata + one binary column per category) plus a `Counter` of unmatched tokens for QA.
- **Two-pass matching strategy:**
  1. Greedy regex scan for every canonical name (longest first) to catch multi-word names and names containing commas.
  2. Remaining text is comma/newline-split, normalized, and looked up in a case-insensitive dictionary.
- **`assign_categories(text, lookup)`** — Returns `(matched, unmatched)` lists for a single response string.
- **`parse_response_text(text)`** — Tolerant tokenizer that handles bullets, numbered lists, leading headers, and quoted tokens.
- **CLI** — Runnable as `python -m src.data_loader.multihot_from_responses --responses ... --output ...` with optional `--unmatched-report` for diagnostics.
- **Output schema** — Columns: `image_id`, `category`, `image_index`, `path`, `raw_text`, `n_matched`, `n_unmatched`, then one 0/1 column per canonical IAB Tier 2 category. Compatible with `ADS16DataProcessor` when a matching `image_id_for` callable is provided.

### `ads16_processor.py` (existing)

Per-user data processor that combines a user's 300 ad ratings with the multi-hot feature matrix to produce a weighted IAB preference vector (`user_vector = ratings @ multihot_matrix`). Consumes the CSV output of `multihot_from_responses`.

## Ad Design Feature Extraction (`src/ad_design/`)

Extracts structured visual/design features from ad images using an LLM. Produces a 20-field feature vector per ad covering composition, subject, branding, text, messaging, and emotional tone. These features complement the IAB category vectors by capturing *how* an ad is designed, not just *what* it's about.

### Module Structure

```
src/ad_design/
├── schema.py      # Field definitions (FieldDef dataclass), validation
├── prompt.py      # Builds the LLM prompt with rubrics and examples
├── extract.py     # Batch invocation (invoke_one, batch_invoke)
├── parse.py       # JSON extraction from LLM responses, DataFrame assembly
└── validate.py    # Consistency checks (re-invoke subset, measure agreement)
```

### Feature Schema (20 fields)

| Category | Fields |
|----------|--------|
| Visual composition | `design_quality` (enum), `visual_clutter` (1-10), `focal_point_presence` (1-10), `contrast_level` (1-10), `visual_saliency_score` (1-10) |
| Subject | `primary_subject_type` (enum), `human_presence` (bool) |
| Product/brand | `product_visibility` (1-10), `usage_context` (enum), `brand_prominence` (1-10), `logo_present` (bool) |
| Text | `word_count_bin` (enum), `text_density` (1-10), `readability` (1-10) |
| Messaging | `value_proposition_present` (bool), `cta_present` (bool), `offer_present` (bool) |
| Emotion/trust | `emotion_valence` (enum), `perceived_credibility` (enum), `spamminess` (enum) |

### Design Decisions

- Subjective 1-10 scales without objective rubrics are downgraded to 3-bin enums (low/medium/high) for reproducibility.
- `word_count` is binned (0, 1-5, 6-15, 16-40, 40+) because LLMs are unreliable at exact counts.
- Every surviving numeric field gets a 3-anchor rubric in the prompt for inter-call consistency.
- `validate.py` re-invokes a random subset and measures agreement (Cohen's κ for categoricals, Spearman for ints) to detect drift.

### CLI

```bash
# Extract features for all ads
python -m src.ad_design.extract --ads-root archive/ADS16_Benchmark_part1/... --output data/ad_design_responses.jsonl

# Parse responses into a feature CSV
python -m src.ad_design.parse --responses data/ad_design_responses.jsonl --output data/ad_design_features.csv
```

---

## Deployment Utility (`src/agentcore/`)

Reusable library for managing AgentCore runtimes. Wraps the full lifecycle (package → upload → create/update → wait → invoke → delete) behind a simple Python API.

### Module Structure

```
src/agentcore/
├── __init__.py
└── deploy.py      # deploy_agent(), invoke_agent(), delete_agent(), AgentConfig
```

### Public API

| Function | Purpose |
|----------|---------|
| `deploy_agent(name, system_prompt, model_id, ...)` | Create or update an AgentCore runtime. Packages code + deps into a zip, uploads to S3, registers the runtime, waits for READY. Returns `{id, arn, name, status}`. |
| `invoke_agent(runtime_arn, payload, region)` | Send a JSON payload to a deployed runtime. Returns parsed JSON response. |
| `delete_agent(name_or_id)` | Delete a runtime by name or ID. |
| `AgentConfig.from_env(env_path)` | Load config (region, bucket, role, model_id) from `config/agentcore.env`. |

### Agent Code Generation

`generate_agent_code(system_prompt, model_id)` produces a `main.py` that:
1. Creates a `BedrockAgentCoreApp`
2. Initializes a `strands.Agent` with the given system prompt and model
3. Handles both text-only and image+text payloads
4. Returns the agent's response as JSON

Custom handler code can be passed directly via `handler_code` to bypass generation.

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/deploy_hello_world.py` | End-to-end validation of AgentCore setup (create → invoke → delete) |
| `scripts/deploy_image_ranking.py` | Deploys the image ranking agent from `image_ranking_agent_src/` |

> **Full deployment guide:** See #[[file:agentcore-deployment.md]] for IAM permissions, troubleshooting, and API reference.

---

## Runtime Inference Layer (`agent_model/`)

The bridge between the web application and the deployed image ranking agent. Implements the `CustomInferenceInterface` from `app/services/model_service.py` so it can be used as a drop-in replacement for the stub model.

### Module Structure

```
agent_model/
├── agent_model.py    # AgentModel class (the real implementation)
└── model_service.py  # Legacy BaseModel/StubModel (superseded by app/services/model_service.py)
```

### `AgentModel` Class

Subclasses `CustomInferenceInterface` and calls the deployed image ranking agent via `boto3.client("bedrock-agentcore").invoke_agent_runtime(...)`.

**Configuration:**
- ARN resolved from `IMAGE_RANKING_AGENT_ARN` env var or `config/agentcore.env`
- Region from `AWS_REGION` env var (default: `us-east-1`)

**Payload sent to the ranking agent:**

```json
{
  "user_id": "ui_user",
  "profile": {
    "inf": {"gender": "...", "age": "...", "job": "...", "income": "...", "timepass": "...", "fave_sports": "..."},
    "pref": {"websites": "...", "music": "...", "movies": "...", "tv": "...", "books": "..."},
    "pos_labels": [],
    "neg_labels": []
  },
  "images": [
    {"image_id": "filename.png", "image_base64": "...", "image_format": "png"}
  ]
}
```

**Response expected from the ranking agent:**

```json
{
  "classifications": [{"image_id": "...", "category": "...", "confidence": 0.9}],
  "scores": [{"image_id": "...", "score": 0.7, "reasoning": "..."}]
}
```

Scores are normalized from `[-1, 1]` to `[0, 1]` for the UI.

### Integration with the Web App

```
app/routers/web.py
    └── model_service.default_agent_model.predict(images, profile)
            │
            ├── Default: CustomInferenceInterface (stub — hash-based scores)
            └── Production: AgentModel (calls deployed ranking agent)
```

To activate the real model, replace `default_agent_model` at startup:

```python
import app.services.model_service as ms
from agent_model.agent_model import AgentModel
ms.default_agent_model = AgentModel()
```

### Note on Duplicate `model_service.py`

`agent_model/model_service.py` contains an older `BaseModel`/`StubModel` abstraction with a simpler `ImagePrediction(filename, affinity, reason, category)`. This is **superseded** by `app/services/model_service.py` which defines the canonical `CustomInferenceInterface` with `ImagePrediction(slot_index, filename, affinity, reason, image_attributes)`. The `agent_model/model_service.py` file should be considered legacy/dead code.

---

## Web Application (`app/`)

A FastAPI + HTMX application that provides the user-facing interface for the image ranking system.

### Module Structure

```
app/
├── main.py                    # FastAPI app setup, static files, router mount
├── routers/web.py             # All HTTP endpoints (index, submit, results)
├── services/
│   ├── model_service.py       # CustomInferenceInterface + stub (canonical interface)
│   ├── submission.py          # Form/CSV parsing, image validation
│   ├── vocab.py               # Profile attribute vocabulary
│   └── results_cache.py       # In-memory results cache (UUID-keyed)
├── templates/                 # Jinja2 HTML templates (base, index, results)
├── static/style.css           # Stylesheet
├── profile_attributes.json    # Attribute definitions (loaded at startup)
└── user_features_manifest.json
```

### User Flow

1. User visits `/` — sees a form with profile fields (demographics + preferences) and image upload slots (up to 5)
2. User fills profile (or uploads a CSV via template) and attaches ad images
3. POST `/results` — validates inputs, calls `model_service.default_agent_model.predict()`
4. Redirects to `/results/view?rid=<uuid>` — displays images ranked by affinity with reasoning
5. HTMX partial `/results/partials/detail/{rid}/{slot}` — expands per-image detail on click

### Interface Contract

The web app depends on `CustomInferenceInterface.predict()` returning `list[ImagePrediction]` where each prediction has:
- `slot_index` — original upload position
- `filename` — original filename
- `affinity` — float in [0, 1], higher = better match
- `reason` — human-readable explanation
- `image_attributes` — dict of string key-value pairs (displayed in detail view)

Any model (stub, AgentModel, or future implementations) must conform to this interface.

---

## User Interest Modeling (`src/model/`)

The model module trains supervised models that predict per-category user interest from demographic/behavioral features. It consumes the multi-hot ad vectors (from the classification pipeline) and user ratings to learn the mapping from sparse user metadata to rich IAB preference profiles.

**High-level flow:**
1. Build a ground-truth interest matrix from ratings × multi-hot (exposure-corrected)
2. Train per-category models (Logistic Regression, Ridge, kNN) on user features
3. Evaluate via cross-validated ranking metrics (AUC, AP, Spearman)

**Three model families are compared:**
- **Logistic Regression** — binary "does user like category k?", interpretable coefficients
- **Ridge Regression** — continuous interest score prediction
- **kNN** — non-parametric similarity-based prediction (cosine distance)

> **Full specification:** See #[[file:model-module.md]] for detailed API documentation, data flow diagrams, label construction logic, and design decisions.

## Two-Stage Architecture: LR Profile → LLM Ranking

### Why the LR?

Users in this dataset have limited self-reported interest tags (e.g., "Classical Music, Jazz" or "Action, Thriller"). That's too sparse to rank ads against ~30 IAB categories. But each user *did* rate 300 ads, and those ads now have IAB vectors.

The LR learns to predict a user's full IAB preference profile from their limited metadata:

```
Training target (ground truth):
  user_profile = ratings @ multihot_matrix  (what ads16_processor.py computes)
  This is the "true" preference profile derived from actual rating behavior.

Training inputs (what we have for any user):
  • Sparse interest tags from *-PREF.csv
  • B5 personality scores (5 dims)
  • pos_profile / neg_profile from personal images

LR learns:
  sparse_user_info → full IAB preference weights
```

At inference time, you can produce a rich IAB profile from minimal user info — no rating history needed. The LLM then uses this profile to rank ads.

### Stage 1: LR — User Profile Generation (Training/Offline)

```
Inputs:
  • Sparse interest tags (*-PREF.csv)
  • B5 personality scores (5 dims)
  • pos_profile / neg_profile from personal images (~30 dims each)

Training target:
  • user_profile = ratings @ multihot_matrix (~30 dims)
    The "true" IAB preference weights from actual ad ratings

Output at inference:
  • Predicted user_profile for users with limited/no rating history
```

### Stage 2: Image Ranking Agent — Ad Ranking (Inference/Online)

Deployed as an AgentCore runtime (`image_ranking_agent`). Invoked via `agent_model/AgentModel` from the web app.

```
Agent Input:
  • user profile (demographics + preferences from form/CSV)
    e.g. {"inf": {"gender": "M", "age": "25-34", ...}, "pref": {"music": "Jazz", ...}}
  • candidate ad images (base64-encoded PNGs)
  • (internally) IAB category knowledge from system prompt

Agent Output:
  • per-image classification (IAB category + confidence)
  • per-image affinity score [-1, 1] with reasoning
  • ranked order by predicted user interest
```

### Why this split?

- **LR** is good at: learning stable preference weights from structured data, generalizing from sparse user info to full profiles
- **Ranking Agent** is good at: reasoning over profiles + visual content, handling new/unseen ads, explaining recommendations, combining IAB classification with user matching in a single call
- Together: the LR provides a grounded, data-driven profile from minimal input; the agent uses it (plus visual analysis) for flexible ranking

## Correctness Properties

1. **Coverage**: Output contains one row per image (300 ads, ~1200 user images), no duplicates
2. **Binary encoding**: All feature columns contain only 0 or 1
3. **Taxonomy conformance**: All column headers are valid IAB Tier 2 category names
4. **ID consistency**: Image IDs match `ads16_processor.py`'s `image_id_for` convention
5. **Non-empty classification**: Every classified image has at least one category = 1
6. **Idempotency**: Same images + same model → same output (temperature = 0)

## Dependencies

- Python ≥ 3.10
- pandas, numpy (data handling)
- scikit-learn (model training: LogisticRegression, Ridge, KNeighborsRegressor)
- scipy (spearmanr correlation in model evaluation)
- boto3 (AgentCore runtime invocation + S3 upload)
- strands-agents (agent framework, bundled in deployed runtimes)
- bedrock-agentcore (agent runtime SDK, bundled in deployed runtimes)
- FastAPI + Jinja2 + python-multipart (web application)
- pydantic (validation, optional)

## Data Flow (Implemented)

```
IAB-t2.csv
    │
    ▼
categories.py ──► build_categorization_prompt()
                        │
                        ▼
batch_invoke_ads.py ──► ads16_agent_responses.jsonl
                              │
                              ▼
multihot_from_responses.py ──► ads16_multihot.csv
                                    │
                                    ▼
ads16_processor.py ──► user_vector (ratings @ multihot)
                              │
                              ▼
                    src/model/ (LR, Ridge, kNN training)
```

```
ad_design/schema.py + prompt.py
    │
    ▼
ad_design/extract.py ──► ad_design_responses.jsonl
                              │
                              ▼
ad_design/parse.py ──► ad_design_features.csv
```

```
src/agentcore/deploy.py
    │
    ├── scripts/deploy_hello_world.py ──► validates setup
    └── scripts/deploy_image_ranking.py ──► deploys ranking agent
                                                │
                                                ▼
                                    IMAGE_RANKING_AGENT_ARN
                                                │
                                                ▼
agent_model/AgentModel ◄── app/routers/web.py (user request)
    │
    ▼
invoke_agent_runtime() ──► ranked predictions ──► results page
```

## Open Questions

- ~~Exact IAB category list to use~~ → Resolved: IAB Tier 2 via `IAB-t2.csv`
- ~~API interface details for Tom & Kevin's facility~~ → Resolved: AgentCore runtime via `boto3.invoke_agent_runtime`, ARN-based addressing
- ~~Batch size and concurrency~~ → Resolved: ThreadPoolExecutor with 8 workers, resumable JSONL output
- ~~Batch classification script~~ → Resolved: `batch_invoke_ads.py` implemented and working
- Whether to include `diff_profile` or keep it optional
- Whether `agent_model/model_service.py` should be deleted (it's superseded by `app/services/model_service.py`)
- How to integrate ad_design features into the ranking agent's decision-making (currently extracted but not consumed by the ranking agent)
- Whether the ranking agent should receive pre-computed IAB vectors or classify on-the-fly (currently does on-the-fly classification)
