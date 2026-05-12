# Design Document: Agentic Image Classification Pipeline

## Overview

An autonomous pipeline that classifies all images in the ADS-16 dataset into IAB Content Taxonomy Tier 2 categories, producing multi-hot vectors. These vectors feed into the existing `ads16_processor.py` and a downstream Logistic Regression model.

The pipeline processes two image sets:
1. **Ad images** (300) — produces the multi-hot CSV that `ads16_processor.py` already expects
2. **User personal images** (~1,200) — 5 positive + 5 negative per user, aggregated into per-user preference profiles in the same IAB space

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
│  Classification Backend (Tom & Kevin's Facility)            │
│  AgentCore runtime (boto3 invoke_agent_runtime)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Outputs                                                    │
│  • ad_multihot.csv (300 rows × ~30 IAB columns)             │
│  • user_profiles.csv (120 rows × pos/neg/diff profiles)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Downstream                                                 │
│  • ads16_processor.py → user_profile (ratings × multihot)   │
│  • LR model refines profile with B5 + personal image signal │
│  • LLM ranks candidate ads using the profile                │
└─────────────────────────────────────────────────────────────┘
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

### Stage 2: LLM — Ad Ranking (Inference/Online)

```
LLM Input:
  • user_profile (IAB scores from Stage 1)
    e.g. "Sports: 4.2, Food & Drink: 3.8, Technology: 1.1, ..."
  • candidate_ads (list of ads with their IAB vectors)
    e.g. [{id: "ad_42", categories: ["Sports", "Health & Fitness"]}, ...]

LLM Output:
  • ranked list of ad IDs by predicted user interest
  • optional: reasoning for ranking decisions
```

### Why this split?

- **LR** is good at: learning stable preference weights from structured data, generalizing from sparse user info to full profiles
- **LLM** is good at: reasoning over profiles, handling new/unseen ads, explaining recommendations
- Together: the LR provides a grounded, data-driven profile from minimal input; the LLM uses it for flexible ranking

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
- boto3 (AgentCore runtime invocation)
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
```

## Open Questions

- ~~Exact IAB category list to use~~ → Resolved: IAB Tier 2 via `IAB-t2.csv`
- ~~API interface details for Tom & Kevin's facility~~ → Resolved: AgentCore runtime via `boto3.invoke_agent_runtime`, ARN-based addressing
- Whether to include `diff_profile` or keep it optional
- ~~Batch size and concurrency~~ → Resolved: ThreadPoolExecutor with 8 workers, resumable JSONL output
