# Design Document: Agentic Image Classification Pipeline

## Overview

An autonomous pipeline that classifies all images in the ADS-16 dataset into IAB Content Taxonomy Tier 1 categories, producing multi-hot vectors. These vectors feed into the existing `ads16_processor.py` and a downstream Logistic Regression model.

The pipeline processes two image sets:
1. **Ad images** (300) — produces the multi-hot CSV that `ads16_processor.py` already expects
2. **User personal images** (~1,200) — 5 positive + 5 negative per user, aggregated into per-user preference profiles in the same IAB space

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Inputs                                                     │
│  • 300 ad PNGs (archive/Ads/)                               │
│  • ~1200 personal PNGs (archive/Corpus/U*/IM-POS & IM-NEG)  │
│  • IAB Tier 1 category list (~30 categories)                │
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
│  Vision-language model, prompted with IAB categories        │
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

Sends images to the vision-language model with a prompt containing the IAB Tier 1 category list. Returns a list of matching categories per image.

### 3. Validator

Checks that returned categories are valid IAB Tier 1 names. Strips invalid entries, normalizes casing. If an image gets zero valid categories, it's retried once with a stricter prompt.

### 4. CSV Assembler

Converts validated category lists into binary multi-hot vectors and writes the output CSVs. Ensures row ordering and ID format match what `ads16_processor.py` expects.

### 5. Profile Aggregator

Groups user image classifications by user and sentiment, then sums:
- `pos_profile` = sum of IAB vectors for 5 positive images
- `neg_profile` = sum of IAB vectors for 5 negative images
- `diff_profile` = pos - neg (optional preference direction signal)

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
3. **Taxonomy conformance**: All column headers are valid IAB Tier 1 category names
4. **ID consistency**: Image IDs match `ads16_processor.py`'s `image_id_for` convention
5. **Non-empty classification**: Every classified image has at least one category = 1
6. **Idempotency**: Same images + same model → same output (temperature = 0)

## Dependencies

- Python ≥ 3.10
- pandas, numpy (data handling)
- Pillow (image loading)
- httpx (API client)
- pydantic (validation)

## Open Questions

- Exact IAB Tier 1 list to use (see `iab-taxonomy-reference.md`)
- API interface details for Tom & Kevin's facility (auth, rate limits, payload format)
- Whether to include `diff_profile` or keep it optional
- Batch size and concurrency (see `pipeline-configuration.md` for tuning guidance)
