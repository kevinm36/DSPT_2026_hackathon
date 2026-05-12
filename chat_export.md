# Chat Export — ADS-16 Project Discussion

## 1. Archive Summary

The `archive/` folder contains the **ADS-16 Benchmark Dataset** — a research benchmark for studying personality in computational advertising, published by Roffo & Vinciarelli at the EMPIRE 2016 workshop (ACM RecSys).

The archive is split into two parts (`ADS16_Benchmark_part1` and `ADS16_Benchmark_part2`), each containing the same three sections:

### Structure

- **Ads/** — 300 real advertisements organized into 20 product/service categories (folders `1` through `20`), each containing ~15 PNG images.

- **Corpus/** — Data from 120 participants (U0001–U0120), split across the two parts (part1 has U0001–U0060, part2 has the rest). Each participant folder includes:
  - `*-B5.csv` — Big Five personality scores (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
  - `*-PREF.csv` — Ad preference ratings (1–5 stars)
  - `*-RT.csv` — Response times / ratings
  - `*-INF.csv` — Demographic/info data
  - `*-IM-POS/` and `*-IM-NEG/` — Personal images labeled as positive/negative (with corresponding CSVs)

- **Documents/** — The reference paper PDF ("Personality in Computational Advertising: A Benchmark")

### Purpose

The dataset links users' Big Five personality traits to their ad preferences, enabling research on personality-aware ad recommendation systems. It's restricted to non-commercial scientific use only.

---

## 2. Agentic Image Classification Pipeline

### What it does

The goal is to build a feature vector for each ad image that captures *what the ad is about* in a structured, machine-readable way. This vector then gets combined with user preference ratings to form the label/feature input for a Logistic Regression model that predicts ad relevance based on personality.

### The pipeline, step by step

**1. Image → IAB Category Classification (multi-hot)**

Each of the ~300 ad images gets processed through a vision/classification model to produce a multi-hot vector over a fixed label space. Using the IAB Content Taxonomy (~700+ categories across tiers), you pick a tier level (likely Tier 1 or Tier 2 for tractability) and classify each image into one or more relevant categories.

Example for a Nike running shoe ad:
```
IAB Tier 1 categories (simplified):
[Sports, Shopping, Health & Fitness, Style & Fashion, ...]

Multi-hot vector:
[1, 1, 1, 1, 0, 0, 0, 0, ...]  
 ^Sports ^Shopping ^Health ^Fashion
```

**2. Why IAB as the label space**

- Industry standard for ad categorization — already used in programmatic advertising
- Hierarchical structure (Tier 1 → Tier 2 → Tier 3) lets you tune granularity
- Fixed and well-defined, so vectors are consistent across all images
- Aligns with how ad targeting already works in production systems

**3. The "agentic" part — Tom Overman & Kevin Mueller's facility**

This refers to using a batch image processing capability (likely a vision-language model or multimodal classifier) that can:
- Ingest an image
- Be prompted with the fixed IAB label space
- Return which categories apply (multi-hot)

The "agentic function" means this runs autonomously across all 300 images — an agent orchestrates the batch, handles retries, validates outputs, and assembles the final matrix.

**4. Combining with user ratings → LR input**

Once you have the multi-hot vectors, the data assembly looks like:

```
For each user u and ad a:
  x = [IAB_vector(a), personality_B5(u)]   # features
  y = preference_rating(u, a)              # label (or binarized: liked/not liked)
```

### Practical considerations

| Decision | Options |
|----------|---------|
| IAB tier level | Tier 1 (~30 cats) vs Tier 2 (~400 cats) — start with Tier 1 for sparsity reasons |
| Classification model | GPT-4o / Claude vision / CLIP-based classifier |
| Threshold | Confidence cutoff for "hot" (e.g., top-k or > 0.5 probability) |
| Validation | Spot-check a sample of images against human labels |
| Output format | `{ad_id: str, iab_vector: List[int], categories: List[str]}` |

### Sketch of the output

```python
# Per-image output
{
  "ad_id": "cat3/img7",
  "iab_tier1_vector": [0,0,1,0,0,1,0,0,0,0,1,0,...],  # len = num_tier1_categories
  "matched_categories": ["Food & Drink", "Health & Fitness", "Shopping"]
}

# Combined with user data for LR
# X matrix: [n_users * n_ads, n_iab_categories + 5_personality_dims]
# y vector: [n_users * n_ads] preference ratings
```

### What's needed to execute

1. A fixed IAB category list (decide on tier level)
2. Access to a vision model that can batch-classify images against that list
3. An orchestration layer (the "agent") that processes all 300 images, handles the prompt template, collects results
4. A join step that merges the resulting vectors with the Corpus user data (B5 scores + PREF ratings)

---

## 3. Personality Scores — User-Level, Not Ad-Level

The **Big Five personality scores are unique to the user, not the advertisement.**

- **`U0001-B5.csv`** contains 10 questionnaire answers (the TIPI — Ten-Item Personality Inventory) that map to the Big Five traits (O-C-E-A-N). It's a single, fixed personality profile per person.

- **`U0001-RT.csv`** contains the actual **per-ad ratings** — each of the 20 product categories has 15 comma-separated scores (1–5 stars), one per ad image. So that's 300 ratings per user.

### Summary for the LR model:

| Data | Scope | Source file |
|------|-------|-------------|
| Big Five personality | 1 per user (static trait) | `*-B5.csv` |
| Ad ratings (1–5 stars) | 1 per user × per ad (300 total) | `*-RT.csv` |
| IAB multi-hot vector | 1 per ad image (from classification) | To be generated |

The personality vector gets **repeated/broadcast** across all ads for that user. The model learns: given this person's personality + this ad's content categories → predicted preference rating.

---

## 4. `ads16_processor.py` — Core Data Processing Module

Located at `src/data_loader/ads16_processor.py`, written by Cher Wang (`@chwang2`).

### What it does

Takes two inputs for a given user and produces a single **weighted profile vector**:

```
user_vector = Σ (rating_i × multihot_i)   for all 300 ads
```

It multiplies each ad's multi-hot feature vector by how much the user liked that ad, then sums everything up into one vector that represents the user's *content-weighted preference profile*.

### Inputs

1. **Rating CSV** (`U0001-RT.csv` from the Corpus) — the 300 star ratings (20 categories × 15 images).

2. **Multi-hot feature CSV** — the output of the agentic image classification pipeline (processing each ad image into IAB categories). Each row is one image, each column is a binary feature, values are 0 or 1.

### How it links to the project

```
┌─────────────────────────────────────────────────────────┐
│  Agentic Pipeline (Tom/Kevin's facility)                │
│  300 ad images → vision model → IAB multi-hot CSV       │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  ads16_processor.py                                      │
│  Per user: ratings (300,) × multi-hot (300, n_features)  │
│  → user_vector (n_features,)                             │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  LR Model                                                │
│  X = user_vectors (or combined with B5 personality)      │
│  y = some target (e.g., ad preference prediction)        │
└──────────────────────────────────────────────────────────┘
```

### Key design details

- **`ADS16DataProcessor`** class handles parsing the quirky CSV format (semicolons, comma-separated ratings packed into cells).
- **`image_id_for`** is a pluggable function that maps `(category_index, image_index)` → image ID string, so it can match whatever naming scheme the multi-hot CSV uses.
- **`UserProfile`** dataclass bundles the output: user ID, raw ratings, image IDs, feature column names, and the final weighted vector.
- The math is vectorized via NumPy: `ratings @ multihot_matrix` — a single matrix multiply.

### What's still missing

The **multi-hot CSV** (the `multihot_csv_path` input) is the piece that the agentic classification pipeline needs to generate. Once that file exists with IAB category columns and one row per ad image, this processor can immediately consume it and produce user vectors for all 120 participants.
