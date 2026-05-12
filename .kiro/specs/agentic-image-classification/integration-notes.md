# Integration Notes

## Overview

This document describes how the agentic image classification pipeline connects to the existing codebase and the downstream LR model. It covers data flow, interface contracts, and the end-to-end feature assembly process.

---

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Agentic Classification Pipeline                           │
│                                                                     │
│  archive/Ads/  ──→  Vision Model  ──→  ad_multihot.csv (300×30)    │
│  archive/Corpus/U*/IM-POS/  ──→  Vision Model  ──→                 │
│  archive/Corpus/U*/IM-NEG/  ──→  Vision Model  ──→                 │
│                                      user_image_classifications.csv │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Profile Aggregation                                       │
│                                                                     │
│  user_image_classifications.csv  ──→  Profile Aggregator            │
│                                       ──→  user_profiles.csv        │
│                                            (120 × 90 dims)          │
│                                            pos_*, neg_*, diff_*     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: ads16_processor.py (existing)                             │
│                                                                     │
│  U{XXXX}-RT.csv  +  ad_multihot.csv                                │
│       ──→  user_vector = ratings @ multihot_matrix                  │
│       ──→  shape: (30,) per user — content-weighted ad preference   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: LR Feature Assembly                                       │
│                                                                     │
│  For each (user, ad) pair:                                          │
│    x = concat [                                                     │
│      IAB_vector(ad),          # 30 dims — what the ad is about      │
│      B5_personality(user),    # 5 dims  — who the user is           │
│      pos_profile(user),       # 30 dims — what user likes           │
│      neg_profile(user),       # 30 dims — what user dislikes        │
│      diff_profile(user),      # 30 dims — preference direction      │
│    ]                                                                │
│    y = rating(user, ad)       # 1-5 star preference                 │
│                                                                     │
│  Total feature dim: ~125 (or ~95 without diff_profile)              │
│  Total samples: 120 users × 300 ads = 36,000                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Integration with `ads16_processor.py`

### Current Interface

The existing `ADS16DataProcessor` expects:

```python
processor = ADS16DataProcessor(
    rating_csv_path="archive/.../U0001/U0001-RT.csv",
    multihot_csv_path="data/output/ad_multihot.csv",  # ← our output
    image_id_column="image_id",
    image_id_for=lambda c, i: f"Cat{c}_{i + 1}",     # ← must match our IDs
)
profile = processor.process()
# profile.user_vector shape: (n_iab_categories,)
```

### Contract Requirements

For seamless integration, our `ad_multihot.csv` must satisfy:

1. **Column `image_id`** exists and contains values matching `Cat{c}_{i+1}` format
2. **All 300 image IDs** are present (Cat0_1 through Cat19_15)
3. **No duplicate IDs**
4. **Feature columns** contain only 0 and 1 (integer)
5. **Delimiter** is comma (`,`)
6. **No extra whitespace** in column headers

### Verification Script

```python
"""Quick integration check — run after pipeline completes."""
from src.data_loader.ads16_processor import ADS16DataProcessor

# Test with one user
processor = ADS16DataProcessor(
    rating_csv_path="archive/ADS16_Benchmark_part1/ADS16_Benchmark_part1/Corpus/Corpus/U0001/U0001-RT.csv",
    multihot_csv_path="data/output/ad_multihot.csv",
)
profile = processor.process()

assert profile.user_vector.shape[0] > 0, "Empty user vector"
assert profile.ratings.shape == (300,), f"Expected 300 ratings, got {profile.ratings.shape}"
print(f"✓ Integration check passed. Vector dim: {profile.user_vector.shape[0]}")
```

---

## Integration with User Profile Vectors

### New Component: Profile Loader

A new module is needed to load `user_profiles.csv` and join it with the existing user data:

```python
"""Loads aggregated user preference profiles from personal image classifications."""

import pandas as pd
import numpy as np

def load_user_profiles(profiles_path: str = "data/output/user_profiles.csv") -> pd.DataFrame:
    """Load user IAB preference profiles.
    
    Returns DataFrame indexed by user_id with columns:
      pos_<category>, neg_<category>, diff_<category> for each IAB category.
    """
    df = pd.read_csv(profiles_path)
    return df.set_index("user_id")
```

### Feature Assembly for LR

```python
"""Assembles the full feature matrix for the Logistic Regression model."""

import numpy as np
import pandas as pd
from src.data_loader.ads16_processor import ADS16DataProcessor

def build_lr_dataset(
    multihot_path: str,
    user_profiles_path: str,
    corpus_root: str,
    user_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build X (features) and y (ratings) for all user-ad pairs.
    
    Returns:
        X: shape (n_users * 300, feature_dim)
        y: shape (n_users * 300,)
    """
    # Load user preference profiles
    profiles_df = pd.read_csv(user_profiles_path).set_index("user_id")
    
    # Load B5 personality scores (separate loader needed)
    # b5_df = load_b5_scores(corpus_root, user_ids)
    
    X_rows = []
    y_rows = []
    
    for user_id in user_ids:
        # Get user's ad preference vector via existing processor
        processor = ADS16DataProcessor(
            rating_csv_path=f"{corpus_root}/{user_id}/{user_id}-RT.csv",
            multihot_csv_path=multihot_path,
        )
        profile = processor.process(user_id=user_id)
        
        # Load the multihot matrix directly for per-ad features
        multihot_df = pd.read_csv(multihot_path).set_index("image_id")
        
        # User-level features (repeated for each ad)
        user_profile = profiles_df.loc[user_id].values  # pos/neg/diff profiles
        # b5_scores = b5_df.loc[user_id].values           # Big Five (5 dims)
        
        for idx, image_id in enumerate(profile.image_ids):
            ad_vector = multihot_df.loc[image_id].values  # IAB vector for this ad
            
            # Concatenate features
            x = np.concatenate([
                ad_vector,        # what the ad is about
                # b5_scores,      # who the user is
                user_profile,     # what user likes/dislikes (pos + neg + diff)
            ])
            X_rows.append(x)
            y_rows.append(profile.ratings[idx])
    
    return np.array(X_rows), np.array(y_rows)
```

---

## Feature Interaction Options

Beyond simple concatenation, the LR model can benefit from interaction features that capture "does this ad match what the user likes?":

### Option A: Element-wise Product (Recommended for LR)

```python
# Match signal: overlap between ad content and user positive preferences
match_pos = ad_vector * pos_profile   # element-wise, shape (30,)
match_neg = ad_vector * neg_profile   # element-wise, shape (30,)

x = np.concatenate([ad_vector, b5_scores, pos_profile, neg_profile, match_pos, match_neg])
# Total dims: 30 + 5 + 30 + 30 + 30 + 30 = 155
```

### Option B: Dot Product Scalars

```python
# Single scalar features capturing overall alignment
match_score = np.dot(ad_vector, pos_profile)   # scalar
avoid_score = np.dot(ad_vector, neg_profile)   # scalar

x = np.concatenate([ad_vector, b5_scores, [match_score, avoid_score]])
# Total dims: 30 + 5 + 2 = 37 (much more compact)
```

### Option C: Difference-based

```python
# How much does this ad align with user's net preference direction?
alignment = np.dot(ad_vector, diff_profile)  # scalar

x = np.concatenate([ad_vector, b5_scores, [alignment]])
# Total dims: 30 + 5 + 1 = 36
```

---

## File Dependencies Graph

```
config/pipeline_config.yaml
config/iab_tier1_categories.json
    │
    ▼
[Pipeline Execution]
    │
    ├──→ data/output/ad_multihot.csv
    │        │
    │        └──→ src/data_loader/ads16_processor.py (existing)
    │                  │
    │                  └──→ user_vector per user
    │
    ├──→ data/output/user_image_classifications.csv
    │        │
    │        └──→ [Profile Aggregator]
    │                  │
    │                  └──→ data/output/user_profiles.csv
    │
    ├──→ data/output/assembly_report.json
    │
    └──→ data/output/.pipeline_state.json (checkpoint)

[LR Feature Assembly]
    ├── reads: ad_multihot.csv
    ├── reads: user_profiles.csv
    ├── reads: U*-RT.csv (ratings)
    ├── reads: U*-B5.csv (personality)
    └── produces: X matrix, y vector for sklearn LogisticRegression
```

---

## Compatibility Notes

### `ads16_processor.py` Assumptions

The existing processor makes these assumptions that our pipeline must respect:

1. **`image_id_for` default**: `lambda c, i: f"Cat{c}_{i + 1}"` — our CSV must use this exact format
2. **Rating CSV has 3 rows**: header, category names, ratings — we don't modify this
3. **20 categories, 15 images each** — our manifest must find exactly these
4. **Multi-hot CSV uses comma delimiter** — matches our output config
5. **`image_id` column name** — configurable but defaults to "image_id"

### Breaking Changes to Watch For

If `ads16_processor.py` is modified in the future:
- Changes to `image_id_for` convention → update Image Discovery component
- Changes to `image_id_column` name → update CSV Assembler config
- Changes to expected CSV delimiter → update output config
- Addition of new required columns → update CSV Assembler schema

### Version Pinning

The pipeline output should include a metadata header or sidecar file indicating:
- Pipeline version
- Taxonomy version (IAB Tier 1 v1)
- Model used for classification
- Timestamp of generation

This enables downstream consumers to detect stale or incompatible data.
