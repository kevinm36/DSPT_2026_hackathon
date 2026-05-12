# Data Dictionary

## Overview

This document defines the schema for all input, intermediate, and output data files used in the agentic image classification pipeline.

---

## Input Files

### 1. Ad Images

| Property | Value |
|----------|-------|
| Location | `archive/ADS16_Benchmark_part{1,2}/ADS16_Benchmark_part{1,2}/Ads/Ads/{1..20}/{1..15}.png` |
| Format | PNG |
| Count | 300 (20 categories × 15 images) |
| Naming | `{image_number}.png` (1-based within each category folder) |
| Folder structure | Category folders are 1-indexed (1..20), mapped to 0-indexed (0..19) internally |

### 2. User Rating Files (`*-RT.csv`)

| Property | Value |
|----------|-------|
| Location | `archive/.../Corpus/Corpus/U{0001..0120}/U{XXXX}-RT.csv` |
| Delimiter | `;` (semicolon) |
| Encoding | UTF-8 with quoted fields |
| Rows | 3 (header + category names + ratings) |
| Columns | 20 (Cat0..Cat19) |

**Row structure**:
```
Row 0: "Cat0";"Cat1";...;"Cat19"                    (column headers)
Row 1: "Clothing & Shoes";"Automotive";...          (human-readable names)
Row 2: "1,1,1,3,1,...";"2,1,1,..."                  (15 comma-separated ratings per cell)
```

**Rating values**: Integer in [0, 5] where:
- 0 = not rated / skipped
- 1 = lowest interest
- 5 = highest interest

### 3. User Personality Files (`*-B5.csv`)

| Property | Value |
|----------|-------|
| Location | `archive/.../Corpus/Corpus/U{XXXX}/U{XXXX}-B5.csv` |
| Delimiter | `;` |
| Rows | 11 (1 header + 10 questions) |
| Columns | 2 (`Question#`, `Answer`) |

**Answer values**: Integer in [-2, 2] (TIPI scale: disagree strongly to agree strongly)

**TIPI → Big Five mapping**:
| Questions | Trait | Scoring |
|-----------|-------|---------|
| 1, 6 | Extraversion | Q1 forward, Q6 reverse |
| 2, 7 | Agreeableness | Q2 reverse, Q7 forward |
| 3, 8 | Conscientiousness | Q3 forward, Q8 reverse |
| 4, 9 | Emotional Stability | Q4 reverse, Q9 forward |
| 5, 10 | Openness | Q5 forward, Q10 reverse |

### 4. User Positive Images (`*-IM-POS/`)

| Property | Value |
|----------|-------|
| Location | `archive/.../Corpus/Corpus/U{XXXX}/U{XXXX}-IM-POS/{1..5}.png` |
| Format | PNG |
| Count | 5 per user (600 total across 120 users) |
| Metadata | `U{XXXX}-IM-POS.csv` |

**Metadata CSV structure** (`*-IM-POS.csv`):
```
Row 0: "fave1";"fave2";"fave3";"fave4";"fave5"       (column headers)
Row 1: "U0001-IM-POS/1.png";...                       (file paths)
Row 2: "my cats";"movie we are in";...                 (user-provided text labels)
```

### 5. User Negative Images (`*-IM-NEG/`)

| Property | Value |
|----------|-------|
| Location | `archive/.../Corpus/Corpus/U{XXXX}/U{XXXX}-IM-NEG/{1..5}.png` |
| Format | PNG |
| Count | 5 per user (600 total across 120 users) |
| Metadata | `U{XXXX}-IM-NEG.csv` |

**Metadata CSV structure** (`*-IM-NEG.csv`):
```
Row 0: "unfave1";"unfave2";"unfave3";"unfave4";"unfave5"  (column headers)
Row 1: "U0001-IM-NEG/1.png";...                            (file paths)
Row 2: "news headlines";"homelessness";...                  (user-provided text labels)
```

### 6. User Info Files (`*-INF.csv`)

| Property | Value |
|----------|-------|
| Location | `archive/.../Corpus/Corpus/U{XXXX}/U{XXXX}-INF.csv` |
| Delimiter | `;` |
| Key fields | Gender, Age, Type of Job, Income, Home country |

### 7. User Preference Files (`*-PREF.csv`)

| Property | Value |
|----------|-------|
| Location | `archive/.../Corpus/Corpus/U{XXXX}/U{XXXX}-PREF.csv` |
| Delimiter | `;` |
| Content | Self-reported preferences (websites, music, movies, TV, books) |

---

## Output Files

### 1. Ad Multi-hot CSV (`ad_multihot.csv`)

| Property | Value |
|----------|-------|
| Location | `data/output/ad_multihot.csv` |
| Delimiter | `,` |
| Encoding | UTF-8 |
| Rows | 300 (one per ad image) |
| Columns | 1 (image_id) + N (IAB categories, ~30) |

**Schema**:
```csv
image_id,Arts & Entertainment,Automotive,Business & Finance,...,Travel
Cat0_1,0,0,0,...,0
Cat0_2,1,0,0,...,0
...
Cat19_15,0,0,0,...,1
```

**Column types**:
- `image_id`: string, format `Cat{0-19}_{1-15}`
- All IAB columns: integer, values in {0, 1}

**Ordering**: Rows ordered by category index (0..19), then image index (1..15) within each category.

### 2. User Image Classifications CSV (`user_image_classifications.csv`)

| Property | Value |
|----------|-------|
| Location | `data/output/user_image_classifications.csv` |
| Delimiter | `,` |
| Encoding | UTF-8 |
| Rows | ~1200 (120 users × 10 images) |
| Columns | 3 metadata + N IAB categories |

**Schema**:
```csv
user_id,sentiment,image_index,Arts & Entertainment,Automotive,...,Travel
U0001,positive,1,1,0,...,0
U0001,positive,2,1,0,...,0
...
U0001,negative,1,0,0,...,1
U0001,negative,2,0,1,...,0
...
U0120,negative,5,0,0,...,0
```

**Column types**:
- `user_id`: string, format `U{0001..0120}`
- `sentiment`: string, values in {"positive", "negative"}
- `image_index`: integer, 1-5
- All IAB columns: integer, values in {0, 1}

### 3. User Profile Vectors CSV (`user_profiles.csv`)

| Property | Value |
|----------|-------|
| Location | `data/output/user_profiles.csv` |
| Delimiter | `,` |
| Encoding | UTF-8 |
| Rows | 120 (one per user) |
| Columns | 1 (user_id) + 3N (pos/neg/diff × N IAB categories) |

**Schema**:
```csv
user_id,pos_Arts & Entertainment,pos_Automotive,...,neg_Arts & Entertainment,neg_Automotive,...,diff_Arts & Entertainment,diff_Automotive,...
U0001,2,0,...,1,0,...,1,0,...
U0002,0,1,...,0,0,...,0,1,...
```

**Column types**:
- `user_id`: string
- `pos_*`: integer, range [0, 5] (sum of up to 5 binary values)
- `neg_*`: integer, range [0, 5]
- `diff_*`: integer, range [-5, 5] (pos - neg)

### 4. Pipeline State Checkpoint (`pipeline_state.json`)

| Property | Value |
|----------|-------|
| Location | `data/output/.pipeline_state.json` |
| Format | JSON |

**Schema**:
```json
{
  "run_id": "uuid-string",
  "started_at": "2026-05-12T10:00:00Z",
  "status": "running",
  "manifest_hash": "sha256-of-manifest",
  "completed_images": {
    "Cat0_1": {"categories": ["Style & Fashion", "Shopping"], "timestamp": "..."},
    ...
  },
  "failed_images": {
    "Cat5_3": {"error": "timeout", "attempts": 3, "last_attempt": "..."}
  },
  "current_batch_index": 15,
  "total_batches": 30,
  "config": {
    "batch_size": 10,
    "max_retries": 3,
    "taxonomy_version": "tier1_v1"
  }
}
```

### 5. Assembly Report (`assembly_report.json`)

| Property | Value |
|----------|-------|
| Location | `data/output/assembly_report.json` |
| Format | JSON |

**Schema**:
```json
{
  "run_id": "uuid-string",
  "completed_at": "2026-05-12T10:25:00Z",
  "ad_images": {
    "total": 300,
    "classified": 298,
    "failed": 2,
    "failed_ids": ["Cat5_3", "Cat12_7"]
  },
  "user_images": {
    "total": 1200,
    "classified": 1195,
    "failed": 5,
    "failed_ids": ["U0045_POS_3", "U0078_NEG_2", ...]
  },
  "taxonomy": {
    "version": "tier1_v1",
    "category_count": 30,
    "category_distribution": {
      "Style & Fashion": 45,
      "Shopping": 120,
      ...
    }
  },
  "output_files": [
    "data/output/ad_multihot.csv",
    "data/output/user_image_classifications.csv",
    "data/output/user_profiles.csv"
  ]
}
```

---

## Image ID Conventions

### Ad Images

Format: `Cat{category_index}_{image_number}`

- `category_index`: 0-based (0..19), derived from folder name (folder "1" → index 0)
- `image_number`: 1-based (1..15), derived from filename

Examples: `Cat0_1`, `Cat0_15`, `Cat19_1`, `Cat19_15`

This matches `ads16_processor.py`'s default `image_id_for = lambda c, i: f"Cat{c}_{i + 1}"`

### User Images

Format: `U{user_number}_{sentiment}_{image_number}`

- `user_number`: 4-digit zero-padded (0001..0120)
- `sentiment`: "POS" or "NEG"
- `image_number`: 1-based (1..5)

Examples: `U0001_POS_1`, `U0001_NEG_5`, `U0120_POS_3`

---

## Data Volume Summary

| Dataset | Images | Classifications | Output Rows |
|---------|--------|----------------|-------------|
| Ad images | 300 | 300 | 300 (ad_multihot.csv) |
| User positive images | 600 | 600 | 600 (user_image_classifications.csv) |
| User negative images | 600 | 600 | 600 (user_image_classifications.csv) |
| User profiles (aggregated) | — | — | 120 (user_profiles.csv) |
| **Total classifications** | **1,500** | **1,500** | — |
