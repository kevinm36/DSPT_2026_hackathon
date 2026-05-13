# Ad Design Feature Extraction

This module turns each ADS-16 ad image into a 20-field structured "design + sentiment" descriptor by prompting a vision LLM. Output is one CSV row per ad, joinable to the 300-image rating matrix via `image_id`.

The angle (vs. IAB content categorization): instead of "what topic is this ad about?", we ask "what does this ad look like?" — visual clutter, focal point, brand prominence, emotion valence, etc. Pairs naturally with the user's Big-5 personality features for "what kind of ad design appeals to whom" modeling.

## Pipeline at a glance

| # | Step | What it does | Time | Output |
|---|---|---|---|---|
| 1 | Inspect schema | Eyeball the 20 fields and rubrics | seconds | (read-only) |
| 2 | Eyeball prompt | See exactly what gets sent to the LLM | seconds | (read-only) |
| 3 | Automated test-retest | Score 20 ads twice, compute per-field agreement | ~5 min, ~40 LLM calls | `Data/ads16_design_features_consistency.csv` |
| 4 | **Human spot-check** | Manually review the test-retest output for plausibility | ~10 min, no LLM calls | (judgment call) |
| 5 | Score the corpus | Run on all 300 ads | ~20-40 min, 300 LLM calls | `Data/ads16_design_features.jsonl` |
| 6 | Flatten to CSV | Validate + flatten responses | seconds | `Data/ads16_design_features.csv` |
| 7 | Use in modeling | Join to user features for the recommender | n/a | downstream |

Steps 3 and 4 are **gates**. If automated agreement is poor or human spot-check flags issues, fix the prompt/schema in `src/ad_design/{schema,prompt}.py` and re-run before scaling to 300 ads.

---

## Step 1 — Inspect the schema

The schema is defined once in [`src/ad_design/schema.py`](../src/ad_design/schema.py) (20 fields with types, ranges, enum values, and rubric anchors).

```bash
python -c "from src.ad_design.schema import FIELD_NAMES; \
print(len(FIELD_NAMES), 'fields:'); \
[print(' ', f) for f in FIELD_NAMES]"
```

20 fields, in 5 logical groups:

| Group | Fields |
|---|---|
| Visual composition | `design_quality`, `visual_clutter`, `focal_point_presence`, `contrast_level`, `visual_saliency_score` |
| Subject | `primary_subject_type`, `human_presence` |
| Product / brand | `product_visibility`, `usage_context`, `brand_prominence`, `logo_present` |
| Text | `word_count_bin`, `text_density`, `readability` |
| Messaging | `value_proposition_present`, `cta_present`, `offer_present` |
| Emotion / trust | `emotion_valence`, `perceived_credibility`, `spamminess` |

**Refinements vs. the original 20-field draft:**

- `aesthetic_score` (1-10) → `design_quality` (`low`/`medium`/`high`)
- `perceived_credibility` (1-10) → `low`/`medium`/`high`
- `spamminess_score` (1-10) → `spamminess` `low`/`medium`/`high`
- `word_count` (int) → `word_count_bin` (`0`/`1-5`/`6-15`/`16-40`/`40+`)

These four fields are too subjective for an LLM to score consistently on a 10-point scale across calls. Binning preserves the dimension while making it reproducible. Every surviving 1-10 field gets a 3-anchor rubric (low/mid/high) injected into the prompt.

---

## Step 2 — Eyeball the prompt

```bash
python -c "from src.ad_design.prompt import build_prompt; print(build_prompt())"
```

The prompt has three blocks (full version is ~60 lines):

1. **Role + framing** — "You are an expert advertising creative analyst..."
2. **Output contract** — strict `<json>...</json>` tags, no commentary, exact enum casing, lowercase booleans, bare-digit ints
3. **Schema with rubrics** — every numeric field shows its 3 anchor descriptions inline

Excerpt:

```
"visual_clutter": <int 1-10>,  # Amount of distracting / unnecessary elements.
    - 1 = single subject, plenty of negative space
    - 5 = multiple elements but a clear hierarchy
    - 10 = chaotic, no clear visual order
"emotion_valence": "positive" | "neutral" | "negative",  # Overall emotional tone.
"word_count_bin": "0" | "1-5" | "6-15" | "16-40" | "40+",  # ...
```

If you want to anchor scores even more tightly, pass `examples=` to `build_prompt()` with 2-3 hand-scored ads. The recommended workflow: run validation first without examples, hand-pick the cleanest 3 outputs, and only then turn on few-shot.

---

## Step 3 — Automated test-retest validation (GATE 1)

```bash
python -m src.ad_design.validate
```

What it does:

1. Picks 20 images deterministically, stratified across all 20 ADS-16 category folders
2. Scores each one twice in two independent calls (≈40 LLM calls total)
3. Per field, computes agreement between pass 1 and pass 2:
   - **Numeric (1-10) fields**: Pearson + Spearman correlation, mean absolute diff
   - **Boolean / enum fields**: Cohen's kappa, raw accuracy
4. Prints a per-field report and writes `Data/ads16_design_features_consistency.csv`

Verdict thresholds (auto-applied):

| Field type | excellent | ok | weak (tighten rubric) | drop / recast |
|---|---|---|---|---|
| numeric (Pearson) | ≥ 0.85 | 0.70-0.85 | 0.50-0.70 | < 0.50 |
| categorical (κ) | ≥ 0.81 | 0.61-0.81 | 0.41-0.61 | < 0.41 |

Example output (illustrative — your numbers will differ):

```
field                  type   metric    value  verdict
human_presence         bool   kappa     0.92   excellent
logo_present           bool   kappa     0.87   excellent
primary_subject_type   enum   kappa     0.78   ok
visual_clutter         int    pearson   0.81   ok
focal_point_presence   int    pearson   0.74   ok
emotion_valence        enum   kappa     0.65   ok
spamminess             enum   kappa     0.52   weak - tighten rubric
brand_prominence       int    pearson   0.41   drop or recast
```

If anything lands "weak" or "drop", fix the rubric in `schema.py` and re-run with `--report-only` (skips the LLM calls if you're just iterating on rubric wording — but that won't help; you have to re-score after a prompt change).

---

## Step 4 — Human spot-check (GATE 2)

Automated test-retest tells you the LLM is **consistent with itself**. It does NOT tell you the LLM is **correct**. For that, do a quick eyeball pass on 5-10 of the validation outputs:

```bash
python -c "
import json
from pathlib import Path

p = Path('Data/ads16_design_features_pass1.jsonl')
recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
for r in recs[:5]:
    print('---')
    print(f'image: {r[\"path\"]}')
    print(r['text'])
"
```

Open the image at the printed path next to the LLM's JSON output. For each row, ask:

| Check | Pass criterion |
|---|---|
| Subject correctness | Does `primary_subject_type` match what's actually shown? |
| Brand correctness | Is `logo_present` correct? Does `brand_prominence` reflect actual emphasis? |
| Text correctness | Is `word_count_bin` plausible? Does `cta_present` match? |
| Emotion sanity | Does `emotion_valence` feel right looking at the image? |
| Clutter calibration | Does the LLM's `visual_clutter` of "5" look like a 5 to you, or like a 7? |

Reject the run and tighten the schema if you see:
- Systematic miscalibration (everything is 7-8 on every scale)
- Hallucinated fields (LLM invents categories not in the enum)
- Disagreement with obvious facts (LLM says no logo when there's a clear logo)

Calibration drift on numeric fields is the most common issue. Fix by either tightening the rubric anchors in `schema.py` or by adding 2-3 hand-scored few-shot examples to `prompt.build_prompt(examples=...)`.

---

## Step 5 — Score the full corpus

Once both gates pass:

```bash
python -m src.ad_design.extract
```

Defaults:
- All 300 images across **both** ADS-16 release parts (`ADS16_Benchmark_part1` + `ADS16_Benchmark_part2`)
- Output: `Data/ads16_design_features.jsonl`
- Resumable — safe to interrupt and re-run; already-processed paths are skipped
- 8 parallel workers, temperature=0
- ~5-10 seconds per image → 20-40 minutes wall-clock for all 300

Useful overrides:

```bash
# Smoke test (3 images)
python -m src.ad_design.extract --limit 3

# Only score one part of the dataset
python -m src.ad_design.extract --ads-roots Data/ADS-16/ADS16_Benchmark_part1/Ads

# More aggressive parallelism (watch for Bedrock throttling)
python -m src.ad_design.extract --max-workers 16
```

---

## Step 6 — Flatten to CSV

```bash
python -m src.ad_design.parse
```

Reads the JSONL, extracts the JSON payload from each `<json>...</json>` block, validates against the schema, and writes:

- `Data/ads16_design_features.csv` — 300 rows × 20 columns, indexed by `image_id` (`<category>_<image>`, e.g. `1_1`, `20_15`)

The parser also reports any rows that failed validation. Common failure modes:

| Failure | Fix |
|---|---|
| `no JSON found in response` | LLM ignored the `<json>` tag instruction. Re-run that image. |
| `<field>: <v> outside [1, 10]` | LLM emitted a 0 or 11. Re-run, or tighten rubric. |
| `<field>: <v> not in [...]` | LLM hallucinated an enum value. Tighten prompt wording. |

To keep partial rows in the CSV (NaN for failed fields), pass `--include-errors`.

---

## Step 7 — Use the features in modeling

The natural next step is a `(user, ad) → rating > user_mean` classifier. Concatenate ad-side features:

```
ad_features (300 x ~20)  =  design (Step 6)  +  IAB t1 multihot  (optional)
```

with user-side features:

```
user_features (120 x ~140)  =  demographics + prefs  +  Big-5 (B5)  (when available)
```

and join on the per-user rating matrix to get ~36k `(user, ad)` rows — which **breaks the N=120-users ceiling** the IAB-only models hit (macro AUC plateauing around 0.62). The implementation isn't in this folder yet; it will go alongside the existing baselines in `src/model/`.

---

## Examples: same product category, two design styles

These two ads come from **the same category folder** (`Ads/13/`, jewelry) — so
the product type is held constant and only the **design choices** vary. This
is the cleanest demonstration of what the schema is meant to capture: even
when two ads sell similar things, an LLM-derived design profile can clearly
separate "rich product photography with explicit discount" from "pure text
SERP listing".

> **About design variation in this dataset:** ADS-16 is heavily skewed toward
> text-only Google-search-result-style ads (small text block, no imagery,
> mostly white background). True graphic ads with product photography are a
> minority. The pair below spans most of the visual range that actually
> exists in the corpus, not the range of ad design in general.

### What gets sent to the LLM (same for both ads)

A single payload:
- `prompt`: the full prompt from [`build_prompt()`](../src/ad_design/prompt.py) (~60 lines, includes the schema + rubrics)
- `image_base64`: the PNG bytes, base64-encoded
- `image_format`: `"png"` (sniffed from magic bytes — `extract.py` does NOT trust the file extension)
- `temperature`: `0.0`

The LLM is expected to return only a `<json>...</json>` block matching the schema. After parsing, each ad becomes one row in `Data/ads16_design_features.csv` indexed by `image_id`.

---

### Example A — bold / product-rich design

**Image**: `Data/ADS-16/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Ads/Ads/13/8.png` → `image_id = 13_8`

![A diamond engagement ring product ad with a prominent red discounted price and original price strikethrough](../Data/ADS-16/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Ads/Ads/13/8.png)

What's in the image:
- Centered product photography of a princess-cut diamond ring on a white background
- Product name "3/4 ct. Princess Cut Diamond Solitaire Engagement Ring in 18k White Gold"
- Seller "by ND Outlet - Engagement"
- **Prominent red price `£899.00` with strikethrough `£3,100.00`** — visible discount messaging
- Small blue link "Show only ND Outlet - Engagement items"

Plausible LLM scoring (hand-scored, illustrative):

```json
<json>
{
  "design_quality": "high",
  "visual_clutter": 2,
  "focal_point_presence": 9,
  "contrast_level": 6,
  "visual_saliency_score": 7,
  "primary_subject_type": "product",
  "human_presence": false,
  "product_visibility": 10,
  "usage_context": "standalone",
  "brand_prominence": 3,
  "logo_present": false,
  "word_count_bin": "16-40",
  "text_density": 3,
  "readability": 8,
  "value_proposition_present": true,
  "cta_present": false,
  "offer_present": true,
  "emotion_valence": "positive",
  "perceived_credibility": "high",
  "spamminess": "low"
}
</json>
```

Why these scores make sense: the ad has **clear product photography** (`product_visibility: 10`, `primary_subject_type: "product"`), a **single dominant subject** (`focal_point_presence: 9`, `visual_clutter: 2`), and a **visible discount** in red (`offer_present: true`, `emotion_valence: "positive"`). It's not "spammy" — the product photography and explicit retailer name read as legitimate.

---

### Example B — simplistic / text-only design (same jewelry category)

**Image**: `Data/ADS-16/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Ads/Ads/13/3.png` → `image_id = 13_3`

![A plain text-only Google search-result-style ad for an Italian jewellery shop, with a small yellow Ad badge, blue headline, green URL, and a few short lines of body text and blue link sub-categories](../Data/ADS-16/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Ads/Ads/13/3.png)

What's in the image:
- Small yellow `Ad` badge next to the green URL
- Blue headline `Jewellery Shop - Contemporary Jewellery Online`
- Green URL `www.alaricogentili.it`
- Body text: "Made in Italy / Innate gifts of poetry · Innate gifts of beauty"
- Blue sub-link row: "Triangle Stone - Circle Stone - Grunge Stone"
- Mostly empty white space; no product image, no price, no discount

Plausible LLM scoring (hand-scored, illustrative):

```json
<json>
{
  "design_quality": "medium",
  "visual_clutter": 2,
  "focal_point_presence": 4,
  "contrast_level": 4,
  "visual_saliency_score": 2,
  "primary_subject_type": "text-only",
  "human_presence": false,
  "product_visibility": 1,
  "usage_context": "none",
  "brand_prominence": 4,
  "logo_present": false,
  "word_count_bin": "16-40",
  "text_density": 4,
  "readability": 7,
  "value_proposition_present": false,
  "cta_present": false,
  "offer_present": false,
  "emotion_valence": "neutral",
  "perceived_credibility": "medium",
  "spamminess": "low"
}
</json>
```

Why these scores make sense: **no product imagery** (`product_visibility: 1`, `primary_subject_type: "text-only"`), **highly minimal** (`visual_saliency_score: 2` — easy to skim past), no discount, no clear value prop ("Made in Italy / Innate gifts of poetry" is brand voice, not a measurable benefit), and no recognizable brand to anchor credibility (`brand_prominence: 4`, `perceived_credibility: medium`).

---

### Side-by-side: same product, different design

| `image_id` | `design_quality` | `visual_clutter` | `primary_subject_type` | `product_visibility` | `offer_present` | `emotion_valence` | `perceived_credibility` |
|---|---|---|---|---|---|---|---|
| `13_8` (ring) | `high` | `2` | `product` | **`10`** | **`true`** | **`positive`** | **`high`** |
| `13_3` (text) | `medium` | `2` | `text-only` | **`1`** | **`false`** | **`neutral`** | **`medium`** |

Both ads sell jewelry. Both have low `visual_clutter` — but the ring scores `2` because it's a clean *product photo*, while the text ad scores `2` because it's *sparse text*. That's exactly why the schema needs more than one dimension: `clutter` alone can't separate "minimalist product ad" from "barebones search listing". You need `clutter` + `product_visibility` + `primary_subject_type` together.

The fields that flipped most starkly between the two rows — `product_visibility` (10 → 1), `offer_present` (true → false), `emotion_valence` (positive → neutral), `perceived_credibility` (high → medium) — are exactly the signal a downstream personality-aware model would key off, e.g.:

- "Users high in **Openness** rate visually-rich product ads higher than text-only ones" → would land on `product_visibility` and `design_quality`
- "Users high in **Conscientiousness** prefer ads with explicit, credible value props" → would land on `value_proposition_present`, `offer_present`, `perceived_credibility`

```python
import pandas as pd
ads = pd.read_csv("Data/ads16_design_features.csv", index_col="image_id")
ads.loc["13_8"]    # the diamond ring row
ads.loc["13_3"]    # the jewelry text ad row
ads.loc[["13_8", "13_3"]].T   # side-by-side comparison
```

> **Important:** the JSON values above are **hand-scored** by a human looking at the images — they show what a reasonable scoring looks like and what differences the schema picks up. They are NOT actual LLM outputs. Once you've run Step 3 (validation), open `Data/ads16_design_features_pass1.jsonl`, find the records for these two `image_id`s, and replace the JSON blocks with the actual model output. If the LLM's scores diverge wildly from the hand-scored ones above on the **starkly-different** fields (the bolded ones in the table), the prompt needs work — those differences are unmistakable.

---

## Schema reference

Full type / range / rubric per field is the source of truth in [`src/ad_design/schema.py`](../src/ad_design/schema.py). Quick summary:

| Field | Type | Range / values |
|---|---|---|
| `design_quality` | enum | `low` / `medium` / `high` |
| `visual_clutter` | int | 1-10 |
| `focal_point_presence` | int | 1-10 |
| `contrast_level` | int | 1-10 |
| `visual_saliency_score` | int | 1-10 |
| `primary_subject_type` | enum | `product` / `person` / `scene` / `text-only` / `mixed` |
| `human_presence` | bool | true / false |
| `product_visibility` | int | 1-10 |
| `usage_context` | enum | `in-use` / `standalone` / `lifestyle` / `abstract` / `none` |
| `brand_prominence` | int | 1-10 |
| `logo_present` | bool | true / false |
| `word_count_bin` | enum | `0` / `1-5` / `6-15` / `16-40` / `40+` |
| `text_density` | int | 1-10 |
| `readability` | int | 1-10 |
| `value_proposition_present` | bool | true / false |
| `cta_present` | bool | true / false |
| `offer_present` | bool | true / false |
| `emotion_valence` | enum | `positive` / `neutral` / `negative` |
| `perceived_credibility` | enum | `low` / `medium` / `high` |
| `spamminess` | enum | `low` / `medium` / `high` |

---

## Important caveat: temperature is sent but not honored yet

The deployed agent at `basic_img_agent_src/my_agent.py` does NOT currently read `temperature` from the payload. `extract.py` sends `temperature=0` regardless, but to actually pin it on the model you need to update the agent and redeploy:

```python
# in basic_img_agent_src/my_agent.py
_agent = Agent(model=MODEL_ID, model_kwargs={"temperature": 0.0})
```

Until that change is deployed, output consistency relies entirely on the strict prompt + schema rubric + JSON-tag enforcement built into `prompt.py`. That's good for ~70-80% of variance reduction in my experience, but the remainder needs the agent-side fix. **Step 3's test-retest is exactly the experiment that tells you whether prompt-level discipline alone is enough.**
