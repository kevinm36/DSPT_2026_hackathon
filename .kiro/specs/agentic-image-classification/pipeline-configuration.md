# Pipeline Configuration Guide

## Overview

The agentic image classification pipeline is configured via a YAML configuration file and environment variables. This document describes all configurable parameters, their defaults, and when to adjust them.

---

## Configuration File

Location: `config/pipeline_config.yaml`

```yaml
# ============================================================
# Agentic Image Classification Pipeline Configuration
# ============================================================

# --- Data Paths ---
paths:
  # Root directory containing the ADS-16 archive
  archive_root: "archive/ADS16_Benchmark_part1/ADS16_Benchmark_part1"
  
  # Secondary archive (part 2, users U0061-U0120)
  archive_root_part2: "archive/ADS16_Benchmark_part2/ADS16_Benchmark_part2"
  
  # Output directory for generated CSVs
  output_dir: "data/output"
  
  # Checkpoint file location
  checkpoint_path: "data/output/.pipeline_state.json"

# --- Classification Backend ---
classification:
  # API endpoint for Tom & Kevin's facility
  api_endpoint: "${CLASSIFICATION_API_ENDPOINT}"
  
  # Model identifier
  model_name: "vision-language-model-v1"
  
  # Request timeout in seconds
  timeout_seconds: 60
  
  # Temperature (0 = deterministic for idempotency)
  temperature: 0.0
  
  # Maximum tokens in response
  max_response_tokens: 256

# --- Taxonomy ---
taxonomy:
  # Which IAB tier to use
  tier: 1
  
  # Path to the canonical category list (JSON array of strings)
  categories_file: "config/iab_tier1_categories.json"
  
  # Maximum categories per image (soft limit, triggers warning)
  max_categories_per_image: 5
  
  # Minimum categories per image (hard limit, triggers retry)
  min_categories_per_image: 1

# --- Batch Processing ---
batching:
  # Number of images per batch
  batch_size: 10
  
  # Maximum concurrent batches (1 = sequential)
  max_concurrent: 1
  
  # Delay between batches in milliseconds
  inter_batch_delay_ms: 1000
  
  # Whether to process ad images, user images, or both
  image_sets:
    - ads          # 300 ad images
    - user_pos     # 600 positive personal images
    - user_neg     # 600 negative personal images

# --- Retry Policy ---
retry:
  # Maximum retry attempts per image
  max_retries: 3
  
  # Initial backoff delay in milliseconds
  base_delay_ms: 1000
  
  # Maximum backoff delay
  max_delay_ms: 30000
  
  # Backoff multiplier (exponential)
  backoff_multiplier: 2.0
  
  # Circuit breaker: stop if failure rate exceeds this threshold
  circuit_breaker_threshold: 0.5
  
  # Circuit breaker window (number of recent requests to consider)
  circuit_breaker_window: 20
  
  # Errors that warrant retry
  retryable_errors:
    - "timeout"
    - "rate_limit"
    - "server_error"
    - "connection_error"

# --- Validation ---
validation:
  # Strict mode: reject any response with invalid categories
  strict_mode: false
  
  # Whether to attempt category name normalization
  normalize_names: true
  
  # Known aliases (model hallucinations → correct names)
  aliases:
    "Fashion": "Style & Fashion"
    "Tech": "Technology & Computing"
    "Technology": "Technology & Computing"
    "Food": "Food & Drink"
    "Finance": "Business & Finance"
    "Games": "Gaming"
    "Video Games": "Gaming"
    "Fitness": "Health & Fitness"
    "Luxury": "Jewelry & Luxury"

# --- Output ---
output:
  # Column name for image IDs
  image_id_column: "image_id"
  
  # CSV delimiter
  delimiter: ","
  
  # Whether to include confidence scores in a separate file
  include_confidence_scores: false
  
  # Whether to generate the diff_profile column in user_profiles.csv
  include_diff_profile: true

# --- Logging ---
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  
  # Log file location
  log_file: "data/output/pipeline.log"
  
  # Whether to log raw model responses (useful for debugging, large)
  log_raw_responses: false
```

---

## Environment Variables

These override config file values and are used for sensitive data:

| Variable | Required | Description |
|----------|----------|-------------|
| `CLASSIFICATION_API_ENDPOINT` | Yes | URL of the classification backend |
| `CLASSIFICATION_API_KEY` | Yes | Authentication key/token |
| `CLASSIFICATION_API_SECRET` | No | Secondary auth (if required) |
| `PIPELINE_OUTPUT_DIR` | No | Override output directory |
| `PIPELINE_LOG_LEVEL` | No | Override log level |

**Security**: Never commit API keys to source control. Use `.env` files (gitignored) or a secrets manager.

Example `.env`:
```bash
CLASSIFICATION_API_ENDPOINT=https://facility.example.com/v1/classify
CLASSIFICATION_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

---

## Tuning Guide

### Batch Size

| Scenario | Recommended `batch_size` | Rationale |
|----------|--------------------------|-----------|
| Rate-limited API | 5 | Smaller batches = less wasted work on failure |
| High-throughput API | 20-50 | Reduce overhead from batch setup |
| Debugging | 1 | Isolate individual image issues |
| Default | 10 | Balance between throughput and recoverability |

### Concurrency

| Scenario | Recommended `max_concurrent` | Rationale |
|----------|------------------------------|-----------|
| Unknown rate limits | 1 | Safe default, discover limits |
| Known high-capacity API | 3-5 | Parallel processing, ~3-5x speedup |
| Shared/metered API | 1 | Avoid consuming others' quota |

### Retry Policy

| Scenario | `max_retries` | `base_delay_ms` | Notes |
|----------|---------------|-----------------|-------|
| Stable API | 2 | 500 | Quick retries, low overhead |
| Flaky API | 5 | 2000 | More patience, longer backoff |
| Rate-limited API | 3 | 5000 | Respect rate limits |
| Cost-sensitive | 1 | 1000 | Minimize redundant API calls |

### Runtime Estimates

| Configuration | Ad Images (300) | User Images (1200) | Total |
|---------------|-----------------|--------------------|----|
| Sequential, 2s/image | 10 min | 40 min | 50 min |
| Sequential, 5s/image | 25 min | 100 min | 125 min |
| 3x parallel, 2s/image | 3.5 min | 14 min | 17.5 min |
| 5x parallel, 2s/image | 2 min | 8 min | 10 min |

---

## Resumption & Checkpointing

The pipeline automatically saves state after each batch completes. To resume after interruption:

```bash
# Resume from checkpoint (default behavior)
python -m pipeline.run --config config/pipeline_config.yaml

# Force fresh start (ignore checkpoint)
python -m pipeline.run --config config/pipeline_config.yaml --fresh

# Resume only failed images from a previous run
python -m pipeline.run --config config/pipeline_config.yaml --retry-failed
```

The checkpoint file (`.pipeline_state.json`) tracks:
- Which images have been successfully classified
- Which images failed and why
- Current position in the batch queue
- Run metadata (start time, config hash)

---

## Selective Processing

You can run the pipeline on subsets of the data:

```yaml
# Only process ad images (skip user personal images)
batching:
  image_sets:
    - ads

# Only process user images (skip ads)
batching:
  image_sets:
    - user_pos
    - user_neg

# Process everything (default)
batching:
  image_sets:
    - ads
    - user_pos
    - user_neg
```

This is useful for:
- Testing the pipeline on just the 300 ads first
- Re-running only user images after fixing an issue
- Iterating on prompt engineering with a smaller set
