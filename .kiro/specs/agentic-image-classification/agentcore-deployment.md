# AgentCore Deployment Guide

## Overview

The classification pipeline runs as a Bedrock AgentCore runtime. Agents are created dynamically — packaged as a zip, uploaded to S3, and registered as a runtime via the AgentCore API. A standalone Python script handles the full lifecycle.

## Quick Start: Hello World Test

This validates your entire AgentCore setup end-to-end in one command.

### Prerequisites

- Python 3.10+ with `boto3` installed (`pip install boto3`)
- AWS CLI configured with credentials for the `014498646416` account
- Your IAM user needs these permissions:
  - `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject` on `arn:aws:s3:::kevin-agentcore-bucket/*`
  - `bedrock-agentcore:*` (or scoped to create/invoke/delete)
  - `iam:PassRole` on `arn:aws:iam::014498646416:role/service-role/AmazonBedrockAgentCoreRuntimeDefaultServiceRole-*`

### Step 1: Set up config

```bash
cp config/agentcore.env.example config/agentcore.env
```

The example file is pre-filled with working values. No edits needed unless you want a different bucket or role.

### Step 2: Run the hello world

```bash
python scripts/deploy_hello_world.py --env config/agentcore.env
```

### Expected output

```
============================================================
AgentCore Hello World Deployment
============================================================
  Region:    us-east-1
  Bucket:    kevin-agentcore-bucket
  S3 Key:    agents/hello-world-test.zip
  Role ARN:  arn:aws:iam::014498646416:role/service-role/AmazonBedrockAgentCoreRuntimeDefaultServiceRole-rppbd
  Runtime:   hello_world_test
============================================================

[1/5] Packaging agent...
  Created: /tmp/.../hello_agent.zip
[2/5] Uploading to S3...
  Uploaded: s3://kevin-agentcore-bucket/agents/hello-world-test.zip
[3/5] Creating AgentCore runtime...
  Runtime ID: hello_world_test-mcvESLEsvr
  Runtime ARN: arn:aws:bedrock-agentcore:us-east-1:014498646416:runtime/hello_world_test-mcvESLEsvr
[4/5] Waiting for runtime...
  ✓ Runtime is READY (10s)
[5/5] Invoking runtime...
  Status: 200
  Response: {"message": "Hello Kevin! AgentCore is working."}

Cleaning up...
  ✓ Runtime deleted, S3 object removed

✓ Hello world test complete!
```

### Options

```bash
# Leave the runtime running after test (for further manual invocations)
python scripts/deploy_hello_world.py --env config/agentcore.env --no-cleanup

# Override specific settings via CLI
python scripts/deploy_hello_world.py \
    --env config/agentcore.env \
    --region us-east-1 \
    --bucket my-other-bucket
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Build & Deploy (scripts/deploy_hello_world.py)  │
│  1. Package agent code + deps → zip              │
│  2. Upload zip → S3                              │
│  3. Register runtime → Bedrock AgentCore         │
│  4. Wait for READY status                        │
│  5. Invoke with test payload                     │
└──────────────────────────┬───────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────┐
│  Bedrock AgentCore Runtime                       │
│  • Runs the agent code (Python 3.12)             │
│  • Accepts JSON payloads                         │
│  • Returns JSON responses                        │
└──────────────────────────────────────────────────┘
```

## Configuration Reference

### `config/agentcore.env`

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET` | Bucket for agent zip artifacts | `kevin-agentcore-bucket` |
| `S3_KEY` | S3 key for the classifier zip | `agents/image-classifier.zip` |
| `IAM_ROLE_ARN` | Role the runtime assumes | `...service-role/AmazonBedrockAgentCoreRuntimeDefaultServiceRole-rppbd` |
| `RUNTIME_NAME` | Name for the runtime | `image-classifier-agent` |
| `MODEL_ID` | Bedrock model ARN for classification | Claude Sonnet |

### IAM Permissions Required

**For your IAM user** (to deploy and invoke) — attach this as a single inline policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3AgentArtifacts",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::kevin-agentcore-bucket/*"
    },
    {
      "Sid": "BedrockAll",
      "Effect": "Allow",
      "Action": [
        "bedrock:*",
        "bedrock-agentcore:*"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PassRoleToAgentCore",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::014498646416:role/service-role/AmazonBedrockAgentCoreRuntimeDefaultServiceRole-*"
    }
  ]
}
```

If the inline policy alone doesn't work, also attach these **managed policies** directly to the user:
- `BedrockAgentCoreFullAccess`
- `AmazonBedrockFullAccess`

To attach: IAM Console → Users → your_username → Add permissions → Create inline policy → JSON tab → paste the above.

**The runtime role** (`AmazonBedrockAgentCoreRuntimeDefaultServiceRole-rppbd`) already has the correct trust policy and permissions — no changes needed.

## API Details (for reference)

### Two boto3 clients

| Client | Service | Purpose |
|--------|---------|---------|
| `bedrock-agentcore-control` | Control plane | Create, update, delete, list runtimes |
| `bedrock-agentcore` | Data plane | Invoke runtimes |

### Key API calls

```python
import boto3

# Control plane — manage runtimes
control = boto3.client("bedrock-agentcore-control", region_name="us-east-1")
control.create_agent_runtime(...)
control.list_agent_runtimes()
control.get_agent_runtime(agentRuntimeId="...")
control.update_agent_runtime(...)
control.delete_agent_runtime(agentRuntimeId="...")

# Data plane — invoke runtimes
data = boto3.client("bedrock-agentcore", region_name="us-east-1")
response = data.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-east-1:014498646416:runtime/...",
    payload='{"key": "value"}'
)
# Response fields: statusCode, response (StreamingBody), runtimeSessionId
body = response["response"].read().decode()
```

### Runtime naming rules

- Must match pattern: `[a-zA-Z][a-zA-Z0-9_]{0,47}`
- No hyphens allowed — use underscores
- Examples: `hello_world_test`, `image_classifier_agent`, `ranking_agent`

### Runtime lifecycle

```
create_agent_runtime() → CREATING → READY → invoke_agent_runtime()
                                           → update_agent_runtime() → UPDATING → READY
                                           → delete_agent_runtime() → DELETING → gone
```

## Invoking an Existing Runtime (Python)

If a runtime is already deployed and READY:

```python
import boto3
import json

client = boto3.client("bedrock-agentcore", region_name="us-east-1")

response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-east-1:014498646416:runtime/hello_world_test-mcvESLEsvr",
    payload=json.dumps({"name": "YourName"})
)

status = response["statusCode"]
body = response["response"].read().decode()
print(f"Status: {status}")
print(f"Response: {body}")
# Output: {"message": "Hello YourName! AgentCore is working."}
```

## Batch Classification Script (Implemented)

The batch classifier uses the same AgentCore pattern as the hello world:

```bash
python -m src.data_loader.agent_processing.batch_invoke_ads \
    --agent-arn arn:aws:bedrock-agentcore:us-east-1:014498646416:runtime/image_classifier-XXXXX \
    --ads-root archive/ADS16_Benchmark_part1/ADS16_Benchmark_part1 \
    --output data/ads16_agent_responses.jsonl \
    --max-workers 8
```

This script:
1. Discovers all PNG images under `--ads-root`
2. Base64-encodes each image and sends it to the classifier runtime with the IAB taxonomy prompt
3. Writes results to a resumable JSONL file (skips already-processed images on restart)
4. Runs with configurable concurrency via `--max-workers`

Post-processing into multi-hot CSVs:
```bash
python -m src.data_loader.multihot_from_responses \
    --responses data/ads16_agent_responses.jsonl \
    --output data/ads16_multihot.csv
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `AccessDenied: s3:PutObject` | Missing S3 permissions | Add S3 policy to your IAM user |
| `iam:PassRole not authorized` | Missing PassRole permission | Add PassRole policy for the service role |
| `Role validation failed` | Wrong role ARN (missing `service-role/` path) | Use full ARN with `role/service-role/...` |
| `Runtime name failed regex` | Hyphens in name | Use underscores only |
| `RuntimeClientError: initialization time exceeded` | Dependencies not bundled in zip | Ensure pip install --target bundles deps |
| `Status stuck on UPDATING` | Runtime updating after code change | Wait ~10-30s, it will reach READY |

## Dynamic Agent Creation

Agents are created programmatically. This means:
- You can spin up specialized agents for different tasks (classification, ranking, etc.)
- Agents can be versioned by uploading a new zip and calling `update_agent_runtime`
- Multiple agents can run in parallel for throughput
- The same pattern works for the LLM ranking agent in Stage 2
- One S3 bucket serves all agents (different keys per agent)

## Reusable Deployment Utility (`src/agentcore/deploy.py`)

The hello world script demonstrates the raw boto3 workflow. For production use, the project provides a higher-level utility that wraps the full lifecycle:

```python
from src.agentcore.deploy import deploy_agent, invoke_agent, delete_agent

# Deploy with a system prompt (auto-generates agent code)
runtime = deploy_agent(
    name="image_classifier",
    system_prompt="You classify images into IAB categories...",
    model_id="arn:aws:bedrock:us-east-1:...",
)

# Deploy with custom handler code
runtime = deploy_agent(
    name="image_ranking_agent",
    handler_code=open("image_ranking_agent_src/image_ranking_agent.py").read(),
)

# Invoke
result = invoke_agent(runtime["arn"], {"prompt": "...", "image_base64": "..."})

# Delete
delete_agent("image_classifier")
```

### Features

- **Create or update** — if a runtime with the same name exists, it updates rather than failing
- **Auto-packaging** — bundles code + pip dependencies into a zip for `manylinux2014_aarch64` / Python 3.12
- **Config from env** — reads `config/agentcore.env` for bucket, role, region, model ID
- **Wait for ready** — polls until the runtime reaches ACTIVE/READY status
- **Custom code support** — pass `handler_code` to deploy arbitrary agent logic

### Deployment Scripts

| Script | What it deploys | Runtime name |
|--------|----------------|--------------|
| `scripts/deploy_hello_world.py` | Minimal echo agent (validates setup) | `hello_world_test` |
| `scripts/deploy_image_ranking.py` | Full ranking agent from `image_ranking_agent_src/` | `image_ranking_agent` |

### Testing

```bash
# Test the deploy utility end-to-end
python tests/test_deploy.py

# Leave runtime running for manual testing
python tests/test_deploy.py --no-cleanup
```
