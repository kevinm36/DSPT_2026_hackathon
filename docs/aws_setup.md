# Bedrock AgentCore — Image-Capable Claude Agent

A Bedrock AgentCore agent that accepts text and image inputs, sends them to Claude Sonnet via the Strands SDK, and returns the model's response.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [AWS Account Setup](#aws-account-setup)
4. [Local Environment Setup](#local-environment-setup)
5. [Building the Deployment Zip](#building-the-deployment-zip)
6. [Deploying to AgentCore via the AWS Console](#deploying-to-agentcore-via-the-aws-console)
7. [How the Agent Code Works](#how-the-agent-code-works)

---

## Project Structure

```
hackathon/
├── my_agent.py          # Agent code deployed to AgentCore
├── requirements.txt     # Python dependencies for the agent
├── build_zip.sh         # Script to bundle agent + deps into deployment.zip
└── README.md            # This file
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Agent runtime |
| pip | latest | Package installation |
| AWS CLI | v2+ | AWS credential configuration |
| zip | any | Bundling the deployment package |

---

## AWS Account Setup

### 1. Install the AWS CLI

**macOS (Homebrew):**

```bash
brew install awscli
```

**macOS (official installer):**

```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
```

**Linux:**

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**Windows:**

Download and run the installer from https://awscli.amazonaws.com/AWSCLIV2.msi

**Verify installation:**

```bash
aws --version
```

### 2. Create an IAM User and Generate Access Keys

If you don't already have an IAM user with programmatic access:

1. Open the [IAM console](https://console.aws.amazon.com/iam/home)
2. In the left sidebar, click **Users** → **Create user**
3. Enter a username (e.g., `hackathon-dev`) and click **Next**
4. On the permissions page, choose **Attach policies directly**
5. Search for and attach `AdministratorAccess` (or a more scoped policy — see [Section 5](#5-iam-permissions-for-your-user--caller) below for the minimum permissions needed)
6. Click **Next** → **Create user**
7. Click on the newly created user to open their detail page
8. Go to the **Security credentials** tab
9. Scroll down to **Access keys** → **Create access key**
10. Select **Command Line Interface (CLI)** as the use case
11. Click **Next** → **Create access key**
12. **Copy both values now** — the Secret Access Key is only shown once:
    - **Access key ID** (starts with `AKIA...`)
    - **Secret access key**

### 3. Configure AWS CLI Credentials

```bash
aws configure
```

You will be prompted for:

| Prompt | What to enter |
|--------|---------------|
| AWS Access Key ID | The `AKIA...` key from step 2 |
| AWS Secret Access Key | The secret key from step 2 |
| Default region name | `us-east-1` (must match the region used throughout) |
| Default output format | `json` |

Verify it worked:

```bash
aws sts get-caller-identity
```

You should see your account ID and user ARN.

### 4. Enable Model Access in Bedrock

1. Open the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/home)
2. Make sure the region selector (top-right) is set to **US East (N. Virginia) / us-east-1**
3. In the left sidebar, click **Model access**
4. Click **Manage model access**
5. Find **Anthropic → Claude Sonnet 4** and check the box
6. Click **Save changes**
7. Wait until the status shows **Access granted**

### 5. Create the AgentCore Execution Role

The execution role is what the AgentCore runtime assumes to call Bedrock on your behalf.

1. Open the [IAM console](https://console.aws.amazon.com/iam/home)
2. Go to **Roles** → **Create role**
3. For **Trusted entity type**, choose **Custom trust policy** and paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock-agentcore.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

4. Click **Next**
5. Click **Create policy** (opens a new tab), switch to **JSON**, and paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvokeModel",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/bedrock-agentcore/*"
    }
  ]
}
```

6. Name the policy (e.g., `AgentCoreExecutionPolicy`) and create it
7. Back on the role creation tab, attach this policy
8. Name the role (e.g., `AgentCoreExecutionRole`) and create it
9. Copy the **Role ARN** — you'll need it when creating the agent runtime

### 6. IAM Permissions for Your User / Caller

Your IAM user (the one whose access keys you configured in `aws configure`) needs permission to invoke the agent. If you attached `AdministratorAccess` in step 2, you already have this. Otherwise, attach this as an inline policy on your IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "bedrock-agentcore:InvokeAgentRuntime",
      "Resource": "*"
    }
  ]
}
```

---

## Local Environment Setup

```bash
cd hackathon
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
```

Install the AWS SDK (needed for invoking the deployed agent):

```bash
pip install boto3
```

---

## Building the Deployment Zip

The agent runs on AgentCore's ARM64 (Graviton) Linux runtime, so dependencies must be packaged for that platform.

```bash
cd hackathon
chmod +x build_zip.sh
./build_zip.sh
```

This script:
1. Creates a `.build/` directory
2. Installs all pip dependencies from `requirements.txt` targeting `manylinux2014_aarch64` / Python 3.12
3. Copies `my_agent.py` and `requirements.txt` into the build directory
4. Zips everything into `deployment.zip`

The resulting zip is a flat archive with your code and all libraries at the root:

```
deployment.zip
├── my_agent.py
├── requirements.txt
├── bedrock_agentcore/
├── strands/
├── boto3/
├── botocore/
└── ... (transitive dependencies)
```

---

## Deploying to AgentCore via the AWS Console

### 1. Create the Agent Runtime

1. Open the [Bedrock AgentCore console](https://console.aws.amazon.com/bedrock-agentcore/home)
2. Ensure the region is **us-east-1**
3. Navigate to **Runtimes** → **Create agent runtime**
4. Configure:
   - **Name**: e.g., `tom_hackathon`
   - **Deployment method**: **Direct code deploy**
   - **Upload**: Select `deployment.zip`
   - **Entrypoint**: `my_agent.py`
   - **Execution role**: Select or paste the ARN of the `AgentCoreExecutionRole` you created above
5. Click **Create**
6. Wait for the status to show **Active**

### 2. Note the Agent Runtime ARN

After creation, copy the agent runtime ARN from the console. It will look like:

```
arn:aws:bedrock-agentcore:us-east-1:014498646416:runtime/tom_hackathon-ChO02O61W2
```

You'll use this ARN when invoking the agent via `boto3`.

### 3. Re-deploying After Code Changes

Each time you change `my_agent.py`:
1. Re-run `./build_zip.sh` to rebuild `deployment.zip`
2. In the AgentCore console, update the agent runtime and upload the new zip

---

## How the Agent Code Works

### Architecture Overview

```
┌──────────────┐       JSON payload        ┌─────────────────────┐
│    Caller     │ ──────────────────────►   │  AgentCore Runtime  │
│  (boto3 SDK)  │                           │                     │
│               │ ◄──────────────────────   │  my_agent.py        │
└──────────────┘       JSON response        │    ├── invoke()     │
                                            │    └── get_agent()  │
                                            └────────┬────────────┘
                                                     │
                                                     │ Bedrock API
                                                     ▼
                                            ┌─────────────────────┐
                                            │  Claude Sonnet 4    │
                                            │  (Foundation Model) │
                                            └─────────────────────┘
```

### Code Breakdown — `my_agent.py`

#### Top-Level: App Registration

```python
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()
```

Only the lightweight `BedrockAgentCoreApp` is imported at module level. This registers the HTTP entrypoint that AgentCore expects (listening on port 8080). Heavy dependencies are deferred to avoid the 30-second initialization timeout.

#### Lazy Agent Initialization: `get_agent()`

```python
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from strands import Agent
        _agent = Agent(model=MODEL_ID)
    return _agent
```

The `strands` import and `Agent()` constructor are expensive — they pull in `boto3`, `botocore`, and connect to the Bedrock service. By deferring this to the first invocation (rather than module load), we stay within the 30-second cold-start limit. The agent is created once and reused for all subsequent requests.

#### Entrypoint: `invoke(payload)`

```python
@app.entrypoint
def invoke(payload):
```

This is the function AgentCore calls for every incoming request. The `payload` is the JSON body sent by the caller.

#### Text-Only Path

```python
prompt = payload.get("prompt", "What do you see in this image?")
# ...
result = agent(prompt)
```

When no image is provided, the prompt string is passed directly to the Strands `Agent`, which forwards it to Claude via the Bedrock `InvokeModel` API.

#### Multimodal (Image) Path

```python
image_b64 = payload.get("image_base64")
image_format = payload.get("image_format", "png")

if image_b64:
    image_bytes = base64.b64decode(image_b64)
    content_blocks = [
        {"text": prompt},
        {
            "image": {
                "format": image_format,
                "source": {"bytes": image_bytes},
            },
        },
    ]
    result = agent(content_blocks)
```

When `image_base64` is present in the payload:
1. The base64 string is decoded back to raw bytes
2. A list of **content blocks** is built following the Bedrock/Claude message format:
   - A `text` block with the user's prompt
   - An `image` block with the format and raw bytes
3. The content block list is passed to the Strands Agent, which sends it as a multimodal message to Claude

#### Response

```python
return {"result": result.message}
```

`result.message` is Claude's full response object (including `role`, `content`, and `metadata` with token usage and latency metrics). This is serialized to JSON and returned to the caller.

### Payload Reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | No | `"What do you see in this image?"` | The text prompt sent to Claude |
| `image_base64` | string | No | `null` | Base64-encoded image bytes |
| `image_format` | string | No | `"png"` | Image format: `png`, `jpeg`, `gif`, or `webp` |

### Size Limits

- Base64 encoding adds ~33% overhead to the raw image size
- Claude supports images up to **3.75 MB** and **8000 x 8000 px**
- Keep source images under ~3 MB to stay within AgentCore's payload limits
