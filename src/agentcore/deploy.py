"""
AgentCore deployment utility.

Provides a single function to create/update and invoke an AgentCore runtime.
All AWS config (bucket, role, region) comes from the env file — callers only
need to provide the agent name, prompt/system instructions, and optionally
custom handler code.

Usage:
    from src.agentcore.deploy import deploy_agent, invoke_agent

    # Deploy a classifier agent
    runtime = deploy_agent(
        name="image_classifier",
        system_prompt="You classify images into IAB categories...",
        model_id="arn:aws:bedrock:us-east-1:014498646416:inference-profile/global.anthropic.claude-sonnet-4-6",
    )

    # Invoke it
    result = invoke_agent(runtime["arn"], {"image_base64": "...", "taxonomy": [...]})
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3


# ============================================================
# CONFIG LOADING
# ============================================================

def load_env(env_path: str = "config/agentcore.env") -> dict:
    """Parse a KEY=VALUE env file."""
    env = {}
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {env_path}. "
            f"Copy config/agentcore.env.example to config/agentcore.env"
        )
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


@dataclass
class AgentConfig:
    region: str
    bucket: str
    role_arn: str
    model_id: str

    @classmethod
    def from_env(cls, env_path: str = "config/agentcore.env") -> "AgentConfig":
        env = load_env(env_path)
        return cls(
            region=env.get("AWS_REGION", "us-east-1"),
            bucket=env["S3_BUCKET"],
            role_arn=env["IAM_ROLE_ARN"],
            model_id=env.get("MODEL_ID", ""),
        )


# ============================================================
# AGENT CODE GENERATION
# ============================================================

def generate_agent_code(
    system_prompt: str,
    model_id: str,
    handler_code: Optional[str] = None,
) -> str:
    """Generate the main.py agent code.

    Args:
        system_prompt: The system instructions for the agent.
        model_id: Bedrock model ARN to use.
        handler_code: Optional custom handler. If None, uses a default
            that passes the payload as a user message to the model.
    """
    # Escape for embedding in a Python string
    escaped_prompt = system_prompt.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    escaped_model = model_id.replace("\\", "\\\\").replace('"', '\\"')

    if handler_code:
        return handler_code

    return f'''
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

SYSTEM_PROMPT = """{system_prompt}"""
MODEL_ID = "{escaped_model}"

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from strands import Agent
        _agent = Agent(model=MODEL_ID, system_prompt=SYSTEM_PROMPT)
    return _agent


@app.entrypoint
def invoke(payload):
    import base64
    agent = get_agent()

    prompt = payload.get("prompt", "")
    image_b64 = payload.get("image_base64")
    image_format = payload.get("image_format", "png")

    if image_b64:
        image_bytes = base64.b64decode(image_b64)
        content_blocks = [
            {{"text": prompt}},
            {{
                "image": {{
                    "format": image_format,
                    "source": {{"bytes": image_bytes}},
                }},
            }},
        ]
        result = agent(content_blocks)
    else:
        result = agent(prompt)

    return {{"result": result.message}}


if __name__ == "__main__":
    app.run()
'''


# ============================================================
# PACKAGING
# ============================================================

def package_agent(agent_code: str, extra_requirements: Optional[list[str]] = None) -> Path:
    """Bundle agent code + dependencies into a deployable zip."""
    tmpdir = tempfile.mkdtemp()
    project_dir = Path(tmpdir) / "agent"
    project_dir.mkdir()

    (project_dir / "main.py").write_text(agent_code)

    requirements = ["bedrock-agentcore", "strands-agents"]
    if extra_requirements:
        requirements.extend(extra_requirements)
    (project_dir / "requirements.txt").write_text("\n".join(requirements) + "\n")

    # Bundle dependencies
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--target", str(project_dir),
            "--platform", "manylinux2014_aarch64",
            "--only-binary=:all:",
            "--python-version", "3.12",
            "-r", str(project_dir / "requirements.txt"),
        ],
        check=True,
        capture_output=True,
    )

    zip_path = Path(tmpdir) / "agent.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_dir.rglob("*"):
            if file_path.is_file() and "__pycache__" not in str(file_path):
                zf.write(file_path, file_path.relative_to(project_dir))

    return zip_path


# ============================================================
# DEPLOY
# ============================================================

def deploy_agent(
    name: str,
    system_prompt: str = "",
    model_id: Optional[str] = None,
    handler_code: Optional[str] = None,
    extra_requirements: Optional[list[str]] = None,
    env_path: str = "config/agentcore.env",
    wait: bool = True,
) -> dict:
    """Deploy (create or update) an AgentCore runtime.

    Args:
        name: Runtime name (alphanumeric + underscores only, e.g. "image_classifier").
        system_prompt: System instructions for the agent.
        model_id: Override model ID (defaults to config value).
        handler_code: Full custom main.py code. If provided, system_prompt is ignored.
        extra_requirements: Additional pip packages to bundle.
        env_path: Path to the env config file.
        wait: Whether to wait for READY status.

    Returns:
        dict with keys: id, arn, name, status
    """
    config = AgentConfig.from_env(env_path)
    if model_id is None:
        model_id = config.model_id

    # Generate and package
    code = generate_agent_code(system_prompt, model_id, handler_code)
    zip_path = package_agent(code, extra_requirements)

    # Upload to S3
    s3_key = f"agents/{name}.zip"
    s3 = boto3.client("s3", region_name=config.region)
    s3.upload_file(str(zip_path), config.bucket, s3_key)

    # Create or update runtime
    control = boto3.client("bedrock-agentcore-control", region_name=config.region)

    # Check if runtime already exists
    existing = _find_runtime(control, name)

    if existing:
        control.update_agent_runtime(
            agentRuntimeId=existing["id"],
            agentRuntimeArtifact={
                "codeConfiguration": {
                    "code": {"s3": {"bucket": config.bucket, "prefix": s3_key}},
                    "runtime": "PYTHON_3_12",
                    "entryPoint": ["main.py"],
                }
            },
            roleArn=config.role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
        )
        runtime_id = existing["id"]
        runtime_arn = existing["arn"]
    else:
        response = control.create_agent_runtime(
            agentRuntimeName=name,
            agentRuntimeArtifact={
                "codeConfiguration": {
                    "code": {"s3": {"bucket": config.bucket, "prefix": s3_key}},
                    "runtime": "PYTHON_3_12",
                    "entryPoint": ["main.py"],
                }
            },
            roleArn=config.role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
        )
        runtime_id = response["agentRuntimeId"]
        runtime_arn = response["agentRuntimeArn"]

    # Wait for ready
    status = "CREATING"
    if wait:
        status = _wait_for_ready(control, runtime_id)

    return {
        "id": runtime_id,
        "arn": runtime_arn,
        "name": name,
        "status": status,
    }


# ============================================================
# INVOKE
# ============================================================

def invoke_agent(runtime_arn: str, payload: dict, region: Optional[str] = None) -> dict:
    """Invoke a deployed AgentCore runtime.

    Args:
        runtime_arn: The runtime ARN.
        payload: JSON-serializable payload to send.
        region: AWS region (defaults to config).

    Returns:
        Parsed JSON response from the agent.
    """
    if region is None:
        try:
            config = AgentConfig.from_env()
            region = config.region
        except FileNotFoundError:
            region = "us-east-1"

    client = boto3.client("bedrock-agentcore", region_name=region)
    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        payload=json.dumps(payload),
    )

    body = response.get("response", "")
    if hasattr(body, "read"):
        body = body.read().decode()

    try:
        return json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return {"raw": body, "statusCode": response.get("statusCode")}


# ============================================================
# DELETE
# ============================================================

def delete_agent(name_or_id: str, env_path: str = "config/agentcore.env") -> bool:
    """Delete an AgentCore runtime by name or ID."""
    config = AgentConfig.from_env(env_path)
    control = boto3.client("bedrock-agentcore-control", region_name=config.region)

    # If it looks like a name, find the ID
    if "-" not in name_or_id or len(name_or_id) < 20:
        existing = _find_runtime(control, name_or_id)
        if not existing:
            return False
        runtime_id = existing["id"]
    else:
        runtime_id = name_or_id

    control.delete_agent_runtime(agentRuntimeId=runtime_id)
    return True


# ============================================================
# HELPERS
# ============================================================

def _find_runtime(control_client, name: str) -> Optional[dict]:
    """Find a runtime by name."""
    resp = control_client.list_agent_runtimes()
    for rt in resp.get("agentRuntimes", []):
        if rt.get("agentRuntimeName") == name:
            return {
                "id": rt["agentRuntimeId"],
                "arn": rt["agentRuntimeArn"],
                "name": rt["agentRuntimeName"],
                "status": rt.get("status"),
            }
    return None


def _wait_for_ready(control_client, runtime_id: str, timeout: int = 300) -> str:
    """Poll until runtime is READY or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = control_client.get_agent_runtime(agentRuntimeId=runtime_id)
        status = resp.get("status", "UNKNOWN")
        if status in ("ACTIVE", "READY"):
            return status
        if status in ("FAILED", "DELETE_FAILED"):
            raise RuntimeError(f"Runtime entered {status}: {resp.get('statusReason', 'unknown')}")
        time.sleep(10)
    raise TimeoutError(f"Runtime did not reach READY within {timeout}s")
