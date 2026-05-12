"""
Hello World: Deploy and invoke a minimal AgentCore runtime.

This script validates your setup end-to-end:
  1. Packages a trivial agent into a zip
  2. Uploads it to S3
  3. Creates an AgentCore runtime
  4. Waits for it to become ACTIVE
  5. Invokes it with a test payload
  6. Cleans up (optional)

Usage:
    # With env file:
    python scripts/deploy_hello_world.py --env config/agentcore.env

    # With explicit args:
    python scripts/deploy_hello_world.py \
        --region us-east-1 \
        --bucket my-bucket \
        --role-arn arn:aws:iam::123456789012:role/AgentCoreRole

    # Skip cleanup (leave runtime running for further testing):
    python scripts/deploy_hello_world.py --env config/agentcore.env --no-cleanup
"""

import argparse
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import boto3


# ============================================================
# HELLO WORLD AGENT CODE
# ============================================================
AGENT_CODE = '''
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    """Minimal hello world agent for testing deployment."""
    name = payload.get("name", "world")
    return {"message": f"Hello {name}! AgentCore is working."}

if __name__ == "__main__":
    app.run()
'''

AGENT_REQUIREMENTS = "bedrock-agentcore\n"


def load_env_file(env_path: str) -> dict:
    """Parse a KEY=VALUE env file, ignoring comments and blanks."""
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def package_agent() -> Path:
    """Create a zip containing the hello world agent with bundled dependencies."""
    tmpdir = tempfile.mkdtemp()
    project_dir = Path(tmpdir) / "hello_agent"
    project_dir.mkdir()

    (project_dir / "main.py").write_text(AGENT_CODE)
    (project_dir / "requirements.txt").write_text(AGENT_REQUIREMENTS)

    # Install dependencies into the project directory (bundled)
    import subprocess
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

    zip_path = Path(tmpdir) / "hello_agent.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_dir.rglob("*"):
            if file_path.is_file() and "__pycache__" not in str(file_path):
                zf.write(file_path, file_path.relative_to(project_dir))

    return zip_path


def wait_for_active(control_client, agent_runtime_id: str, timeout: int = 300) -> bool:
    """Poll until runtime is READY/ACTIVE or timeout."""
    print("Waiting for runtime to become READY", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        response = control_client.get_agent_runtime(agentRuntimeId=agent_runtime_id)
        status = response.get("status", "UNKNOWN")
        if status in ("ACTIVE", "READY"):
            print(f"\n✓ Runtime is {status} ({int(time.time() - start)}s)")
            return True
        if status in ("FAILED", "DELETING", "DELETE_FAILED"):
            print(f"\n✗ Runtime entered {status} state")
            failure_reason = response.get("statusReason", "unknown")
            print(f"  Reason: {failure_reason}")
            return False
        print(f"[{status}].", end="", flush=True)
        time.sleep(10)
    print(f"\n✗ Timeout after {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description="Deploy & test a hello world AgentCore runtime")
    parser.add_argument("--env", help="Path to env config file (e.g. config/agentcore.env)")
    parser.add_argument("--region", default=None, help="AWS region")
    parser.add_argument("--bucket", default=None, help="S3 bucket for agent zip")
    parser.add_argument("--s3-key", default="agents/hello-world-test.zip", help="S3 key")
    parser.add_argument("--role-arn", default=None, help="IAM role ARN for the runtime")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't delete runtime after test")
    args = parser.parse_args()

    # Load config from env file if provided
    env = {}
    if args.env:
        if not Path(args.env).exists():
            print(f"Error: env file not found: {args.env}")
            print(f"Copy config/agentcore.env.example to {args.env} and fill in your values.")
            sys.exit(1)
        env = load_env_file(args.env)

    # Resolve config (CLI args override env file)
    region = args.region or env.get("AWS_REGION", "us-east-1")
    bucket = args.bucket or env.get("S3_BUCKET")
    s3_key = args.s3_key or env.get("S3_KEY", "agents/hello-world-test.zip")
    role_arn = args.role_arn or env.get("IAM_ROLE_ARN")

    if not bucket:
        print("Error: --bucket or S3_BUCKET in env file is required")
        sys.exit(1)
    if not role_arn:
        print("Error: --role-arn or IAM_ROLE_ARN in env file is required")
        sys.exit(1)

    runtime_name = "hello_world_test"

    print("=" * 60)
    print("AgentCore Hello World Deployment")
    print("=" * 60)
    print(f"  Region:    {region}")
    print(f"  Bucket:    {bucket}")
    print(f"  S3 Key:    {s3_key}")
    print(f"  Role ARN:  {role_arn}")
    print(f"  Runtime:   {runtime_name}")
    print("=" * 60)

    # Step 1: Package
    print("\n[1/5] Packaging agent...")
    zip_path = package_agent()
    print(f"  Created: {zip_path}")

    # Step 2: Upload to S3
    print("\n[2/5] Uploading to S3...")
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(str(zip_path), bucket, s3_key)
    print(f"  Uploaded: s3://{bucket}/{s3_key}")

    # Step 3: Create runtime
    print("\n[3/5] Creating AgentCore runtime...")
    control_client = boto3.client("bedrock-agentcore-control", region_name=region)
    data_client = boto3.client("bedrock-agentcore", region_name=region)

    try:
        response = control_client.create_agent_runtime(
            agentRuntimeName=runtime_name,
            agentRuntimeArtifact={
                "codeConfiguration": {
                    "code": {
                        "s3": {
                            "bucket": bucket,
                            "prefix": s3_key
                        }
                    },
                    "runtime": "PYTHON_3_12",
                    "entryPoint": ["main.py"]
                }
            },
            roleArn=role_arn,
            networkConfiguration={
                "networkMode": "PUBLIC"
            }
        )
        agent_runtime_id = response.get("agentRuntimeId")
        agent_runtime_arn = response.get("agentRuntimeArn")
    except Exception as e:
        if "Conflict" in str(type(e).__name__) or "already exists" in str(e).lower():
            print("  Runtime already exists, fetching info...")
            list_resp = control_client.list_agent_runtimes()
            for rt in list_resp.get("agentRuntimes", []):
                if rt.get("agentRuntimeName") == runtime_name:
                    agent_runtime_id = rt["agentRuntimeId"]
                    agent_runtime_arn = rt["agentRuntimeArn"]
                    # Update with new code
                    print("  Updating runtime with new code...")
                    control_client.update_agent_runtime(
                        agentRuntimeId=agent_runtime_id,
                        agentRuntimeArtifact={
                            "codeConfiguration": {
                                "code": {
                                    "s3": {
                                        "bucket": bucket,
                                        "prefix": s3_key
                                    }
                                },
                                "runtime": "PYTHON_3_12",
                                "entryPoint": ["main.py"]
                            }
                        },
                        roleArn=role_arn,
                        networkConfiguration={
                            "networkMode": "PUBLIC"
                        }
                    )
                    break
            else:
                print("  Could not find existing runtime!")
                sys.exit(1)
        else:
            raise
    print(f"  Runtime ID: {agent_runtime_id}")
    print(f"  Runtime ARN: {agent_runtime_arn}")

    # Step 4: Wait for ACTIVE
    print("\n[4/5] Waiting for runtime...")
    if not wait_for_active(control_client, agent_runtime_id):
        print("\nFailed to reach ACTIVE state. Check AWS console for details.")
        sys.exit(1)

    # Step 5: Invoke
    print("\n[5/5] Invoking runtime...")
    invoke_response = data_client.invoke_agent_runtime(
        agentRuntimeArn=agent_runtime_arn,
        payload=json.dumps({"name": "Kevin"})
    )
    status_code = invoke_response.get("statusCode")
    response_body = invoke_response.get("response", "")
    if hasattr(response_body, "read"):
        response_body = response_body.read().decode()
    print(f"  Status: {status_code}")
    print(f"  Response: {response_body}")

    # Cleanup
    if not args.no_cleanup:
        print("\nCleaning up...")
        control_client.delete_agent_runtime(agentRuntimeId=agent_runtime_id)
        s3.delete_object(Bucket=bucket, Key=s3_key)
        print("  ✓ Runtime deleted, S3 object removed")
    else:
        print(f"\n  Runtime left running: {agent_runtime_id}")
        print(f"  ARN: {agent_runtime_arn}")
        print(f"  Delete later with: aws bedrock-agentcore-control delete-agent-runtime --agent-runtime-id {agent_runtime_id}")

    print("\n✓ Hello world test complete!")


if __name__ == "__main__":
    main()
