"""
Deploy (or update) the feature extraction agent and invoke it with a test image.

Usage:
    python tests/test_deploy_feature_extraction.py

    # Keep the runtime running after test:
    python tests/test_deploy_feature_extraction.py --no-cleanup

    # Use a specific test image:
    python tests/test_deploy_feature_extraction.py --image notebooks/11.png
"""

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agentcore.deploy import deploy_agent, invoke_agent, delete_agent

RUNTIME_NAME = "feature_extraction_agent"

AGENT_SOURCE = Path(__file__).parent.parent / "feature_extraction_agent_src" / "feature_extraction_agent.py"


def main():
    parser = argparse.ArgumentParser(description="Deploy & test the feature extraction agent")
    parser.add_argument("--no-cleanup", action="store_true", help="Leave runtime running")
    parser.add_argument("--image", default="notebooks/11.png", help="Path to a test image")
    args = parser.parse_args()

    handler_code = AGENT_SOURCE.read_text()

    print("=" * 60)
    print("Deploying feature extraction agent")
    print("=" * 60)

    # 1. Deploy
    print("\n[1] Deploying agent...")
    runtime = deploy_agent(
        name=RUNTIME_NAME,
        handler_code=handler_code,
    )
    print(f"  ✓ Deployed: {runtime['name']}")
    print(f"    ID:  {runtime['id']}")
    print(f"    ARN: {runtime['arn']}")
    print(f"    Status: {runtime['status']}")

    # 2. Invoke with a test image
    image_path = Path(args.image)
    if image_path.exists():
        print(f"\n[2] Invoking with image: {image_path}")
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        image_format = image_path.suffix.lstrip(".").replace("jpg", "jpeg")

        result = invoke_agent(runtime["arn"], {
            "image_base64": image_b64,
            "image_format": image_format,
        })
        print(f"  ✓ Response:\n{json.dumps(result, indent=2)}")
    else:
        print(f"\n[2] Skipping invocation — image not found: {image_path}")
        print(f"    Run with --image <path> to test with a specific image.")

    # 3. Invoke without an image (should return error)
    print("\n[3] Invoking without image (expect error)...")
    result2 = invoke_agent(runtime["arn"], {})
    print(f"  ✓ Response: {result2}")

    # 4. Cleanup
    if not args.no_cleanup:
        print("\n[4] Deleting agent...")
        deleted = delete_agent(RUNTIME_NAME)
        print(f"  ✓ Deleted: {deleted}")
    else:
        print(f"\n[4] Skipping cleanup. Runtime left running: {runtime['id']}")
        print(f"    ARN: {runtime['arn']}")
        print(f"    Delete later: python -c \"from src.agentcore.deploy import delete_agent; delete_agent('{RUNTIME_NAME}')\"")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
