"""
Quick test for the deploy_agent / invoke_agent / delete_agent workflow.

Usage:
    python scripts/test_deploy.py

    # Keep the runtime running after test:
    python scripts/test_deploy.py --no-cleanup
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agentcore.deploy import deploy_agent, invoke_agent, delete_agent


def main():
    parser = argparse.ArgumentParser(description="Test the deploy_agent utility")
    parser.add_argument("--no-cleanup", action="store_true", help="Leave runtime running")
    args = parser.parse_args()

    print("=" * 50)
    print("Testing deploy_agent utility")
    print("=" * 50)

    # 1. Deploy
    print("\n[1] Deploying test agent...")
    runtime = deploy_agent(
        name="deploy_util_test",
        system_prompt="You are a helpful test agent. When given a name, respond with a greeting.",
    )
    print(f"  ✓ Deployed: {runtime['name']}")
    print(f"    ID:  {runtime['id']}")
    print(f"    ARN: {runtime['arn']}")
    print(f"    Status: {runtime['status']}")

    # 2. Invoke
    print("\n[2] Invoking agent...")
    result = invoke_agent(runtime["arn"], {"prompt": "Say hello to Kevin in one sentence."})
    print(f"  ✓ Response: {result}")

    # 3. Invoke again with different input
    print("\n[3] Invoking again...")
    result2 = invoke_agent(runtime["arn"], {"prompt": "What is 2 + 2? Reply with just the number."})
    print(f"  ✓ Response: {result2}")

    # 4. Cleanup
    if not args.no_cleanup:
        print("\n[4] Deleting agent...")
        deleted = delete_agent("deploy_util_test")
        print(f"  ✓ Deleted: {deleted}")
    else:
        print(f"\n[4] Skipping cleanup. Runtime left running: {runtime['id']}")
        print(f"    Delete later: python -c \"from src.agentcore.deploy import delete_agent; delete_agent('deploy_util_test')\"")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
