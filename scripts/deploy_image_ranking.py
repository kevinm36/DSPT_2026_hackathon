"""
Deploy the image ranking agent and print the ARN to set in config.

Usage:
    python scripts/deploy_image_ranking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentcore.deploy import deploy_agent

AGENT_NAME = "image_ranking_agent"
AGENT_SRC = Path(__file__).resolve().parents[1] / "image_ranking_agent_src/image_ranking_agent.py"


def main():
    print(f"Deploying {AGENT_NAME}...")
    handler_code = AGENT_SRC.read_text()
    runtime = deploy_agent(
        name=AGENT_NAME,
        handler_code=handler_code,
        extra_requirements=["strands-agents", "bedrock-agentcore"],
    )
    print(f"  ✓ Deployed: {runtime['name']}")
    print(f"    ID:     {runtime['id']}")
    print(f"    ARN:    {runtime['arn']}")
    print(f"    Status: {runtime['status']}")
    print()
    print("Add this to config/agentcore.env:")
    print(f"  IMAGE_RANKING_AGENT_ARN={runtime['arn']}")


if __name__ == "__main__":
    main()
