"""Run the web app with optional ``--agent-model`` (then start uvicorn).

Examples::

    python -m app --agent-model IabAgentInferenceModel --reload --app-dir .
    AGENT_MODEL=ImageRankingAgentModel python -m app --reload --app-dir .

``uvicorn`` itself does not understand ``--agent-model``; this wrapper sets
``AGENT_MODEL`` and re-invokes uvicorn with the remaining arguments.
"""

from __future__ import annotations

import argparse
import os
import sys

from app.services.model_service import AGENT_MODEL_CHOICES, AGENT_MODEL_ENV


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the affinity UI (uvicorn) with optional agent model selection.",
    )
    parser.add_argument(
        "--agent-model",
        choices=AGENT_MODEL_CHOICES,
        default=os.environ.get(AGENT_MODEL_ENV, "ImageRankingAgentModel"),
        metavar="NAME",
        help=f"Subclass of CustomInferenceInterface to load ({', '.join(AGENT_MODEL_CHOICES)}). "
        f"Also configurable via {AGENT_MODEL_ENV} when using plain uvicorn.",
    )
    args, uvicorn_argv = parser.parse_known_args()
    os.environ[AGENT_MODEL_ENV] = args.agent_model
    sys.argv = ["uvicorn", "app.main:app", *uvicorn_argv]
    from uvicorn.main import main as uvicorn_main

    uvicorn_main()


if __name__ == "__main__":
    main()
