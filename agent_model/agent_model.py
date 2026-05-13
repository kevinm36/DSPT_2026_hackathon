"""
AgentModel — calls the deployed image ranking agent on Bedrock AgentCore.

Subclasses CustomInferenceInterface from app.services.model_service so it can
be used as a drop-in replacement for the stub.

Usage:
    from app.services.model_service import default_agent_model
    from agent_model.agent_model import AgentModel

    # Replace the default stub with the real agent
    import app.services.model_service as ms
    ms.default_agent_model = AgentModel(agent_arn="arn:aws:bedrock-agentcore:...")
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import boto3

from app.services.model_service import CustomInferenceInterface, ImagePrediction

_REPO_ROOT = Path(__file__).resolve().parents[1]


class AgentModel(CustomInferenceInterface):
    """Calls the deployed image ranking agent to classify and score ads."""

    def __init__(self, agent_arn: str | None = None, region: str | None = None):
        self.agent_arn = agent_arn or self._resolve_arn()
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")

        if not self.agent_arn:
            raise RuntimeError(
                "IMAGE_RANKING_AGENT_ARN not set. Deploy the agent with:\n"
                "  python scripts/deploy_image_ranking.py\n"
                "Then set the ARN in config/agentcore.env or as an env var."
            )

    @staticmethod
    def _resolve_arn() -> str:
        arn = os.environ.get("IMAGE_RANKING_AGENT_ARN", "")
        if arn:
            return arn

        env_path = _REPO_ROOT / "config" / "agentcore.env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("IMAGE_RANKING_AGENT_ARN="):
                    return line.split("=", 1)[1].strip()
        return ""

    def predict(
        self,
        image_payloads: list[tuple[str, bytes]],
        profile: dict[str, str],
    ) -> list[ImagePrediction]:
        agent_profile = self._build_agent_profile(profile)
        images = self._encode_images(image_payloads)

        payload = {
            "user_id": "ui_user",
            "profile": agent_profile,
            "images": images,
        }

        client = boto3.client("bedrock-agentcore", region_name=self.region)
        response = client.invoke_agent_runtime(
            agentRuntimeArn=self.agent_arn,
            payload=json.dumps(payload),
        )

        body = response.get("response", "")
        if hasattr(body, "read"):
            body = body.read().decode()
        result = json.loads(body)

        return self._parse_response(result, image_payloads)

    @staticmethod
    def _build_agent_profile(profile: dict[str, str]) -> dict:
        """Map UI form field IDs to the agent's expected profile format."""

        def _label_from_value(val: str) -> str:
            """Extract human-readable label from encoded value.
            e.g. 'pref__most_visited_websites__jewellery_watches_sites'
                 -> 'jewellery watches sites'
            """
            if not val:
                return ""
            parts = val.split("__")
            return parts[-1].replace("_", " ") if len(parts) > 1 else val

        gender_val = profile.get("inf__gender_male", "")
        gender = "M" if gender_val == "1" else "F" if gender_val == "0" else ""

        return {
            "inf": {
                "gender": gender,
                "age": profile.get("inf__age", ""),
                "job": "",
                "income": profile.get("inf__income", ""),
                "timepass": "",
                "fave_sports": _label_from_value(profile.get("inf__fave_sports", "")),
            },
            "pref": {
                "websites": _label_from_value(profile.get("pref__most_visited_websites", "")),
                "music": _label_from_value(profile.get("pref__most_listened_musics", "")),
                "movies": _label_from_value(profile.get("pref__most_watched_movies", "")),
                "tv": _label_from_value(profile.get("pref__most_watched_tv_programmes", "")),
                "books": _label_from_value(profile.get("pref__most_read_books", "")),
            },
            "pos_labels": [],
            "neg_labels": [],
        }

    @staticmethod
    def _encode_images(image_payloads: list[tuple[str, bytes]]) -> list[dict]:
        images = []
        for filename, blob in image_payloads:
            ext = Path(filename).suffix.lower()
            fmt = "png" if ext == ".png" else "jpeg"
            images.append({
                "image_id": filename,
                "image_base64": base64.b64encode(blob).decode(),
                "image_format": fmt,
            })
        return images

    def _parse_response(
        self,
        result: dict,
        image_payloads: list[tuple[str, bytes]],
    ) -> list[ImagePrediction]:
        classifications = {c["image_id"]: c for c in result.get("classifications", [])}
        scores_by_id = {s["image_id"]: s for s in result.get("scores", [])}

        predictions = []
        for slot_index, (filename, blob) in enumerate(image_payloads):
            score_info = scores_by_id.get(filename, {})
            class_info = classifications.get(filename, {})
            # Normalize score from [-1, 1] to [0, 1]
            raw_score = score_info.get("score", 0.0)
            affinity = (raw_score + 1) / 2.0
            predictions.append(ImagePrediction(
                slot_index=slot_index,
                filename=filename,
                affinity=affinity,
                reason=score_info.get("reasoning", "No reasoning provided"),
                image_attributes={
                    "ad_category": class_info.get("category", "Unknown"),
                    "classification_confidence": str(class_info.get("confidence", 0.0)),
                    **self._get_image_attributes(blob),
                },
            ))

        predictions.sort(key=lambda p: p.affinity, reverse=True)
        return predictions
