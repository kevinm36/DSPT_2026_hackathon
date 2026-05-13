from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

AGENT_MODEL_ENV: Final[str] = "AGENT_MODEL"
AGENT_MODEL_CHOICES: Final[tuple[str, ...]] = (
    "ImageRankingAgentModel",
    "IabAgentInferenceModel",
)


@dataclass(frozen=True)
class ImagePrediction:
    """Per-image output from `CustomInferenceInterface.predict`."""

    slot_index: int
    filename: str
    affinity: float
    reason: str
    image_attributes: dict[str, str]


class CustomInferenceInterface:
    """
    Affinity model interface: images + user profile in, ranked predictions out.

    Subclass or replace `predict` with a trained pipeline; keep `ImagePrediction`
    fields stable for `routers/web.py`.
    """

    @staticmethod
    def _get_image_attributes(blob: bytes) -> dict[str, str]:
        """Placeholder image-side categoricals until the real model returns them."""
        h = hashlib.sha256(blob).hexdigest()
        tones = ["warm", "cool", "neutral"]
        densities = ["minimal", "balanced", "rich"]
        moods = ["calm", "energetic", "formal"]
        return {
            "image_visual_tone": tones[int(h[:2], 16) % len(tones)],
            "image_layout_density": densities[int(h[2:4], 16) % len(densities)],
            "image_mood_signal": moods[int(h[4:6], 16) % len(moods)],
            "image_palette_family": f"family_{int(h[6:8], 16) % 5}",
        }

    def predict(
        self,
        image_payloads: list[tuple[str, bytes]],
        profile: dict[str, str],
    ) -> list[ImagePrediction]:
        """
        Parameters
        ----------
        image_payloads
            ``(filename, raw_bytes)`` per upload slot, in order (slot 0, 1, …).
        profile
            Validated customer profile: attribute id → string value.

        Returns
        -------
        One ``ImagePrediction`` per input image, sorted by ``affinity`` descending.
        ``slot_index`` always refers to the original upload index.
        """
        profile_blob = "|".join(f"{k}={v}" for k, v in sorted(profile.items()))
        profile_digest = hashlib.sha256(profile_blob.encode("utf-8")).hexdigest()[:12]

        scored: list[ImagePrediction] = []
        for slot_index, (filename, blob) in enumerate(image_payloads):
            h = hashlib.sha256(blob).hexdigest()
            affinity = int(h[:8], 16) / 0xFFFFFFFF
            reason = (
                f"Stub model: affinity derived from image hash and profile digest "
                f"{profile_digest}. Replace CustomInferenceInterface.predict with your trained model."
            )
            scored.append(
                ImagePrediction(
                    slot_index=slot_index,
                    filename=filename,
                    affinity=float(affinity),
                    reason=reason,
                    image_attributes=self._get_image_attributes(blob),
                )
            )

        scored.sort(key=lambda p: p.affinity, reverse=True)
        return scored


def configure_agent_model(name: str | None = None) -> CustomInferenceInterface:
    """Load ``ImageRankingAgentModel`` or ``IabAgentInferenceModel``; fall back to stub on failure.

    Tries the real IAB+ranking-agent model first (loads the saved Ridge
    bundle from ``saved_models/ridge_model.joblib`` and prepares the
    Bedrock tagger client). Falls back to the hash-based
    ``CustomInferenceInterface`` stub if the bundle is missing or import
    fails, so the dev server still boots when the trained artifacts
    aren't on disk.


    Updates the module-level ``default_agent_model`` used by ``routers/web.py``.
    """
    global default_agent_model
    key = (name if name is not None else os.environ.get(AGENT_MODEL_ENV, "")).strip()
    if not key:
        key = "ImageRankingAgentModel"
    if key not in AGENT_MODEL_CHOICES:
        logger.warning(
            "Unknown %s=%r (valid: %s). Using hash stub.",
            AGENT_MODEL_ENV,
            key,
            ", ".join(AGENT_MODEL_CHOICES),
        )
        print(
            f"[agent-model] Unknown {AGENT_MODEL_ENV}={key!r}; "
            f"valid choices are {', '.join(AGENT_MODEL_CHOICES)}. "
            "Loading CustomInferenceInterface (hash stub).",
            flush=True,
        )
        default_agent_model = CustomInferenceInterface()
        return default_agent_model

    try:
        if key == "ImageRankingAgentModel":
            from image_ranking_agent_pipeline.image_ranking_agent_model import ImageRankingAgentModel

            inst: CustomInferenceInterface = ImageRankingAgentModel()
        else:
            from src.inference.iab_agent_model import IabAgentInferenceModel

            inst = IabAgentInferenceModel()
    except Exception:
        logger.exception("%s failed to construct; using hash stub.", key)
        print(
            f"[agent-model] Failed to load {key}; falling back to "
            "CustomInferenceInterface (hash stub). See logs for details.",
            flush=True,
        )
        default_agent_model = CustomInferenceInterface()
        return default_agent_model

    default_agent_model = inst
    logger.info("Using agent model %s (%s)", key, type(inst).__name__)
    print(
        f"[agent-model] Loaded {key} -> {type(inst).__module__}.{type(inst).__name__}",
        flush=True,
    )
    return inst


default_agent_model: CustomInferenceInterface = CustomInferenceInterface()

# Honour AGENT_MODEL at import time so uvicorn picks up the env var.
# Safe to call: configure_agent_model has its own try/except and falls back
# to the stub on any failure.
configure_agent_model()


def stub_predict(
    image_payloads: list[tuple[str, bytes]],
    profile: dict[str, str],
) -> list[ImagePrediction]:
    """Backward-compatible entry point; delegates to ``default_agent_model``."""
    return default_agent_model.predict(image_payloads, profile)
