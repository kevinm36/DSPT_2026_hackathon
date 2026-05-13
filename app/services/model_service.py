from __future__ import annotations

import hashlib
from dataclasses import dataclass


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


default_agent_model = CustomInferenceInterface()


def stub_predict(
    image_payloads: list[tuple[str, bytes]],
    profile: dict[str, str],
) -> list[ImagePrediction]:
    """Backward-compatible entry point; delegates to ``default_agent_model``."""
    return default_agent_model.predict(image_payloads, profile)
