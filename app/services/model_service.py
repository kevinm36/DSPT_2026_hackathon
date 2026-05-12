from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePrediction:
    filename: str
    affinity: float
    reason: str


def stub_predict(
    image_payloads: list[tuple[str, bytes]],
    profile: dict[str, str],
) -> list[ImagePrediction]:
    """
    Placeholder model: deterministic pseudo-scores from file bytes + profile.
    Replace with a call to your trained supervised model.
    """
    profile_blob = "|".join(f"{k}={v}" for k, v in sorted(profile.items()))
    profile_digest = hashlib.sha256(profile_blob.encode("utf-8")).hexdigest()[:12]

    scored: list[ImagePrediction] = []
    for filename, blob in image_payloads:
        h = hashlib.sha256(blob).hexdigest()
        # Map first 8 hex chars to a float in [0, 1)
        affinity = int(h[:8], 16) / 0xFFFFFFFF
        reason = (
            f"Stub model: affinity derived from image hash and profile digest "
            f"{profile_digest}. Replace model_service.stub_predict with your trained model."
        )
        scored.append(ImagePrediction(filename=filename, affinity=float(affinity), reason=reason))

    scored.sort(key=lambda p: p.affinity, reverse=True)
    return scored
