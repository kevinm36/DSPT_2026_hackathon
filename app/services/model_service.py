from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePrediction:
    filename: str
    affinity: float
    reason: str
    category: str = ""


class BaseModel(ABC):
    @abstractmethod
    def predict(
        self,
        image_payloads: list[tuple[str, bytes]],
        profile: dict[str, str],
    ) -> list[ImagePrediction]:
        ...


class StubModel(BaseModel):
    """Placeholder model: deterministic pseudo-scores from file bytes + profile."""

    def predict(
        self,
        image_payloads: list[tuple[str, bytes]],
        profile: dict[str, str],
    ) -> list[ImagePrediction]:
        profile_blob = "|".join(f"{k}={v}" for k, v in sorted(profile.items()))
        profile_digest = hashlib.sha256(profile_blob.encode("utf-8")).hexdigest()[:12]

        scored: list[ImagePrediction] = []
        for filename, blob in image_payloads:
            h = hashlib.sha256(blob).hexdigest()
            affinity = int(h[:8], 16) / 0xFFFFFFFF
            reason = (
                f"Stub model: affinity derived from image hash and profile digest "
                f"{profile_digest}. Replace with a real model."
            )
            scored.append(ImagePrediction(filename=filename, affinity=float(affinity), reason=reason))

        scored.sort(key=lambda p: p.affinity, reverse=True)
        return scored


# Default model instance used by the router
_model: BaseModel = StubModel()


def get_model() -> BaseModel:
    return _model


def set_model(model: BaseModel) -> None:
    global _model
    _model = model
