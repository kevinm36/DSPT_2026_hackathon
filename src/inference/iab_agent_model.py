"""IAB content + ranking-agent inference model for the FastAPI demo app.

End-to-end pipeline (per request)::

    form profile (dict[str,str]) ─┐
                                  │  [step 1]
                                  ▼
                       ModelBundle.predict()           (Ridge on user demographics, default)
                                  │
                                  ▼
                       user_profile = {IAB cat: interest score in [0,1]}
                       (Ridge net_likes per-cat, min-max normalised across the 26 cats)

    uploaded images list ─┐
                          │  [step 2]   always-LLM tag via Bedrock AgentCore
                          ▼
                  candidate_ads = [{id, iab_profile: {cat: 0/1}, ...}]

                                  │  [step 3]
                                  ▼
                    ranking_agent_src.ranking_agent.invoke(payload)
                                  │
                                  ▼
                  {ranked_ads: [{id, score, reasoning}, ...], analysis}

                                  │  [step 4]
                                  ▼
                  list[ImagePrediction] (sorted by affinity desc)

Design choices the user signed off on:
  * Always LLM-tag uploaded images (no filename lookup shortcut).
  * Call ``ranking_agent.invoke`` in-process (Strands Agent runs inside this
    Python process; needs AWS credentials in env).
  * ``image_attributes`` surfaces the matched IAB tags themselves
    (real signal instead of placeholder palette/mood strings).
  * ``affinity`` is the raw IAB dot-product score (unbounded). It's the
    sum, over the IAB cats the image was tagged with, of the user's
    per-cat interest score. Higher = better; not normalised to [0, 1].

Failure modes degrade gracefully:
  * Missing LR bundle on disk     -> raise on construction (loud at boot).
  * Bedrock credentials missing    -> per-image LLM tag falls back to all-zero
                                      iab_profile + warning in ``reason``.
  * ranking_agent.invoke fails     -> fall back to local dot-product scoring;
                                      ``reason`` notes "LLM reasoning unavailable".
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

# The base class lives in the FastAPI app side; we import it here so this
# module is a self-contained subclass implementation.
from app.services.model_service import CustomInferenceInterface, ImagePrediction

from src.data_loader.agent_processing.batch_invoke_ads import (
    DEFAULT_AGENT_ARN, DEFAULT_REGION, _build_client, _detect_image_format,
)
from src.data_loader.agent_processing.categories_t1 import (
    DEFAULT_CATEGORIES_PATH, build_categorization_prompt, load_categories,
)
from src.data_loader.multihot_from_responses import (
    _build_canonical_patterns, _build_lookup, assign_categories,
)
from src.model.loader import ModelBundle, load_bundle


_LOGGER = logging.getLogger(__name__)

# Webapp-specific bundle location. The training-side defaults still point to
# ``Data/models/`` (used by compare_models / train_sentient), but the IAB
# bundles served by the FastAPI demo live here. Override at construction
# time via ``IabAgentInferenceModel(models_dir=...)`` if you relocate them.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEBAPP_MODELS_DIR = REPO_ROOT / "saved_models"


def _build_user_feature_dict(
    profile: dict[str, str], feature_names: list[str],
) -> dict[str, float]:
    """Coerce the validated form profile to the LR bundle's feature schema.

    Numerical attributes (e.g. ``inf__age``) come in as the raw string number;
    we cast to float and use 0 for invalid sentinels (the bundle treats missing
    keys as 0 anyway, so this just normalises the explicit invalid case).

    Categorical attributes (e.g. ``inf__fave_sports`` -> value
    ``inf__fave_sports__endurance_sports``) come in as the chosen one-hot
    column name; we set that column to 1 and leave its siblings at 0.
    """
    valid_names = set(feature_names)
    out: dict[str, float] = {}
    for key, raw in profile.items():
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()
        if not raw or raw.startswith("__"):
            continue
        if raw in valid_names:
            out[raw] = 1.0
            continue
        if key in valid_names:
            try:
                out[key] = float(raw)
            except ValueError:
                continue
    return out


# --------------------------------------------------------------------------- #
# Per-image LLM tagging (always-on, parallelised across the upload batch)
# --------------------------------------------------------------------------- #
class _LlmTagger:
    """Thread-safe wrapper around the deployed AgentCore image tagger.

    Built once per ``IabAgentInferenceModel`` instance: caches the boto3
    client, the canonical category list, the lookup table, the compiled
    canonical patterns, and the full prompt.
    """

    def __init__(
        self, *, agent_arn: str, region: str,
        categories_path: Path = DEFAULT_CATEGORIES_PATH,
        max_workers: int = 5,
    ) -> None:
        self.agent_arn = agent_arn
        self.region = region
        self.max_workers = max_workers
        self.categories: list[str] = load_categories(categories_path)
        self.lookup = _build_lookup(self.categories)
        self.patterns = _build_canonical_patterns(self.categories)
        self.prompt = build_categorization_prompt(
            categories=self.categories,
        )
        self._client_lock = threading.Lock()
        self._client: Any = None

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                self._client = _build_client(self.region)
            return self._client

    def _tag_one(self, blob: bytes) -> tuple[list[str], Optional[str]]:
        """Return ``(matched canonical cats, error_message_or_None)``."""
        try:
            image_format = _detect_image_format(blob)
        except ValueError as exc:
            return [], f"unsupported image format: {exc}"

        try:
            client = self._get_client()
        except Exception as exc:
            return [], f"cannot build Bedrock client: {exc!r}"

        payload = json.dumps({
            "prompt": self.prompt,
            "image_base64": base64.b64encode(blob).decode("utf-8"),
            "image_format": image_format,
        })
        try:
            response = client.invoke_agent_runtime(
                agentRuntimeArn=self.agent_arn,
                runtimeSessionId=f"{uuid.uuid4()}-iab-agent-model-session",
                payload=payload,
                qualifier="DEFAULT",
            )
            body = json.loads(response["response"].read().decode("utf-8"))
            text = body["result"]["content"][0]["text"]
        except Exception as exc:
            return [], f"agent invocation failed: {exc!r}"

        matched, _unmatched = assign_categories(
            text, self.lookup, canonical_patterns=self.patterns,
        )
        return matched, None

    def tag_batch(
        self, blobs: list[bytes],
    ) -> list[tuple[list[str], Optional[str]]]:
        """Tag a batch of images in parallel, preserving input order."""
        if not blobs:
            return []
        results: list[tuple[list[str], Optional[str]]] = [([], None)] * len(blobs)
        n_workers = max(1, min(self.max_workers, len(blobs)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(self._tag_one, blob): i
                for i, blob in enumerate(blobs)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    results[idx] = ([], f"tagger crashed: {exc!r}")
        return results


# --------------------------------------------------------------------------- #
# Main subclass
# --------------------------------------------------------------------------- #
class IabAgentInferenceModel(CustomInferenceInterface):
    """Affinity model that combines an LR-IAB user profile with the
    ranking-agent for per-request reranking + reasoning.
    """

    def __init__(
        self, *,
        user_bundle: Optional[ModelBundle] = None,
        bundle_name: str = "Ridge",
        models_dir: Path = DEFAULT_WEBAPP_MODELS_DIR,
        agent_arn: str = DEFAULT_AGENT_ARN,
        region: str = DEFAULT_REGION,
        max_tagger_workers: int = 5,
        top_k_for_reasoning: int = 5,
    ) -> None:
        # ``user_bundle`` is the per-(user, IAB-cat) scorer. Defaults to the
        # Ridge bundle (continuous signed net_likes) because LR's saturated
        # 0/1 probabilities + binary image multi-hot produce integer-valued
        # dot products that quantize the affinity to evenly-spaced fractions
        # (e.g. {0.0, 0.5, 1.0} for 3 images). Ridge's continuous outputs,
        # then normalised to [0, 1] in ``_user_iab_profile``, give a smooth
        # affinity score.
        self.user_bundle = (
            user_bundle if user_bundle is not None
            else load_bundle(bundle_name, models_dir=models_dir)
        )
        self.tagger = _LlmTagger(
            agent_arn=agent_arn, region=region,
            max_workers=max_tagger_workers,
        )
        self.scorable_categories: list[str] = list(
            self.user_bundle.scorable_categories
        )
        self.top_k_for_reasoning = top_k_for_reasoning

    # Backwards-compat alias: older smoke-test code accessed ``.lr_bundle``.
    @property
    def lr_bundle(self) -> ModelBundle:
        return self.user_bundle

    # ------------------------------------------------------------------ #
    # Step 1: form profile -> IAB user_profile dict
    # ------------------------------------------------------------------ #
    def _user_iab_profile(self, profile: dict[str, str]) -> dict[str, float]:
        """Return ``{IAB cat: interest score in [0, 1]}`` for the agent.

        For LR (``score_kind == "predict_proba"``) the bundle already returns
        ``P(like) in [0, 1]``; we just sort it into the canonical cat order.

        For Ridge / kNN (``score_kind == "predict"``) the bundle returns
        signed real ``net_likes`` predictions. We min-max normalise across
        the 26 scorable cats so the agent receives clean ``[0, 1]`` interest
        scores -- preserves rank order, makes the dot-product non-negative,
        and avoids the integer-quantization that LR exhibits in production.
        """
        feat = _build_user_feature_dict(profile, self.user_bundle.feature_names)
        scores = self.user_bundle.predict(feat)
        ordered = {c: float(scores.get(c, 0.0)) for c in self.scorable_categories}

        if self.user_bundle.score_kind == "predict_proba":
            return ordered  # already in [0, 1]

        # Ridge / kNN: continuous signed reals -> per-cat min-max to [0, 1].
        vals = list(ordered.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {c: 0.5 for c in ordered}
        return {c: (v - lo) / (hi - lo) for c, v in ordered.items()}

    # ------------------------------------------------------------------ #
    # Step 2: image -> IAB profile via LLM tagger
    # ------------------------------------------------------------------ #
    def _tag_images(
        self, image_payloads: list[tuple[str, bytes]],
    ) -> list[dict]:
        """Return per-image dicts ``{id, slot_index, filename, iab_profile, tag_error}``."""
        blobs = [blob for _, blob in image_payloads]
        tagged = self.tagger.tag_batch(blobs)

        records: list[dict] = []
        used_ids: set[str] = set()
        for slot_index, ((filename, _blob), (cats, err)) in enumerate(
            zip(image_payloads, tagged)
        ):
            ad_id = self._make_ad_id(filename, slot_index, used_ids)
            iab_profile = {c: int(c in cats) for c in self.scorable_categories}
            records.append({
                "id": ad_id,
                "slot_index": slot_index,
                "filename": filename,
                "iab_profile": iab_profile,
                "matched_cats": cats,
                "tag_error": err,
            })
        return records

    @staticmethod
    def _make_ad_id(filename: str, slot_index: int, used: set[str]) -> str:
        """Stable, agent-friendly ad id derived from filename + slot.

        The ranking_agent uses ``id`` to round-trip per-ad reasoning back to
        us, so it must be unique within a request and free of characters
        that would confuse a JSON LLM response.
        """
        stem = Path(filename).stem
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:40] or "img"
        candidate = f"slot{slot_index:02d}_{cleaned}"
        # Defensive: guarantee uniqueness even if two slots have identical names.
        while candidate in used:
            candidate += "_x"
        used.add(candidate)
        return candidate

    # ------------------------------------------------------------------ #
    # Step 3: ranking_agent invocation (in-process)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _local_dot_product(
        user_profile: dict[str, float], candidate_ads: list[dict],
    ) -> dict[str, float]:
        """Used both as the agent's score AND as fallback when the LLM call
        fails. Same formula the ranking_agent uses internally.
        """
        scores: dict[str, float] = {}
        for ad in candidate_ads:
            iab = ad.get("iab_profile", {})
            scores[ad["id"]] = sum(
                user_profile.get(cat, 0.0) * float(val)
                for cat, val in iab.items()
            )
        return scores

    def _invoke_ranking_agent(
        self, user_profile: dict[str, float], candidate_ads: list[dict],
    ) -> tuple[Optional[dict], Optional[str]]:
        """Return ``(parsed_result_or_None, error_message_or_None)``."""
        try:
            from ranking_agent_src.ranking_agent import invoke as rank_invoke
        except Exception as exc:
            return None, f"could not import ranking_agent: {exc!r}"

        payload = {
            "user_profile": user_profile,
            "candidate_ads": [
                {"id": ad["id"], "iab_profile": ad["iab_profile"]}
                for ad in candidate_ads
            ],
            "top_k": min(self.top_k_for_reasoning, len(candidate_ads)),
        }
        try:
            raw = rank_invoke(payload)
        except Exception as exc:
            return None, f"ranking_agent.invoke crashed: {exc!r}"

        if isinstance(raw, dict) and "error" in raw:
            return None, f"ranking_agent returned error: {raw['error']}"
        if not isinstance(raw, dict) or "result" not in raw:
            return None, f"ranking_agent returned unexpected shape: {type(raw).__name__}"

        result = raw["result"]
        if not isinstance(result, dict):
            return None, "ranking_agent result is not a dict"
        return result, None

    # ------------------------------------------------------------------ #
    # Step 4: assemble ImagePrediction list
    # ------------------------------------------------------------------ #
    def predict(
        self,
        image_payloads: list[tuple[str, bytes]],
        profile: dict[str, str],
    ) -> list[ImagePrediction]:
        if not image_payloads:
            return []

        # ---- Step 1: user IAB profile via LR bundle ----
        try:
            user_profile = self._user_iab_profile(profile)
        except Exception as exc:
            _LOGGER.exception("LR predict failed; using zero user profile")
            user_profile = {c: 0.0 for c in self.scorable_categories}
            profile_warning = f"User profile predict failed: {exc!r}"
        else:
            profile_warning = None

        # ---- Step 2: image IAB profiles via LLM tagger ----
        ad_records = self._tag_images(image_payloads)
        n_tag_failures = sum(1 for r in ad_records if r["tag_error"])
        if n_tag_failures:
            _LOGGER.warning(
                "LLM tagging failed for %d/%d images", n_tag_failures, len(ad_records)
            )

        # ---- Step 3: ranking_agent (with local-dot-product fallback) ----
        agent_result, agent_err = self._invoke_ranking_agent(
            user_profile, ad_records
        )
        per_ad_reasoning: dict[str, str] = {}
        analysis_text: str = ""
        if agent_result:
            for entry in agent_result.get("ranked_ads", []):
                if isinstance(entry, dict) and "id" in entry:
                    per_ad_reasoning[entry["id"]] = str(
                        entry.get("reasoning", "")
                    ).strip()
            analysis_text = str(agent_result.get("analysis", "")).strip()

        # Affinity = raw IAB dot product (no per-batch normalisation).
        # The downstream UI may render unbounded values; that's intentional.
        raw_scores = self._local_dot_product(user_profile, ad_records)

        # ---- Step 4: build ImagePrediction list ----
        predictions: list[ImagePrediction] = []
        for rec in ad_records:
            ad_id = rec["id"]
            reason_parts: list[str] = []
            if profile_warning:
                reason_parts.append(profile_warning)
            if rec["tag_error"]:
                reason_parts.append(
                    f"Image tagging failed: {rec['tag_error']}. "
                    f"This image got an empty IAB profile and a baseline affinity."
                )
            if agent_err:
                reason_parts.append(
                    f"Claude reasoning unavailable ({agent_err}); "
                    f"affinity is the raw IAB dot-product score."
                )
            if not agent_err and ad_id in per_ad_reasoning and per_ad_reasoning[ad_id]:
                reason_parts.append(per_ad_reasoning[ad_id])
            elif not agent_err:
                reason_parts.append(
                    "No per-ad reasoning returned by the ranking agent; "
                    "showing IAB dot-product score."
                )
            if analysis_text:
                reason_parts.append(f"Overall analysis: {analysis_text}")

            predictions.append(
                ImagePrediction(
                    slot_index=rec["slot_index"],
                    filename=rec["filename"],
                    affinity=float(raw_scores.get(ad_id, 0.0)),
                    reason=" \n\n".join(reason_parts) if reason_parts
                    else "IAB dot-product affinity (no additional context).",
                    image_attributes=self._image_attributes_from_iab(rec),
                )
            )

        predictions.sort(key=lambda p: p.affinity, reverse=True)
        return predictions

    # ------------------------------------------------------------------ #
    # image_attributes: surface the matched IAB tags (and key user stats)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _image_attributes_from_iab(rec: dict) -> dict[str, str]:
        cats = rec.get("matched_cats", []) or []
        if rec.get("tag_error"):
            return {
                "iab_categories": "(tagging failed)",
                "iab_category_count": "0",
            }
        return {
            "iab_categories": ", ".join(cats) if cats else "(none matched)",
            "iab_category_count": str(len(cats)),
        }


# --------------------------------------------------------------------------- #
# Convenience constructor used by app.services.model_service to bind
# default_agent_model. Wrapped in a function so that import-time failures
# (missing bundle, etc.) can be caught by the caller and degrade to the stub.
# --------------------------------------------------------------------------- #
def build_default(**kwargs) -> IabAgentInferenceModel:
    return IabAgentInferenceModel(**kwargs)


if __name__ == "__main__":
    # Tiny smoke check: instantiate and print expected feature/category counts
    # without doing any network calls.
    m = IabAgentInferenceModel()
    print(
        f"IabAgentInferenceModel ready: "
        f"{len(m.lr_bundle.feature_names)} user features, "
        f"{len(m.scorable_categories)} scorable IAB categories.",
        file=sys.stderr,
    )
