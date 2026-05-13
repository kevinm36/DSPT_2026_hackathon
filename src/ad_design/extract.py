"""Batch-invoke the deployed AgentCore image agent with the design-features prompt.

Reuses every helper from ``batch_invoke_ads`` (parallel pool, magic-byte
format detection, canonical paths, resume logic) - the only differences are:

  * uses ``build_prompt`` from this package instead of the IAB tier-1 prompt
  * adds a ``temperature`` knob in the payload (default 0.0)
  * default output path is ``Data/ads16_design_features.jsonl``
  * no canonical-category list is appended

Important - agent-side requirement
----------------------------------
The deployed agent at ``basic_img_agent_src/my_agent.py`` does not currently
read ``temperature`` from the payload. Sending it is harmless (it's ignored),
but to actually pin temperature=0 on the model you need to update the agent::

    # in my_agent.py, where we currently do:
    _agent = Agent(model=MODEL_ID)
    # change to:
    _agent = Agent(model=MODEL_ID, model_kwargs={"temperature": 0.0})

(or read the value from ``payload.get("temperature", 0.0)``) and redeploy.
Until then, output consistency relies on the strict prompt + schema rubric +
JSON-tag enforcement built into ``prompt.py``.

Run as a script::

    # Smoke test on a few images
    python -m src.ad_design.extract --limit 3

    # Full corpus (300 images)
    python -m src.ad_design.extract
"""

from __future__ import annotations

import argparse
import base64
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

from src.data_loader import ADS_ROOTS, discover_images
from src.data_loader.agent_processing.batch_invoke_ads import (
    DEFAULT_AGENT_ARN,
    DEFAULT_REGION,
    _build_client,
    _canonical_path,
    _detect_image_format,
    _load_processed_paths,
)

from .prompt import build_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
# Both ADS-16 release parts. Yields 300 images in canonical (folder 1..20,
# file 1..15) order via src.data_loader.discover_images.
DEFAULT_ADS_ROOTS: list[Path] = list(ADS_ROOTS)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "Data/ads16_design_features.jsonl"
DEFAULT_MAX_WORKERS = 8
DEFAULT_TEMPERATURE = 0.0


def invoke_one(
    image_path: Path,
    *,
    client: Any,
    agent_arn: str = DEFAULT_AGENT_ARN,
    prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """Invoke the agent on a single image and return a flat result record.

    Forked from ``batch_invoke_ads.invoke_one`` so we can include
    ``temperature`` in the payload (the upstream version doesn't).
    """
    image_bytes = image_path.read_bytes()
    image_format = _detect_image_format(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = json.dumps(
        {
            "prompt": prompt,
            "image_base64": image_b64,
            "image_format": image_format,
            "temperature": temperature,  # honored only if my_agent.py forwards it
        }
    )
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=f"{uuid.uuid4()}-ads16-design-session",
        payload=payload,
        qualifier="DEFAULT",
    )
    body = json.loads(response["response"].read().decode("utf-8"))
    text = body["result"]["content"][0]["text"]
    return {
        "category": image_path.parent.name,
        "image_id": image_path.stem,
        "path": _canonical_path(image_path),
        "text": text,
        "usage": body["result"].get("metadata", {}).get("usage"),
    }


def batch_invoke(
    ads_roots: Iterable[Path] = DEFAULT_ADS_ROOTS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    agent_arn: str = DEFAULT_AGENT_ARN,
    region: str = DEFAULT_REGION,
    max_workers: int = DEFAULT_MAX_WORKERS,
    temperature: float = DEFAULT_TEMPERATURE,
    examples: Optional[list[dict]] = None,
    limit: Optional[int] = None,
    resume: bool = True,
    images: Optional[Iterable[Path]] = None,
) -> Path:
    """Discover all ads under ``ads_roots`` and score every image.

    By default ``ads_roots`` is both ADS-16 release parts (300 images total),
    walked via ``src.data_loader.discover_images`` so order matches the rest
    of the pipeline (folder 1..20, file 1..15 within each folder).

    Parameters mirror ``batch_invoke_ads.batch_invoke`` plus:

    temperature:
        Sampling temperature. 0.0 (default) for max reproducibility. See the
        module docstring for the agent-side change needed to actually honor
        this; it's sent in the payload either way.
    examples:
        Optional list of fully-scored example dicts to inline as few-shot
        anchors in the prompt. See ``prompt.build_prompt``.
    images:
        Optional explicit iterable of paths. When provided, ``ads_roots`` is
        ignored. Used by ``validate.py`` to score a fixed sample twice.
    """
    if images is not None:
        image_list = list(images)
    else:
        image_list = discover_images(ads_roots)
    if limit is not None:
        image_list = image_list[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = _load_processed_paths(output_path) if resume else set()
    pending = [p for p in image_list if _canonical_path(p) not in already_done]

    print(
        f"Found {len(image_list)} image(s); "
        f"{len(already_done)} already scored; "
        f"{len(pending)} to invoke."
    )
    if not pending:
        return output_path

    full_prompt = build_prompt(examples=examples)
    client = _build_client(region=region)

    with output_path.open("a") as out, ThreadPoolExecutor(max_workers) as pool:
        futures = {
            pool.submit(
                invoke_one,
                p,
                client=client,
                agent_arn=agent_arn,
                prompt=full_prompt,
                temperature=temperature,
            ): p
            for p in pending
        }
        for i, fut in enumerate(as_completed(futures), 1):
            path = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:
                record = {"path": str(path), "error": repr(exc)}
            out.write(json.dumps(record) + "\n")
            out.flush()
            status = "ERROR" if "error" in record else "ok"
            print(f"[{i}/{len(pending)}] {path.parent.name}/{path.name} - {status}")

    return output_path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ads-roots", type=Path, nargs="+", default=DEFAULT_ADS_ROOTS,
        help="One or more ads root directories. Default: both ADS-16 parts "
             "(300 images total).",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--agent-arn", default=DEFAULT_AGENT_ARN)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N images (smoke test).")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    batch_invoke(
        ads_roots=args.ads_roots,
        output_path=args.output,
        agent_arn=args.agent_arn,
        region=args.region,
        max_workers=args.max_workers,
        temperature=args.temperature,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
