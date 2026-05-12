"""Batch-invoke the deployed AgentCore image agent over the ADS-16 Ads folder.

The deployed agent (``basic_img_agent_src/my_agent.py``) accepts one image at a
time as a JSON payload of the form::

    {"prompt": "...", "image_base64": "...", "image_format": "png"}

This module walks a directory of ``.png`` ads, encodes each image, calls the
agent runtime over ``boto3.invoke_agent_runtime``, and writes the responses to
a JSONL file - one record per image - so the run is resumable-friendly and easy
to post-process.

Run as a script::

    python -m src.data_loader.agent_processing.batch_invoke_ads

Or import and call programmatically::

    from src.data_loader.agent_processing import batch_invoke
    batch_invoke(ads_root=Path("Data/ADS-16/.../Ads/Ads"),
                 output_path=Path("Data/ads16_agent_responses.jsonl"))
"""

from __future__ import annotations

import argparse
import base64
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import boto3

from .categories_t1 import (
    DEFAULT_CATEGORIES_PATH,
    PROMPT_INSTRUCTION,
    build_categorization_prompt,
)


DEFAULT_ADS_ROOT = Path(
    "Data/ADS-16/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Ads/Ads"
)
DEFAULT_OUTPUT_PATH = Path("Data/ads16_agent_responses.jsonl")
DEFAULT_AGENT_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:"
    "014498646416:runtime/tom_hackathon-ChO02O61W2"
)
DEFAULT_PROMPT = PROMPT_INSTRUCTION
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_WORKERS = 8


def _build_client(region: str = DEFAULT_REGION) -> Any:
    return boto3.client("bedrock-agentcore", region_name=region)


def _canonical_path(p: Path | str) -> str:
    """Resolve to an absolute, symlink-free path string.

    Used everywhere we compare or store image paths so that different spellings
    of the same image (relative vs absolute, ``./`` prefix, symlinks) collapse
    to a single key. Without this, the resume check treats the relative path
    used by the CLI default and the absolute path built by ``main.py`` as
    different images and re-invokes the agent.
    """
    return str(Path(p).resolve())


def _detect_image_format(image_bytes: bytes) -> str:
    """Sniff the image format from magic bytes.

    Returns one of ``png``, ``jpeg``, ``gif``, ``webp`` (the four formats
    Bedrock/Claude accepts). Raises ``ValueError`` if the bytes don't match
    any of them - lying about ``image_format`` to Claude is what produces the
    "RuntimeClientError 500" we were getting on misnamed ADS-16 files (e.g.
    files with a ``.png`` extension that are actually GIF or JPEG).
    """
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    raise ValueError(
        f"Unsupported image format (first 16 bytes: {image_bytes[:16]!r}). "
        f"Bedrock/Claude only accepts png, jpeg, gif, webp."
    )


def invoke_one(
    image_path: Path,
    *,
    client: Any,
    agent_arn: str = DEFAULT_AGENT_ARN,
    prompt: str = DEFAULT_PROMPT,
) -> dict:
    """Invoke the agent on a single image and return a flat result record."""
    image_bytes = image_path.read_bytes()
    image_format = _detect_image_format(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = json.dumps(
        {
            "prompt": prompt,
            "image_base64": image_b64,
            "image_format": image_format,
        }
    )
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=f"{uuid.uuid4()}-ads16-batch-session",
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


def _load_processed_paths(output_path: Path) -> set[str]:
    """Return the set of canonical image paths already present in ``output_path``."""
    if not output_path.is_file():
        return set()
    seen: set[str] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = record.get("path")
            if path and "error" not in record:
                # Canonicalize so relative records from older runs match new
                # absolute records (and vice versa).
                seen.add(_canonical_path(path))
    return seen


def _discover_images(ads_root: Path) -> list[Path]:
    if not ads_root.is_dir():
        raise FileNotFoundError(f"Ads root not found: {ads_root}")
    return sorted(ads_root.rglob("*.png"))


def batch_invoke(
    ads_root: Path = DEFAULT_ADS_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    agent_arn: str = DEFAULT_AGENT_ARN,
    prompt: str = DEFAULT_PROMPT,
    categories_file: Optional[Path] = DEFAULT_CATEGORIES_PATH,
    region: str = DEFAULT_REGION,
    max_workers: int = DEFAULT_MAX_WORKERS,
    limit: Optional[int] = None,
    resume: bool = True,
    images: Optional[Iterable[Path]] = None,
) -> Path:
    """Walk ``ads_root`` and invoke the agent on every ``.png`` in parallel.

    Parameters
    ----------
    ads_root:
        Directory to walk recursively for ``.png`` files.
    output_path:
        JSONL file to append results into. Created if missing.
    agent_arn:
        ARN of the deployed AgentCore runtime to call.
    prompt:
        Text instruction sent to Claude. The full canonical category list is
        appended automatically when ``categories_file`` is provided.
    categories_file:
        Path to the newline-delimited canonical category list. When set
        (default), the categories are read once and appended to ``prompt``
        under a ``Full list of categories`` header. Pass ``None`` to send the
        bare instruction without any list.
    region:
        AWS region of the agent runtime.
    max_workers:
        Thread pool size. Calls are I/O-bound (~5-10s each), so 8 is a sane
        default. Drop this if you hit Bedrock throttling.
    limit:
        Optional cap on the number of images to process (handy for smoke tests).
    resume:
        If ``True`` (default), skip images already present in ``output_path``
        with a successful record.
    images:
        Optional explicit iterable of paths, overriding the directory walk.

    Returns
    -------
    Path
        The output JSONL path.
    """
    image_list = list(images) if images is not None else _discover_images(ads_root)
    if limit is not None:
        image_list = image_list[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = _load_processed_paths(output_path) if resume else set()
    pending = [p for p in image_list if _canonical_path(p) not in already_done]

    print(
        f"Found {len(image_list)} image(s); "
        f"{len(already_done)} already processed; "
        f"{len(pending)} to invoke."
    )
    if not pending:
        return output_path

    full_prompt = (
        build_categorization_prompt(prompt, categories_path=categories_file)
        if categories_file is not None
        else prompt
    )
    client = _build_client(region=region)

    with output_path.open("a") as out, ThreadPoolExecutor(max_workers) as pool:
        futures = {
            pool.submit(
                invoke_one,
                p,
                client=client,
                agent_arn=agent_arn,
                prompt=full_prompt,
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ads-root", type=Path, default=DEFAULT_ADS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--agent-arn", default=DEFAULT_AGENT_ARN)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--categories-file", type=Path, default=DEFAULT_CATEGORIES_PATH,
        help=(
            "Path to the canonical category list (one per line). "
            "Pass an empty string to send the prompt without any list."
        ),
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help="Thread pool size for parallel agent invocations.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N images (useful for smoke tests).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-invoke even images already present in the output JSONL.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    cats_file = args.categories_file if str(args.categories_file) else None
    batch_invoke(
        ads_root=args.ads_root,
        output_path=args.output,
        agent_arn=args.agent_arn,
        prompt=args.prompt,
        categories_file=cats_file,
        region=args.region,
        max_workers=args.max_workers,
        limit=args.limit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
