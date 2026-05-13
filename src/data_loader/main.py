"""End-to-end ADS-16 user-vector pipeline.

Pipeline stages:

1. **Discover** all 300 ad images across the two ADS-16 release parts, in
   canonical category order (folder ``1`` -> folder ``20``, image ``1`` ->
   ``15``).
2. **Invoke** the deployed AgentCore image agent on every image, appending the
   IAB tier-2 category list to the prompt. Responses are written to
   ``Data/ads16_agent_responses.jsonl`` (resumable).
3. **Multi-hot** the comma-separated category responses against the canonical
   IAB list -> ``Data/ads16_multihot.csv``.
4. **Weight** each user's per-image ratings against the multi-hot matrix to
   produce one weighted profile vector per user (``user_vector =
   sum_i rating_i * multihot_i``) -> ``Data/user_vectors.csv``.

Run the whole thing against the already-deployed default runtime::

    python -m src.data_loader.main

One-shot lifecycle (deploy fresh runtime, run pipeline, delete)::

    python -m src.data_loader.main --deploy
    python -m src.data_loader.main --deploy --no-cleanup        # leave runtime running
    python -m src.data_loader.main --deploy --agent-name my_run # custom runtime name

Skip stages you've already completed::

    python -m src.data_loader.main --skip invoke      # reuse existing JSONL
    python -m src.data_loader.main --skip invoke multihot

Single-user smoke test::

    python -m src.data_loader.main --users U0001 --limit 6
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.agentcore import delete_agent, deploy_agent

from .ads16_processor import ADS16DataProcessor
from .agent_processing import (
    DEFAULT_CATEGORIES_PATH,
    batch_invoke,
    load_categories,
)
from .agent_processing.batch_invoke_ads import DEFAULT_AGENT_ARN
from .multihot_from_responses import responses_to_multihot


REPO_ROOT = Path(__file__).resolve().parents[2]
ADS_ROOTS: list[Path] = [
    REPO_ROOT / "Data/ADS-16/ADS16_Benchmark_part1/Ads",
    REPO_ROOT / "Data/ADS-16/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Ads/Ads",
]
CORPUS_ROOTS: list[Path] = [
    REPO_ROOT / "Data/ADS-16/ADS16_Benchmark_part1/Corpus",
    REPO_ROOT / "Data/ADS-16/ADS16_Benchmark_part2/ADS16_Benchmark_part2/Corpus/Corpus",
]

RESPONSES_PATH = REPO_ROOT / "Data/ads16_agent_responses_t1.jsonl"
MULTIHOT_PATH = REPO_ROOT / "Data/ads16_multihot_t1.csv"
USER_VECTORS_PATH = REPO_ROOT / "Data/user_vectors_t1.csv"

DEFAULT_AGENT_NAME = "ads16_image_agent"
# Same handler shape as basic_img_agent_src/my_agent.py: accepts
# {prompt, image_base64, image_format} and returns {"result": ...}.
DEFAULT_SYSTEM_PROMPT = (
    "You are an image classifier. Read the user's instructions carefully and "
    "respond exactly as requested."
)

NUM_CATEGORIES = 20
IMAGES_PER_CATEGORY = 15
NUM_IMAGES = NUM_CATEGORIES * IMAGES_PER_CATEGORY  # 300

# image_id scheme used by both the multi-hot CSV and the rating-weighting step.
# c is the rating column index 0..19; folders on disk are named "1".."20".
IMAGE_ID_FOR = lambda c, i: f"{c + 1}_{i + 1}"  # noqa: E731
IMAGE_ID_FORMAT = "{category}_{image_id}"


# ---------------------------------------------------------------------------
# Stage 1: discover images in canonical order
# ---------------------------------------------------------------------------
def discover_images(ads_roots: Iterable[Path]) -> list[Path]:
    """Return the 300 ADS-16 image paths in canonical (folder 1..20) order.

    Each ads root is expected to contain numeric subfolders (e.g. ``Ads/1``,
    ``Ads/2``, ...). Folders are merged across roots and sorted numerically;
    files within a folder are sorted by their numeric stem so that ``2.png``
    comes before ``10.png`` (lexicographic sort would invert that).
    """
    folders: dict[int, Path] = {}
    for root in ads_roots:
        if not root.is_dir():
            print(f"  [warn] ads root missing, skipping: {root}", file=sys.stderr)
            continue
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            if not sub.name.isdigit():
                continue
            cat = int(sub.name)
            if cat in folders:
                raise ValueError(
                    f"Duplicate category folder {cat!r} in both "
                    f"{folders[cat]} and {sub}"
                )
            folders[cat] = sub

    ordered: list[Path] = []
    for cat in sorted(folders):
        files = sorted(
            folders[cat].glob("*.png"),
            key=lambda p: int(re.match(r"\d+", p.stem).group(0)),
        )
        # The rating CSVs encode exactly IMAGES_PER_CATEGORY ratings per cat;
        # ADS-16 ships at least one folder with an extra image (folder 1 has
        # 16 PNGs). Truncate so indexing stays aligned with the rating vector.
        if len(files) > IMAGES_PER_CATEGORY:
            print(
                f"  [warn] folder {folders[cat]} has {len(files)} images; "
                f"truncating to first {IMAGES_PER_CATEGORY}.",
                file=sys.stderr,
            )
            files = files[:IMAGES_PER_CATEGORY]
        ordered.extend(files)
    return ordered


# ---------------------------------------------------------------------------
# Stage 4: weight a user's ratings against the multi-hot matrix
# ---------------------------------------------------------------------------
def discover_users(corpus_roots: Iterable[Path]) -> dict[str, Path]:
    """Return ``{user_id: rating_csv_path}`` for every user across both parts."""
    users: dict[str, Path] = {}
    for root in corpus_roots:
        if not root.is_dir():
            print(f"  [warn] corpus root missing, skipping: {root}", file=sys.stderr)
            continue
        for user_dir in sorted(root.iterdir()):
            if not user_dir.is_dir():
                continue
            rt = user_dir / f"{user_dir.name}-RT.csv"
            if rt.is_file():
                users[user_dir.name] = rt
    return users


RATING_NORMS = {"none", "center", "zscore"}
VECTOR_NORMS = {"none", "l2", "l1"}


def _normalize_ratings(ratings: np.ndarray, mode: str) -> np.ndarray:
    """Per-user rating normalization. ``ratings`` is a flat (300,) vector."""
    r = ratings.astype(np.float64)
    if mode == "none":
        return r
    if mode == "center":
        return r - r.mean()
    if mode == "zscore":
        std = r.std()
        return (r - r.mean()) / std if std > 0 else r - r.mean()
    raise ValueError(f"Unknown rating-norm {mode!r}; expected one of {RATING_NORMS}")


def _normalize_vector(vec: np.ndarray, mode: str) -> np.ndarray:
    """Per-user output-vector normalization."""
    if mode == "none":
        return vec
    if mode == "l2":
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    if mode == "l1":
        n = np.abs(vec).sum()
        return vec / n if n > 0 else vec
    raise ValueError(f"Unknown vector-norm {mode!r}; expected one of {VECTOR_NORMS}")


def weight_user(
    rating_csv: Path,
    multihot_csv: Path,
    *,
    feature_columns: list[str],
    rating_norm: str = "center",
    vector_norm: str = "l2",
) -> pd.Series:
    """Compute one user's weighted profile vector as a Series over IAB cats.

    Pipeline:
      1. Load ratings and the multi-hot matrix via ``ADS16DataProcessor``.
      2. Apply ``rating_norm`` per user (e.g. mean-centering removes the
         "everyone has a different baseline" bias).
      3. Compute ``user_vector = ratings @ multihot``.
      4. Apply ``vector_norm`` (e.g. L2 makes vectors comparable for cosine
         similarity across users).
    """
    proc = ADS16DataProcessor(
        rating_csv_path=rating_csv,
        multihot_csv_path=multihot_csv,
        image_id_column="image_id",
        image_id_for=IMAGE_ID_FOR,
        feature_columns=feature_columns,
    )
    raw_ratings = proc.load_ratings()
    multihot = proc.load_multihot()

    image_ids = [
        IMAGE_ID_FOR(c, i)
        for c in range(NUM_CATEGORIES)
        for i in range(IMAGES_PER_CATEGORY)
    ]
    aligned = multihot.loc[image_ids].to_numpy(dtype=np.float64)

    ratings = _normalize_ratings(raw_ratings, rating_norm)
    user_vector = ratings @ aligned
    user_vector = _normalize_vector(user_vector, vector_norm)

    user_id = rating_csv.stem.split("-")[0]
    return pd.Series(user_vector, index=feature_columns, name=user_id)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(
    *,
    skip: set[str],
    users: Optional[list[str]],
    limit: Optional[int],
    max_workers: int,
    responses_path: Path,
    multihot_path: Path,
    user_vectors_path: Path,
    rating_norm: str = "center",
    vector_norm: str = "l2",
    deploy_first: bool = False,
    cleanup: bool = True,
    agent_name: str = DEFAULT_AGENT_NAME,
    agent_arn: Optional[str] = None,
) -> None:
    # ----- Stage 0 (optional): deploy a fresh AgentCore runtime
    deployed_runtime: Optional[dict] = None
    if deploy_first:
        if "invoke" in skip:
            raise ValueError(
                "--deploy is incompatible with --skip invoke: deploying a runtime "
                "we never call wastes time and money. Drop one of the two flags."
            )
        print("=" * 72)
        print("Stage 0: deploy AgentCore runtime")
        print("=" * 72)
        deployed_runtime = deploy_agent(
            name=agent_name,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        print(f"  deployed: {deployed_runtime['name']}")
        print(f"    id:     {deployed_runtime['id']}")
        print(f"    arn:    {deployed_runtime['arn']}")
        print(f"    status: {deployed_runtime['status']}")
        agent_arn = deployed_runtime["arn"]

    # The rest of the pipeline runs inside try/finally so a freshly deployed
    # runtime gets torn down even if a later stage blows up.
    try:
        # ----- Stage 1: discover images
        print()
        print("=" * 72)
        print("Stage 1: discover images")
        print("=" * 72)
        images = discover_images(ADS_ROOTS)
        print(f"Discovered {len(images)} images across {len(ADS_ROOTS)} roots.")
        if images:
            print(f"  first: {images[0].relative_to(REPO_ROOT)}")
            print(f"  last:  {images[-1].relative_to(REPO_ROOT)}")
        if len(images) != NUM_IMAGES:
            print(
                f"  [warn] expected {NUM_IMAGES} images (20 categories x 15), "
                f"got {len(images)}.",
                file=sys.stderr,
            )

        images_for_invoke = images[:limit] if limit else images

        # ----- Stage 2: batch invoke
        print()
        print("=" * 72)
        print("Stage 2: batch-invoke agent")
        print("=" * 72)
        if "invoke" in skip:
            print(f"  [skip] reusing existing {responses_path}")
            if not responses_path.is_file():
                raise FileNotFoundError(
                    f"--skip invoke requested but {responses_path} does not exist."
                )
        else:
            invoke_kwargs = dict(
                output_path=responses_path,
                categories_file=DEFAULT_CATEGORIES_PATH,
                max_workers=max_workers,
                images=images_for_invoke,
            )
            if agent_arn:
                invoke_kwargs["agent_arn"] = agent_arn
            batch_invoke(**invoke_kwargs)

        # ----- Stage 3: multi-hot
        print()
        print("=" * 72)
        print("Stage 3: multi-hot from responses")
        print("=" * 72)
        if "multihot" in skip:
            print(f"  [skip] reusing existing {multihot_path}")
            if not multihot_path.is_file():
                raise FileNotFoundError(
                    f"--skip multihot requested but {multihot_path} does not exist."
                )
        else:
            df, unmatched = responses_to_multihot(
                responses_path,
                categories_path=DEFAULT_CATEGORIES_PATH,
                image_id_format=IMAGE_ID_FORMAT,
            )
            multihot_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(multihot_path, index=False)
            print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {multihot_path}")
            if unmatched:
                top = unmatched.most_common(5)
                print(
                    f"  unmatched response tokens: {sum(unmatched.values())} "
                    f"({len(unmatched)} unique). Top: {top}"
                )

        # ----- Stage 4: weight user ratings
        print()
        print("=" * 72)
        print("Stage 4: weight ratings -> user vectors")
        print("=" * 72)
        if "weight" in skip:
            print(f"  [skip] not building {user_vectors_path}")
            return

        feature_columns = load_categories(DEFAULT_CATEGORIES_PATH)
        available_users = discover_users(CORPUS_ROOTS)
        if users:
            missing = [u for u in users if u not in available_users]
            if missing:
                raise FileNotFoundError(
                    f"No rating CSV found for users: {missing}. "
                    f"Available: {sorted(available_users)[:5]}..."
                )
            target = {u: available_users[u] for u in users}
        else:
            target = available_users
        print(
            f"Weighting ratings for {len(target)} user(s) "
            f"(rating-norm={rating_norm}, vector-norm={vector_norm})."
        )

        rows: list[pd.Series] = []
        failures: list[tuple[str, str]] = []
        for uid, rt_path in sorted(target.items()):
            try:
                row = weight_user(
                    rt_path, multihot_path,
                    feature_columns=feature_columns,
                    rating_norm=rating_norm,
                    vector_norm=vector_norm,
                )
            except Exception as exc:
                failures.append((uid, repr(exc)))
                print(f"  [error] {uid}: {exc}")
                continue
            rows.append(row)
            print(
                f"  ok   {uid}: nnz={int((row != 0).sum())}, "
                f"min={row.min():+.3f}, max={row.max():+.3f}"
            )

        if rows:
            out = pd.DataFrame(rows)
            out.index.name = "user_id"
            user_vectors_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(user_vectors_path)
            print(f"Wrote {out.shape[0]} user vector(s) x "
                  f"{out.shape[1]} categories to {user_vectors_path}")
        else:
            print("  no user vectors produced.", file=sys.stderr)

        if failures:
            print(f"\n{len(failures)} user(s) failed:", file=sys.stderr)
            for uid, err in failures:
                print(f"  {uid}: {err}", file=sys.stderr)
    finally:
        # ----- Stage 5 (optional): tear down the runtime we deployed
        if deployed_runtime is not None:
            print()
            print("=" * 72)
            print("Stage 5: cleanup AgentCore runtime")
            print("=" * 72)
            if cleanup:
                try:
                    deleted = delete_agent(deployed_runtime["name"])
                    print(f"  deleted {deployed_runtime['name']}: {deleted}")
                except Exception as exc:
                    # Don't mask whatever raised in the try block (if any).
                    print(
                        f"  [warn] failed to delete {deployed_runtime['name']}: "
                        f"{exc!r}",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"  [skip] --no-cleanup; runtime left running: "
                    f"{deployed_runtime['id']}"
                )
                print(
                    f"    delete later: python -c \"from src.agentcore import "
                    f"delete_agent; delete_agent('{deployed_runtime['name']}')\""
                )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip", nargs="+", default=[],
        choices=["invoke", "multihot", "weight"],
        help="Skip one or more pipeline stages and reuse prior artifacts.",
    )
    parser.add_argument(
        "--users", nargs="+", default=None,
        help="User IDs to weight (e.g. U0001 U0002). Default: every user found.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only invoke the first N images (smoke test).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Thread pool size for parallel agent invocations.",
    )
    parser.add_argument("--responses", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--multihot", type=Path, default=MULTIHOT_PATH)
    parser.add_argument("--user-vectors", type=Path, default=USER_VECTORS_PATH)
    parser.add_argument(
        "--rating-norm", choices=sorted(RATING_NORMS), default="center",
        help=(
            "Per-user rating normalization applied BEFORE r @ M. "
            "'center' (default): subtract per-user mean (removes baseline "
            "bias - recommended). 'zscore': also divide by std. 'none': raw."
        ),
    )
    parser.add_argument(
        "--vector-norm", choices=sorted(VECTOR_NORMS), default="l2",
        help=(
            "Per-user vector normalization applied AFTER r @ M. "
            "'l2' (default): unit length (recommended for cosine similarity). "
            "'l1': sums to 1. 'none': raw magnitudes."
        ),
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help=(
            "Deploy a fresh AgentCore runtime before stage 2 and (by default) "
            "delete it after stage 4. Without this flag the pipeline calls the "
            "existing runtime at --agent-arn (default: the hardcoded "
            "DEFAULT_AGENT_ARN in batch_invoke_ads.py)."
        ),
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help=(
            "Only meaningful with --deploy: leave the deployed runtime running "
            "after the pipeline finishes (matches tests/test_deploy.py)."
        ),
    )
    parser.add_argument(
        "--agent-name", default=DEFAULT_AGENT_NAME,
        help=(
            f"Runtime name to deploy/delete when --deploy is set. "
            f"Default: {DEFAULT_AGENT_NAME!r}."
        ),
    )
    parser.add_argument(
        "--agent-arn", default=None,
        help=(
            "Override the AgentCore runtime ARN passed to batch_invoke. "
            "Ignored when --deploy is set (the freshly-deployed ARN wins). "
            f"Default: the hardcoded ARN in batch_invoke_ads.py "
            f"({DEFAULT_AGENT_ARN!r})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    run(
        skip=set(args.skip),
        users=args.users,
        limit=args.limit,
        max_workers=args.max_workers,
        responses_path=args.responses,
        multihot_path=args.multihot,
        user_vectors_path=args.user_vectors,
        rating_norm=args.rating_norm,
        vector_norm=args.vector_norm,
        deploy_first=args.deploy,
        cleanup=not args.no_cleanup,
        agent_name=args.agent_name,
        agent_arn=args.agent_arn,
    )


if __name__ == "__main__":
    main()
