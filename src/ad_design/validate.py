"""Test-retest validation: how consistent is the LLM across two independent calls?

This is the gate that tells you whether the schema + prompt + temperature are
reliable enough to commit to scoring all 300 ads. The flow:

  1. Pick N (default 20) representative images, deterministically.
  2. Score each one TWICE in two independent calls (separate JSONL outputs).
  3. Per field, compute agreement between pass 1 and pass 2:
       * int 1-10 fields  -> Pearson and Spearman correlation
       * bool fields      -> Cohen's kappa, accuracy
       * enum fields      -> Cohen's kappa, accuracy
  4. Print a sortable report. Fields with low agreement (the heuristic in the
     output) are the ones to either tighten the rubric for or drop.

Run::

    # Run pass 1 + pass 2 + report (does ~40 LLM calls = ~3-5 min)
    python -m src.ad_design.validate

    # Pick 30 images instead of 20
    python -m src.ad_design.validate --n 30

    # Re-run only the report on already-scored passes
    python -m src.ad_design.validate --report-only
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score

from src.data_loader import discover_images

from .extract import DEFAULT_ADS_ROOTS, batch_invoke
from .parse import responses_to_features
from .schema import FIELD_DEFS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PASS1 = REPO_ROOT / "Data/ads16_design_features_pass1.jsonl"
DEFAULT_PASS2 = REPO_ROOT / "Data/ads16_design_features_pass2.jsonl"
DEFAULT_REPORT = REPO_ROOT / "Data/ads16_design_features_consistency.csv"


def pick_validation_images(
    ads_roots, n: int, seed: int = 0,
) -> list[Path]:
    """Pick ``n`` images deterministically across categories for the test set.

    Stratified by sub-folder name (category 1..20) so the sample spans the
    whole corpus regardless of which release part the image lives in.
    """
    all_images = discover_images(ads_roots)
    if not all_images:
        raise FileNotFoundError(f"No images discovered under {ads_roots}")
    by_cat: dict[str, list[Path]] = {}
    for p in all_images:
        by_cat.setdefault(p.parent.name, []).append(p)

    rng = random.Random(seed)
    cats = sorted(by_cat)
    picks: list[Path] = []
    # Round-robin one image per category until we hit n.
    while len(picks) < n:
        for cat in cats:
            if len(picks) >= n:
                break
            pool = by_cat[cat]
            if not pool:
                continue
            picks.append(rng.choice(pool))
    return picks


def _agreement_int(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Pearson + Spearman + mean abs diff, ignoring rows where either is NaN."""
    mask = a.notna() & b.notna()
    a, b = a[mask].astype(float), b[mask].astype(float)
    if len(a) < 2:
        return {"pearson": float("nan"), "spearman": float("nan"),
                "mean_abs_diff": float("nan"), "n": len(a)}
    if a.std() == 0 and b.std() == 0:
        # Both passes constant - perfect agreement but Pearson undefined.
        same = float((a == b).all())
        return {"pearson": same, "spearman": same,
                "mean_abs_diff": float((a - b).abs().mean()), "n": len(a)}
    return {
        "pearson": float(pearsonr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan"),
        "spearman": float(spearmanr(a, b)[0]) if a.std() > 0 and b.std() > 0 else float("nan"),
        "mean_abs_diff": float((a - b).abs().mean()),
        "n": int(len(a)),
    }


def _agreement_categorical(a: pd.Series, b: pd.Series) -> dict[str, float]:
    """Cohen's kappa + raw accuracy, ignoring rows where either is NaN."""
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"kappa": float("nan"), "accuracy": float("nan"), "n": len(a)}
    try:
        kappa = float(cohen_kappa_score(a.astype(str), b.astype(str)))
    except ValueError:
        kappa = float("nan")
    return {
        "kappa": kappa,
        "accuracy": float(accuracy_score(a.astype(str), b.astype(str))),
        "n": int(len(a)),
    }


def _verdict(field_def, score: float) -> str:
    """Heuristic label for the agreement score (used in the report)."""
    if np.isnan(score):
        return "n/a"
    if field_def.type == "int":
        # Pearson thresholds for inter-call reliability on numeric scales
        if score >= 0.85:
            return "excellent"
        if score >= 0.70:
            return "ok"
        if score >= 0.50:
            return "weak - tighten rubric"
        return "drop or recast"
    # bool / enum - use Cohen's kappa thresholds
    if score >= 0.81:
        return "excellent"
    if score >= 0.61:
        return "ok"
    if score >= 0.41:
        return "weak - tighten rubric"
    return "drop or recast"


def build_consistency_report(
    pass1_path: Path = DEFAULT_PASS1,
    pass2_path: Path = DEFAULT_PASS2,
) -> pd.DataFrame:
    """Compute per-field agreement between two parsed passes."""
    df1, _ = responses_to_features(pass1_path)
    df2, _ = responses_to_features(pass2_path)
    common = sorted(set(df1.index) & set(df2.index))
    if not common:
        raise RuntimeError(
            f"No image_ids in common between {pass1_path} and {pass2_path}. "
            "Were the same images scored in both passes?"
        )
    df1 = df1.loc[common]
    df2 = df2.loc[common]

    rows = []
    for fd in FIELD_DEFS:
        if fd.type == "int":
            metrics = _agreement_int(df1[fd.name], df2[fd.name])
            primary = metrics["pearson"]
            rows.append({
                "field": fd.name, "type": "int",
                "primary_metric": "pearson", "primary_value": primary,
                "secondary_metric": "spearman", "secondary_value": metrics["spearman"],
                "extra_metric": "mean_abs_diff", "extra_value": metrics["mean_abs_diff"],
                "n": metrics["n"], "verdict": _verdict(fd, primary),
            })
        else:
            metrics = _agreement_categorical(df1[fd.name], df2[fd.name])
            primary = metrics["kappa"]
            rows.append({
                "field": fd.name, "type": fd.type,
                "primary_metric": "kappa", "primary_value": primary,
                "secondary_metric": "accuracy", "secondary_value": metrics["accuracy"],
                "extra_metric": None, "extra_value": float("nan"),
                "n": metrics["n"], "verdict": _verdict(fd, primary),
            })

    report = pd.DataFrame(rows).set_index("field")
    return report


def run(
    *,
    ads_roots,
    n: int,
    seed: int,
    pass1_path: Path,
    pass2_path: Path,
    report_path: Path,
    report_only: bool,
    temperature: float,
    max_workers: int,
) -> None:
    if not report_only:
        picks = pick_validation_images(ads_roots, n=n, seed=seed)
        print(f"Selected {len(picks)} validation image(s) (seed={seed}).")

        print("\n=== Pass 1 ===")
        batch_invoke(
            ads_roots=ads_roots, output_path=pass1_path,
            images=picks, temperature=temperature,
            max_workers=max_workers, resume=False,
        )

        print("\n=== Pass 2 ===")
        batch_invoke(
            ads_roots=ads_roots, output_path=pass2_path,
            images=picks, temperature=temperature,
            max_workers=max_workers, resume=False,
        )

    print("\n=== Consistency report ===")
    report = build_consistency_report(pass1_path, pass2_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path)
    print(f"Wrote {report_path}")
    print()

    # Sort by primary value, NaNs at bottom.
    sorted_report = report.sort_values("primary_value", ascending=False, na_position="last")
    print(sorted_report[["type", "primary_metric", "primary_value",
                          "secondary_value", "n", "verdict"]].to_string(
        float_format=lambda v: f"{v:.3f}" if not np.isnan(v) else "  nan"
    ))

    n_drop = (report.verdict == "drop or recast").sum()
    n_weak = (report.verdict == "weak - tighten rubric").sum()
    n_ok = (report.verdict.isin(["ok", "excellent"])).sum()
    print(
        f"\nSummary: {n_ok} field(s) ok+, {n_weak} weak (tighten rubric), "
        f"{n_drop} unreliable (drop or recast)."
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ads-roots", type=Path, nargs="+", default=DEFAULT_ADS_ROOTS,
        help="One or more ads root directories. Default: both ADS-16 parts.",
    )
    p.add_argument("--n", type=int, default=20,
                   help="Number of validation images. Default 20.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pass1", type=Path, default=DEFAULT_PASS1)
    p.add_argument("--pass2", type=Path, default=DEFAULT_PASS2)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--report-only", action="store_true",
                   help="Skip both invocation passes; just rebuild the "
                        "report from existing JSONLs.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-workers", type=int, default=4,
                   help="Lower than the default 8 to keep the two passes "
                        "independent in time and reduce caching effects.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run(
        ads_roots=args.ads_roots,
        n=args.n,
        seed=args.seed,
        pass1_path=args.pass1,
        pass2_path=args.pass2,
        report_path=args.report,
        report_only=args.report_only,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
