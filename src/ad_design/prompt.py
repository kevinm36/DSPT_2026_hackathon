"""Prompt builder for ad-design / sentiment feature extraction.

The prompt has three blocks:

1. Role + framing.
2. Output contract: strict JSON inside ``<json>...</json>`` tags. The tags are
   what makes responses parseable even if Claude prepends "Here is the JSON:"
   or wraps the payload in markdown.
3. Schema with per-field rubric (anchors for 1-10 scales) and enum options.

Few-shot calibration examples are supported via ``examples=`` but default to
none. The recommended workflow is: run extraction without examples on ~5-10
deliberately diverse ads, hand-pick 2-3 well-scored ones, then re-run with
those as examples to anchor the LLM's scale across the full corpus.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .schema import FIELD_DEFS, FieldDef


SYSTEM_PREAMBLE = (
    "You are an expert advertising creative analyst. Your task is to score a "
    "single ad image on a fixed set of design and sentiment dimensions. Be "
    "consistent: identical images must always receive identical scores. Use "
    "the rubric anchors literally - do not invent your own scoring philosophy."
)

OUTPUT_CONTRACT = (
    "Output format - strict:\n"
    "  - Wrap your response in <json>...</json> tags.\n"
    "  - Inside the tags, emit a single JSON object with EXACTLY the keys "
    "below, in the order shown.\n"
    "  - Do NOT include any other text, prose, markdown, code fences, or "
    "commentary.\n"
    "  - Use the exact spelling/casing for enum values.\n"
    "  - Booleans are lowercase JSON true / false.\n"
    "  - Integers are bare digits (no quotes)."
)


def _render_field(field_def: FieldDef) -> str:
    """One-line spec + rubric for a single field."""
    if field_def.type == "bool":
        spec = f'  "{field_def.name}": <bool>,  # {field_def.description}'
    elif field_def.type == "int":
        spec = (
            f'  "{field_def.name}": <int {field_def.int_min}-{field_def.int_max}>, '
            f' # {field_def.description}'
        )
        if field_def.rubric:
            spec += "\n" + "\n".join(f"      - {anchor}" for anchor in field_def.rubric)
    elif field_def.type == "enum":
        opts = " | ".join(f'"{v}"' for v in field_def.enum_values)
        spec = f'  "{field_def.name}": {opts},  # {field_def.description}'
    else:
        raise ValueError(f"unknown type {field_def.type}")
    return spec


def _render_schema() -> str:
    """Render the full schema as the JSON template the LLM should fill in."""
    body = "\n".join(_render_field(f) for f in FIELD_DEFS)
    return "{\n" + body + "\n}"


def _render_examples(examples: Sequence[dict]) -> str:
    """Render few-shot examples as ``<json>...</json>`` blocks.

    Each example is a fully-scored dict matching ``FIELD_NAMES``. Ideally
    these come from hand-validated runs on deliberately diverse ads (one
    very clean / minimal, one mid, one cluttered/spammy).
    """
    import json
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Example {i}:\n<json>\n"
            f"{json.dumps(ex, indent=2)}\n</json>"
        )
    return (
        "Calibrated examples (match the scale of your scores to these):\n\n"
        + "\n\n".join(blocks)
    )


def build_prompt(examples: Optional[Sequence[dict]] = None) -> str:
    """Build the full prompt string sent to the agent for one image.

    Parameters
    ----------
    examples:
        Optional sequence of fully-scored example dicts. If provided, they
        are inlined as anchored few-shot demonstrations - the single most
        effective lever for scale consistency on subjective fields.
    """
    parts = [
        SYSTEM_PREAMBLE,
        "",
        OUTPUT_CONTRACT,
        "",
        "Schema (template to fill in, with rubric anchors):",
        _render_schema(),
    ]
    if examples:
        parts += ["", _render_examples(examples)]
    parts += [
        "",
        "Now score the attached image. Respond with ONLY the <json>...</json> "
        "block."
    ]
    return "\n".join(parts)
