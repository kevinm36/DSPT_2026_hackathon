"""Schema for ad-design / sentiment feature extraction.

Single source of truth for which fields the LLM scores per ad, what their
types/ranges are, and what each field means. Used by ``prompt`` to build the
LLM instruction and by ``parse`` to validate / flatten responses.

Refinement vs. the colleague's original 20-field draft:

  - Subjective 1-10 scales that lack an objective rubric are downgraded to
    3-bin enums ("low" / "medium" / "high"). Specifically:
      * ``aesthetic_score``       -> ``design_quality``       enum
      * ``perceived_credibility`` -> ``perceived_credibility`` enum
      * ``spamminess_score``      -> ``spamminess``           enum
    Reason: an LLM's 7-vs-8 distinction on these dimensions is essentially
    noise across calls. The 3-bin version is far more reproducible.
  - ``word_count`` is downgraded to ``word_count_bin`` (5-bin enum). LLMs are
    bad at exact counts; a coarse bin is what's actually reliable.
  - Every numeric 1-10 field that survived gets a 3-anchor rubric in
    ``prompt.py``. This is the single largest lever for inter-call consistency
    on subjective scales.

All other fields from the original draft are kept as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


FieldType = Literal["int", "bool", "enum"]


@dataclass(frozen=True)
class FieldDef:
    name: str
    type: FieldType
    description: str
    # for type="int": inclusive range
    int_min: Optional[int] = None
    int_max: Optional[int] = None
    # for type="enum": allowed string values
    enum_values: Optional[tuple[str, ...]] = None
    # 3-line rubric (low / mid / high anchor) - shown in the prompt for ints.
    # Keep each anchor short.
    rubric: Optional[tuple[str, str, str]] = None


# Order matches the colleague's original draft for readability.
FIELD_DEFS: tuple[FieldDef, ...] = (
    # ----- visual composition -----
    FieldDef(
        name="design_quality",
        type="enum",
        description="Overall design quality (was 'aesthetic_score' 1-10).",
        enum_values=("low", "medium", "high"),
    ),
    FieldDef(
        name="visual_clutter",
        type="int", int_min=1, int_max=10,
        description="Amount of distracting / unnecessary elements.",
        rubric=(
            "1 = single subject, plenty of negative space",
            "5 = multiple elements but a clear hierarchy",
            "10 = chaotic, no clear visual order",
        ),
    ),
    FieldDef(
        name="focal_point_presence",
        type="int", int_min=1, int_max=10,
        description="Clarity of a single main subject.",
        rubric=(
            "1 = no clear focal point, eye wanders",
            "5 = a focal point exists but competes with other elements",
            "10 = one subject obviously dominates",
        ),
    ),
    FieldDef(
        name="contrast_level",
        type="int", int_min=1, int_max=10,
        description="Strength of contrast (color, brightness) drawing attention.",
        rubric=(
            "1 = flat, low-contrast, washed out",
            "5 = moderate contrast, comfortable to view",
            "10 = very high contrast, bold colors / strong lighting",
        ),
    ),
    FieldDef(
        name="visual_saliency_score",
        type="int", int_min=1, int_max=10,
        description="Likelihood the image captures immediate attention.",
        rubric=(
            "1 = easy to ignore, scrolls past",
            "5 = noticeable but not arresting",
            "10 = stops the eye, demands attention",
        ),
    ),

    # ----- subject -----
    FieldDef(
        name="primary_subject_type",
        type="enum",
        description="The dominant kind of subject in the image.",
        enum_values=("product", "person", "scene", "text-only", "mixed"),
    ),
    FieldDef(
        name="human_presence",
        type="bool",
        description="True if at least one human (or clear human silhouette) is visible.",
    ),

    # ----- product / brand -----
    FieldDef(
        name="product_visibility",
        type="int", int_min=1, int_max=10,
        description="How clearly the product is shown.",
        rubric=(
            "1 = no product visible, or only implied",
            "5 = product visible but not the main focus",
            "10 = product is the dominant element, clearly identifiable",
        ),
    ),
    FieldDef(
        name="usage_context",
        type="enum",
        description="How the product is contextualized.",
        enum_values=("in-use", "standalone", "lifestyle", "abstract", "none"),
    ),
    FieldDef(
        name="brand_prominence",
        type="int", int_min=1, int_max=10,
        description="How visible / dominant the brand identity is.",
        rubric=(
            "1 = no brand cues visible",
            "5 = brand visible but not emphasized",
            "10 = brand dominates the composition",
        ),
    ),
    FieldDef(
        name="logo_present",
        type="bool",
        description="True if a logo or brand mark is clearly visible.",
    ),

    # ----- text -----
    FieldDef(
        name="word_count_bin",
        type="enum",
        description="Approximate count of words rendered in the image (was integer).",
        enum_values=("0", "1-5", "6-15", "16-40", "40+"),
    ),
    FieldDef(
        name="text_density",
        type="int", int_min=1, int_max=10,
        description="Text relative to available space.",
        rubric=(
            "1 = almost no text",
            "5 = moderate text, balanced with imagery",
            "10 = text-heavy, image is mostly type",
        ),
    ),
    FieldDef(
        name="readability",
        type="int", int_min=1, int_max=10,
        description="Ease of reading the text at a glance.",
        rubric=(
            "1 = unreadable (too small / low contrast / cluttered)",
            "5 = readable with effort",
            "10 = instantly readable, clear typography",
        ),
    ),

    # ----- messaging -----
    FieldDef(
        name="value_proposition_present",
        type="bool",
        description="True if a clear benefit or message is communicated.",
    ),
    FieldDef(
        name="cta_present",
        type="bool",
        description="True if a call-to-action exists (e.g. 'Shop now', 'Sign up').",
    ),
    FieldDef(
        name="offer_present",
        type="bool",
        description="True if a promotion / discount / time-limited offer is shown.",
    ),

    # ----- emotion / trust -----
    FieldDef(
        name="emotion_valence",
        type="enum",
        description="Overall emotional tone.",
        enum_values=("positive", "neutral", "negative"),
    ),
    FieldDef(
        name="perceived_credibility",
        type="enum",
        description="Overall trustworthiness suggested (was 'perceived_credibility' 1-10).",
        enum_values=("low", "medium", "high"),
    ),
    FieldDef(
        name="spamminess",
        type="enum",
        description="Degree of spam-like / aggressive design (was 'spamminess_score' 1-10).",
        enum_values=("low", "medium", "high"),
    ),
)


FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELD_DEFS)
FIELDS_BY_NAME: dict[str, FieldDef] = {f.name: f for f in FIELD_DEFS}


def validate_response(d: dict) -> list[str]:
    """Return a list of human-readable issues with a parsed response dict.

    Empty list = response matches the schema. Used by ``parse.py`` to flag
    rows worth manual inspection before they go into the feature DataFrame.
    """
    issues: list[str] = []
    for field_def in FIELD_DEFS:
        if field_def.name not in d:
            issues.append(f"missing field: {field_def.name}")
            continue
        v = d[field_def.name]
        if field_def.type == "bool":
            if not isinstance(v, bool):
                issues.append(f"{field_def.name}: expected bool, got {type(v).__name__}={v!r}")
        elif field_def.type == "int":
            if not isinstance(v, int) or isinstance(v, bool):
                issues.append(f"{field_def.name}: expected int, got {type(v).__name__}={v!r}")
            elif not (field_def.int_min <= v <= field_def.int_max):
                issues.append(
                    f"{field_def.name}: {v} outside [{field_def.int_min}, {field_def.int_max}]"
                )
        elif field_def.type == "enum":
            if not isinstance(v, str) or v not in field_def.enum_values:
                issues.append(
                    f"{field_def.name}: {v!r} not in {list(field_def.enum_values)}"
                )
    extra = set(d.keys()) - set(FIELD_NAMES)
    if extra:
        issues.append(f"extra unexpected fields: {sorted(extra)}")
    return issues
