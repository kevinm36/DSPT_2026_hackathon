"""Ad-design / sentiment feature extraction for ADS-16.

Models ad creatives by their **design characteristics** (visual clutter, focal
point, brand prominence, ...) and **sentiment** (emotion valence, perceived
credibility, ...) instead of by IAB content category. Pairs naturally with
user personality (B5) and demographic features for "what kind of ad design
appeals to whom" modeling.

Recommended workflow
--------------------

1. ``schema.py``   - Inspect / refine the field list. Run a quick syntax
                     check::

                         python -c "from src.ad_design.schema import \
                             FIELD_NAMES; print(len(FIELD_NAMES), FIELD_NAMES)"

2. ``prompt.py``   - Generate the prompt and eyeball it. No I/O::

                         python -c "from src.ad_design.prompt import \
                             build_prompt; print(build_prompt())"

3. ``validate.py`` - **MANDATORY GATE.** Run the 20-image test-retest
                     experiment. Drop / fix any field that doesn't agree
                     with itself across two independent calls::

                         python -m src.ad_design.validate

                     Cohen's kappa < 0.4 (categorical) or Pearson r < 0.5
                     (numeric) means the field is unreliable - tighten the
                     rubric in ``schema.py`` and re-run, OR drop the field.

4. ``extract.py``  - Once validation passes, score the full 300-ad corpus::

                         python -m src.ad_design.extract

                     Resumable - safe to interrupt and rerun. Writes JSONL.

5. ``parse.py``    - Flatten JSONL -> CSV indexed by ``image_id``::

                         python -m src.ad_design.parse

Important: the deployed agent (``basic_img_agent_src/my_agent.py``) does not
currently honor the ``temperature`` field in the payload. ``extract.py``
sends ``temperature=0`` regardless, but to actually pin it on the model see
the agent-side change documented in ``extract.py``'s module docstring.
"""

from .extract import batch_invoke
from .parse import parse_record, responses_to_features
from .prompt import build_prompt
from .schema import FIELD_DEFS, FIELD_NAMES, FieldDef, validate_response
from .validate import build_consistency_report, pick_validation_images

__all__ = [
    "batch_invoke",
    "build_prompt",
    "build_consistency_report",
    "parse_record",
    "pick_validation_images",
    "responses_to_features",
    "validate_response",
    "FIELD_DEFS",
    "FIELD_NAMES",
    "FieldDef",
]
