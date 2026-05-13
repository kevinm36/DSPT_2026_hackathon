"""
Bedrock AgentCore image ranking agent (two-step).

Step 1: Classify each ad image into an IAB category (one LLM call per image).
Step 2: Score those categories against the user profile (one LLM call).

Payload:
  {
    "user_id": "U0001",
    "profile": {
      "inf": {"gender": "M", "age": "25", ...},
      "pref": {"websites": "...", ...},
      "pos_labels": ["car", "sport"],
      "neg_labels": ["baby"]
    },
    "images": [
      {"image_base64": "<b64>", "image_format": "png", "image_id": "ad_001"},
      ...
    ]
  }

Response:
  {
    "user_id": "U0001",
    "classifications": [
      {"image_id": "ad_001", "category": "Sports & Outdoors", "confidence": 0.9, "reasoning": "..."},
      ...
    ],
    "scores": [
      {"image_id": "ad_001", "category": "Sports & Outdoors", "score": 0.8, "reasoning": "..."},
      ...
    ]
  }
"""

import base64
import json
import random
from typing import Any

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

MODEL_ID = "arn:aws:bedrock:us-east-1:014498646416:inference-profile/global.anthropic.claude-sonnet-4-6"

ADS16_CATEGORIES = [
    "Clothing & Shoes", "Automotive", "Baby Products", "Health & Beauty",
    "Media (BMVD)", "Consumer Electronics", "Console & Video Games", "DIY & Tools",
    "Garden & Outdoor living", "Grocery", "Kitchen & Home", "Betting",
    "Jewellery & Watches", "Musical Instruments", "Office Products", "Pet Supplies",
    "Computer Software", "Sports & Outdoors", "Toys & Games", "Dating Sites",
]

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from strands import Agent
        _agent = Agent(model=MODEL_ID)
    return _agent


# ---------------------------------------------------------------------------
# Step 1: Classify a single image
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = (
    "You are an ad category classifier. Given this advertisement image, "
    "determine which single category from the list below best describes it.\n\n"
    "Categories: [{categories}]\n\n"
    "Return ONLY valid JSON (no markdown fences):\n"
    '{{"category": "<exact category name from list>", "confidence": <0.0-1.0>, '
    '"reasoning": "<brief explanation, 10 words max>"}}'
)


def classify_image(image_b64: str, image_format: str = "png") -> dict:
    """Classify one image into an IAB category."""
    agent = get_agent()

    categories_str = ", ".join(ADS16_CATEGORIES)
    prompt_text = CLASSIFY_PROMPT.replace("{categories}", categories_str)

    image_bytes = base64.b64decode(image_b64)
    content_blocks = [
        {"text": prompt_text},
        {
            "image": {
                "format": image_format,
                "source": {"bytes": image_bytes},
            },
        },
    ]

    result = agent(content_blocks)

    # Extract text from response
    msg = result.message
    if isinstance(msg, dict):
        content = msg.get("content", [])
        raw = content[0]["text"] if content else ""
    elif isinstance(msg, str):
        raw = msg
    else:
        raw = str(msg)

    # Parse JSON response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1:
        return json.loads(raw[start:end])
    return {"category": "unknown", "confidence": 0.0, "reasoning": "parse error"}


# ---------------------------------------------------------------------------
# Step 2: Score classified categories against user profile
# ---------------------------------------------------------------------------

def build_scoring_prompt(profile: dict, image_categories: list[dict]) -> str:
    """Build prompt to score the classified ad categories against user profile."""
    inf = profile.get("inf", {})
    pref = profile.get("pref", {})
    pos = ", ".join(profile.get("pos_labels", [])) or "none"
    neg = ", ".join(profile.get("neg_labels", [])) or "none"

    demo_parts = []
    if inf.get("age"):
        demo_parts.append(f"Age: {inf['age']}")
    if inf.get("gender"):
        demo_parts.append(f"Gender: {inf['gender']}")
    if inf.get("job"):
        demo_parts.append(f"Job: {inf['job']}")
    if inf.get("income"):
        demo_parts.append(f"Income level: {inf['income']}")
    if inf.get("timepass"):
        demo_parts.append(f"Hobbies: {inf['timepass']}")
    if inf.get("fave_sports"):
        demo_parts.append(f"Sports: {inf['fave_sports']}")
    demo_str = ", ".join(demo_parts) or "unknown"

    pref_parts = []
    if pref.get("websites"):
        pref_parts.append(f"Websites: {pref['websites']}")
    if pref.get("music"):
        pref_parts.append(f"Music: {pref['music']}")
    if pref.get("movies"):
        pref_parts.append(f"Movies: {pref['movies']}")
    if pref.get("tv"):
        pref_parts.append(f"TV: {pref['tv']}")
    if pref.get("books"):
        pref_parts.append(f"Books: {pref['books']}")
    pref_str = "; ".join(pref_parts) or "none"

    # Build the ads list with image_id and category
    ads_list = "\n".join(
        f"  - {item['image_id']}: {item['category']}"
        for item in image_categories
    )

    return (
        f"Score how well each ad matches the user profile based on its category.\n\n"
        f"User Profile:\n"
        f"- Demographics: {demo_str}\n"
        f"- Preferences: {pref_str}\n"
        f"- Liked image content: {pos}\n"
        f"- Disliked image content: {neg}\n\n"
        f"Ads to score:\n{ads_list}\n\n"
        f"Scale: -1 (user would strongly dislike), "
        f"0 (user is indifferent or uncertain), "
        f"1 (strong match with user interests).\n\n"
        f"Rules:\n"
        f"- Score in increments of 0.1\n"
        f"- Default to 0 if uncertain\n\n"
        f"For each ad, provide:\n"
        f"1. A relevance score between -1 and 1 (inclusive)\n"
        f"2. A brief reasoning (less than 10 words)\n"
        f"Return a JSON array:\n"
        f'[{{"image_id": "<id>", "category": "<category>", "score": <float>, '
        f'"reasoning": "<10 words max>"}}, ...]'
    )


# ---------------------------------------------------------------------------
# Agent entrypoint
# ---------------------------------------------------------------------------

@app.entrypoint
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    user_id = payload["user_id"]
    profile = payload["profile"]
    images = payload["images"]

    # Step 1: Classify each image
    classifications = []
    for img in images:
        image_id = img.get("image_id", f"img_{len(classifications)}")
        result = classify_image(
            img["image_base64"],
            img.get("image_format", "png"),
        )
        result["image_id"] = image_id
        classifications.append(result)

    # Step 2: Score the classified categories against user profile
    image_categories = [
        {"image_id": c["image_id"], "category": c["category"]}
        for c in classifications
    ]
    prompt = build_scoring_prompt(profile, image_categories)

    agent = get_agent()
    result = agent(prompt)

    msg = result.message
    if isinstance(msg, dict):
        content = msg.get("content", [])
        raw = content[0]["text"] if content else ""
    elif isinstance(msg, str):
        raw = msg
    else:
        raw = str(msg)

    start = raw.find("[")
    end = raw.rfind("]") + 1
    scores = json.loads(raw[start:end]) if start != -1 else []

    return {
        "user_id": user_id,
        "classifications": classifications,
        "scores": scores,
    }


if __name__ == "__main__":
    app.run()
