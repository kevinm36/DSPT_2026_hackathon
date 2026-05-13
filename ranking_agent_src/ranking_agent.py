"""
Bedrock AgentCore agent that ranks candidate ad images for a user based on
IAB Content Taxonomy profile similarity, then uses Claude to explain *why*
the top images are the best match and what visual/thematic features drive
user engagement.

Payload
-------
{
  "user_profile": {
    "Automotive": 4.2,
    "Sports": 3.8,
    "Technology & Computing": 1.1,
    ...
  },
  "candidate_ads": [
    {
      "id": "Cat3_7",
      "iab_profile": {
        "Sports": 1,
        "Health & Fitness": 1,
        "Style & Fashion": 0,
        ...
      }
    },
    ...
  ],
  "top_k": 5
}

Response
--------
{
  "ranked_ads": [
    {
      "rank": 1,
      "id": "Cat3_7",
      "score": 8.0,
      "reasoning": "This ad strongly aligns with your top interests ..."
    },
    ...
  ],
  "analysis": "Overall, the user gravitates toward visually dynamic ..."
}
"""

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

MODEL_ID = (
    "arn:aws:bedrock:us-east-1:014498646416:"
    "inference-profile/global.anthropic.claude-sonnet-4-6"
)

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from strands import Agent

        _agent = Agent(
            model=MODEL_ID,
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


SYSTEM_PROMPT = """\
You are an ad-ranking analyst. You receive:
1. A user's predicted IAB Content Taxonomy preference profile — a dictionary of
   IAB Tier 1 category names to numeric affinity scores.  Higher scores mean the
   user is more interested in that category.
2. A set of candidate ad images, each with its own IAB profile (binary or
   weighted).

Your job:
- Rank the candidate ads from best to worst fit for the user.
- For each ad (especially the top picks), explain *why* it is a good (or poor)
  match.  Focus on:
  • Which IAB categories overlap between the user and the ad.
  • What visual or thematic features of the ad are likely to drive engagement
    (e.g., bright colors, action imagery, aspirational lifestyle, humor,
    celebrity presence, product close-up, emotional appeal).
  • Any cross-category synergies (e.g., a user who likes both "Sports" and
    "Style & Fashion" may engage with athleisure ads).
- After ranking, provide a short overall analysis summarizing what image
  features and category themes are most likely to resonate with this user.

Respond ONLY with valid JSON matching this schema (no markdown fences):
{
  "ranked_ads": [
    {
      "rank": <int>,
      "id": "<ad id>",
      "score": <float — the dot-product similarity you computed>,
      "reasoning": "<1-3 sentence explanation>"
    }
  ],
  "analysis": "<2-4 sentence summary of what drives engagement for this user>"
}
"""


def _compute_scores(user_profile: dict, candidate_ads: list[dict]) -> list[dict]:
    """Dot-product similarity between user profile and each ad's IAB vector."""
    scored = []
    for ad in candidate_ads:
        iab = ad.get("iab_profile", {})
        score = sum(
            user_profile.get(cat, 0.0) * val
            for cat, val in iab.items()
        )
        scored.append({**ad, "_score": round(score, 4)})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def _build_prompt(user_profile: dict, scored_ads: list[dict], top_k: int) -> str:
    """Build the text prompt sent to Claude for reasoning."""
    profile_lines = "\n".join(
        f"  {cat}: {val}"
        for cat, val in sorted(user_profile.items(), key=lambda x: -x[1])
    )

    ad_lines = []
    for i, ad in enumerate(scored_ads[:top_k], 1):
        iab_str = ", ".join(
            cat for cat, val in ad.get("iab_profile", {}).items() if val
        )
        ad_lines.append(
            f"  #{i}  id={ad['id']}  score={ad['_score']}  "
            f"categories=[{iab_str}]"
        )

    remaining_lines = []
    for ad in scored_ads[top_k:]:
        iab_str = ", ".join(
            cat for cat, val in ad.get("iab_profile", {}).items() if val
        )
        remaining_lines.append(
            f"  id={ad['id']}  score={ad['_score']}  categories=[{iab_str}]"
        )

    prompt = (
        f"User IAB preference profile:\n{profile_lines}\n\n"
        f"Top {top_k} candidate ads (pre-sorted by dot-product score):\n"
        + "\n".join(ad_lines)
    )
    if remaining_lines:
        prompt += (
            f"\n\nRemaining {len(remaining_lines)} ads "
            f"(lower score, included for context):\n"
            + "\n".join(remaining_lines)
        )

    prompt += (
        "\n\nRank these ads for the user. For the top ads, explain what "
        "visual/thematic features would drive engagement. Then give an overall "
        "analysis of what image features resonate with this user profile."
    )
    return prompt


@app.entrypoint
def invoke(payload):
    import json as _json

    agent = get_agent()

    user_profile = payload.get("user_profile", {})
    candidate_ads = payload.get("candidate_ads", [])
    top_k = payload.get("top_k", 5)

    if not user_profile:
        return {"error": "user_profile is required"}
    if not candidate_ads:
        return {"error": "candidate_ads list is required"}

    scored_ads = _compute_scores(user_profile, candidate_ads)

    prompt = _build_prompt(user_profile, scored_ads, top_k)

    result = agent(prompt)

    response_text = ""
    for block in result.message.get("content", []):
        if "text" in block:
            response_text += block["text"]

    try:
        parsed = _json.loads(response_text)
    except _json.JSONDecodeError:
        parsed = {
            "ranked_ads": [
                {"rank": i + 1, "id": ad["id"], "score": ad["_score"],
                 "reasoning": "See raw_response for LLM explanation."}
                for i, ad in enumerate(scored_ads[:top_k])
            ],
            "analysis": "Could not parse structured response from LLM.",
            "raw_response": response_text,
        }

    return {"result": parsed}


if __name__ == "__main__":
    app.run()
