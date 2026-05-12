"""
Bedrock AgentCore agent that accepts an ad image and extracts 20 structured
engagement-predictor features using Claude's vision capabilities.

Payload
-------
{
  "image_base64": "<b64string>",
  "image_format": "png"          # optional, default "png"
}

Response
--------
{
  "result": {
    "aesthetic_score": 7,
    "visual_clutter": 3,
    ...
  }
}
"""

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

MODEL_ID = "arn:aws:bedrock:us-east-1:014498646416:inference-profile/global.anthropic.claude-sonnet-4-6"

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
You are an image feature extraction system for ad engagement prediction.
You will receive an advertisement image. Analyze it and extract exactly the
20 features listed below.

Return ONLY valid JSON (no markdown fences, no commentary) matching this schema:

{
  "aesthetic_score":            <int 1-10, overall visual appeal>,
  "visual_clutter":             <int 1-10, amount of distracting/unnecessary elements; 1=clean, 10=very cluttered>,
  "focal_point_presence":       <int 1-10, clarity of a single main subject; 1=none, 10=very clear>,
  "contrast_level":             <int 1-10, strength of contrast drawing attention>,
  "visual_saliency_score":      <int 1-10, likelihood image captures immediate attention>,

  "primary_subject_type":       <string, one of: "product", "person", "scene", "text-only", "mixed">,
  "product_visibility":         <int 1-10, how clearly the product is shown; 1=not visible, 10=dominant>,
  "usage_context":              <string, one of: "in-use", "standalone", "lifestyle", "abstract", "none">,

  "brand_prominence":           <int 1-10, how visible/dominant the brand is>,
  "logo_present":               <boolean>,

  "word_count":                 <int, total words detected in the image>,
  "text_density":               <int 1-10, text relative to available space; 1=minimal, 10=text-heavy>,
  "readability":                <int 1-10, ease of reading text at a glance>,

  "value_proposition_present":  <boolean, whether a clear benefit/message is communicated>,
  "cta_present":                <boolean, whether a call-to-action exists>,
  "offer_present":              <boolean, whether a promotion/discount/offer is displayed>,

  "emotion_valence":            <string, one of: "positive", "neutral", "negative">,
  "human_presence":             <boolean>,

  "perceived_credibility":      <int 1-10, overall trustworthiness suggested>,
  "spamminess_score":           <int 1-10, degree of spam-like/aggressive design; 1=professional, 10=very spammy>
}

Rules:
- Use the exact field names above.
- Integer scores use 1-10 scale unless otherwise noted.
- word_count is an absolute count, not a scale.
- Be precise and consistent. Base every score on what you observe in the image.
"""

EXTRACTION_PROMPT = (
    "Analyze this advertisement image and extract all 20 engagement-predictor "
    "features. Return only the JSON object, nothing else."
)


@app.entrypoint
def invoke(payload):
    import base64
    import json as _json

    agent = get_agent()

    image_b64 = payload.get("image_base64")
    image_format = payload.get("image_format", "png")

    if not image_b64:
        return {"error": "image_base64 is required"}

    image_bytes = base64.b64decode(image_b64)
    content_blocks = [
        {"text": EXTRACTION_PROMPT},
        {
            "image": {
                "format": image_format,
                "source": {
                    "bytes": image_bytes,
                },
            },
        },
    ]

    result = agent(content_blocks)

    response_text = ""
    for block in result.message.get("content", []):
        if "text" in block:
            response_text += block["text"]

    try:
        parsed = _json.loads(response_text)
    except _json.JSONDecodeError:
        parsed = {"raw_response": response_text, "parse_error": True}

    return {"result": parsed}


if __name__ == "__main__":
    app.run()
