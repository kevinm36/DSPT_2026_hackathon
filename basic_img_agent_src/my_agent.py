"""
Bedrock AgentCore agent that queries Claude via Strands.
Supports text-only and multimodal (text + image) payloads.

Text-only payload:
  {"prompt": "Hello!"}

Image payload:
  {"prompt": "What's in this image?", "image_base64": "<b64string>", "image_format": "png"}
"""

from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

MODEL_ID = "arn:aws:bedrock:us-east-1:014498646416:inference-profile/global.anthropic.claude-sonnet-4-6"

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from strands import Agent
        _agent = Agent(model=MODEL_ID)
    return _agent


@app.entrypoint
def invoke(payload):
    import base64

    agent = get_agent()

    prompt = payload.get("prompt", "What do you see in this image?")
    image_b64 = payload.get("image_base64")
    image_format = payload.get("image_format", "png")

    if image_b64:
        image_bytes = base64.b64decode(image_b64)
        content_blocks = [
            {"text": prompt},
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
    else:
        result = agent(prompt)

    return {"result": result.message}


if __name__ == "__main__":
    app.run()
