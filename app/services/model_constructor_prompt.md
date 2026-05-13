Now I want to refactor `model_service.py` a bit to construct a class `CustomInferenceInterface` that has the following interface:

# Input: parameter input to method `predict`

- Images
- User profile

# Output: parameter returned after calling `predict`

Per image, there is

- slot_index: int
- filename: str
- affinity: float
- reason: str
- image_attributes: dict[str, str]