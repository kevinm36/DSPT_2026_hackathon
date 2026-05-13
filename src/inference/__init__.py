"""Inference-time wrappers around trained models for the FastAPI app.

The classes in this package are subclasses of
``app.services.model_service.CustomInferenceInterface`` and are imported
lazily by ``app/services/model_service.py`` to keep the request/serving
contract owned by the app while the inference logic lives in ``src/``.
"""
