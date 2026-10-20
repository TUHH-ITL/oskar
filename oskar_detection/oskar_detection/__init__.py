"""The one place a segmentation model is described, loaded and run.

Offline and online perception both load through here, so detecting fruitlets
instead of flowers is a different model file, not different code.
"""

from oskar_detection.detector import Detector, ModelSpec, load_detector, load_spec, resolve_model

__all__ = ["Detector", "ModelSpec", "load_detector", "load_spec", "resolve_model"]
