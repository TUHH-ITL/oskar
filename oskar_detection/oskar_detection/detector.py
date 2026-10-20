"""Load and run an instance segmentation model described by a model file.

A model file (models/*.yaml) names what the model detects and how to load it:

    class_label: flower
    architecture: maskdino          # maskdino | mask2former | maskrcnn
    weights: ~/.../model_final.pth
    config: ~/.../detectron2_config.yaml   # optional for maskrcnn: next to the weights
    source_roots:                   # source trees the query-based models import
      mask2former: ~/.../Mask2Former
      maskdino: ~/.../MaskDINO

The confidence threshold is not in the file. It is an operating point of the
pipeline using the model, tuned against that pipeline's output, so the caller
passes it.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ARCHITECTURES = ("maskdino", "mask2former", "maskrcnn")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class ModelSpec:
    class_label: str
    architecture: str
    weights: Path
    config: Path
    source_roots: dict = field(default_factory=dict)


def resolve_model(name_or_path):
    """A model file from a path, or from a bare name in the models directory."""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate
    for directory in (MODELS_DIR, _installed_models_dir()):
        if directory is None:
            continue
        named = directory / f"{name_or_path}.yaml"
        if named.is_file():
            return named
    raise FileNotFoundError(f"no model file {name_or_path!r} (looked in {MODELS_DIR})")


def load_spec(name_or_path):
    path = resolve_model(name_or_path)
    data = yaml.safe_load(path.read_text())
    architecture = str(data["architecture"]).lower()
    if architecture not in ARCHITECTURES:
        raise ValueError(f"{path}: architecture must be one of {ARCHITECTURES}")
    weights = Path(data["weights"]).expanduser()
    config = (Path(data["config"]).expanduser() if data.get("config")
              else weights.parent / "detectron2_config.yaml")
    roots = {k: Path(v).expanduser() for k, v in (data.get("source_roots") or {}).items()}
    return ModelSpec(str(data["class_label"]), architecture, weights, config, roots)


class Detector:
    """A loaded model. Calling it is Detectron2's DefaultPredictor call."""

    def __init__(self, spec, predictor, confidence_threshold, use_amp):
        self.spec = spec
        self.class_label = spec.class_label
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold
        self.use_amp = use_amp

    def __call__(self, image_bgr):
        return self.predictor(image_bgr)

    def detect(self, image_bgr, confidence_threshold=None):
        """(masks (N, H, W) bool, scores (N,), boxes (N, 4)) above the threshold."""
        import torch

        threshold = self.confidence_threshold if confidence_threshold is None \
            else confidence_threshold
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                instances = self.predictor(image_bgr)["instances"]
        scores = instances.scores.cpu().numpy()
        # The query-based models return a fixed set of candidates whatever the
        # image holds, and Detectron2's threshold key only reaches the box head,
        # which they lack: the threshold has to be applied here.
        keep = scores >= threshold
        masks = instances.pred_masks.cpu().numpy()[keep].astype(bool)
        boxes = (instances.pred_boxes.tensor.cpu().numpy()[keep]
                 if instances.has("pred_boxes") else np.zeros((int(keep.sum()), 4)))
        return masks, scores[keep], boxes


def load_detector(name_or_path, confidence_threshold, device="cuda", max_detections=None,
                  use_amp=True):
    """Load the model a model file describes."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    spec = load_spec(name_or_path)
    if not spec.weights.is_file():
        raise FileNotFoundError(f"weights do not exist: {spec.weights}")

    cfg = get_cfg()
    if spec.architecture in ("maskdino", "mask2former"):
        needed = ["mask2former"] + (["maskdino"] if spec.architecture == "maskdino" else [])
        for name in needed:
            root = spec.source_roots.get(name)
            if root is None or not root.is_dir():
                raise FileNotFoundError(
                    f"{spec.architecture} needs the {name} source tree; {root} is not one")
            if str(root) not in sys.path:
                sys.path.append(str(root))
        if spec.architecture == "maskdino":
            from maskdino import add_maskdino_config

            add_maskdino_config(cfg)
        else:
            from mask2former import add_maskformer2_config

            add_maskformer2_config(cfg)
        cfg.set_new_allowed(True)

    if not spec.config.is_file():
        raise FileNotFoundError(f"model config does not exist: {spec.config}")
    cfg.merge_from_file(str(spec.config))
    cfg.MODEL.WEIGHTS = str(spec.weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    if max_detections is not None:
        # A tree face can carry more objects than Detectron2's default 100, and
        # the ones it drops are not the weakest but whatever falls off the list.
        cfg.TEST.DETECTIONS_PER_IMAGE = int(max_detections)
    cfg.MODEL.DEVICE = device
    return Detector(spec, DefaultPredictor(cfg), confidence_threshold, use_amp)


def _installed_models_dir():
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("oskar_detection")) / "models"
    except Exception:
        return None
