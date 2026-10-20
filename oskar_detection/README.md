# oskar_detection

The one place a segmentation model is described and loaded. Offline and online
perception both use it, so a new object class is a new model file.

```python
from oskar_detection import load_detector
detector = load_detector("flowers_maskdino", confidence_threshold=0.6)
masks, scores, boxes = detector.detect(image_bgr)
detector.class_label   # "flower"
```

| Model file | Detects | Architecture |
|---|---|---|
| `models/flowers_maskdino.yaml` | flowers, trained on real images | MaskDINO |
| `models/flowers_maskrcnn_sim.yaml` | flowers, trained on the sim orchard | Mask R-CNN |

For fruitlets: train the model, add `models/fruitlets_<architecture>.yaml` with
`class_label: fruitlet`, and point the pipelines at it. The confidence threshold
is not part of the model file; each pipeline tunes its own.

The offline perception is not a ROS package and imports this one from the source
tree; the online perception imports it from the installed workspace.
