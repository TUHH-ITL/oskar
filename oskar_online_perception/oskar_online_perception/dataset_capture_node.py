#!/usr/bin/env python3
"""Record the simulator's rendered frames and its instance masks as a COCO set.

The renderer labels only the flowers, so every instance id but the background
is one flower and the masks are exact. That makes a training set for a detector
that has to work on Isaac renders, which the models trained on real orchard
photographs do not.

Splits are written one capture run at a time: point --split at train, val or
test and run separate sweeps, so no two near-identical frames straddle a split.

Layout matches experiments/training/register_datasets.py:
  <out_root>/annotations/<split>.json
  <out_root>/<split>/<split>_000123.png
"""

import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class DatasetCaptureNode(Node):
    def __init__(self):
        super().__init__("dataset_capture_node")

        self.declare_parameter("image_topic", "/wrist_cam/left/image_raw")
        self.declare_parameter("instance_topic", "/wrist_cam/left/instance_segmentation")
        self.declare_parameter("label_topic", "/wrist_cam/left/instance_labels")
        self.declare_parameter("label_filter", "Flower")
        self.declare_parameter("out_root", "")
        self.declare_parameter("split", "train")
        # Consecutive frames of a sweep overlap almost completely; keeping every
        # one inflates the set without adding anything to learn from.
        self.declare_parameter("stride", 3)
        self.declare_parameter("min_mask_pixels", 20)
        self.declare_parameter("max_frames", 100000)
        self.declare_parameter("min_instances", 1)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("flush_every", 25)
        # Between sweeps the arm is parked and every frame is the same one.
        self.declare_parameter("gate_on_sweep", True)

        self.split = self.get_parameter("split").value
        self.stride = max(1, self.get_parameter("stride").value)
        self.min_mask_pixels = self.get_parameter("min_mask_pixels").value
        self.max_frames = self.get_parameter("max_frames").value
        self.min_instances = self.get_parameter("min_instances").value
        self.flush_every = self.get_parameter("flush_every").value

        out_root = self.get_parameter("out_root").value
        if not out_root:
            raise RuntimeError("out_root is required")
        self.out_root = Path(out_root)
        self.image_dir = self.out_root / self.split
        self.image_dir.mkdir(parents=True, exist_ok=True)
        (self.out_root / "annotations").mkdir(parents=True, exist_ok=True)
        self.annotation_path = self.out_root / "annotations" / f"{self.split}.json"

        self.images = []
        self.annotations = []
        self.seen = 0
        self.kept = 0
        self.bridge = CvBridge()
        self.label_filter = self.get_parameter("label_filter").value
        self.flower_ids = set()
        self.gated = self.get_parameter("gate_on_sweep").value
        self.recording = not self.gated

        qos = QoSProfile(depth=self.get_parameter("sync_queue_size").value,
                         reliability=ReliabilityPolicy.RELIABLE)
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, self.get_parameter("image_topic").value, qos_profile=qos),
             Subscriber(self, Image, self.get_parameter("instance_topic").value, qos_profile=qos)],
            queue_size=self.get_parameter("sync_queue_size").value,
            slop=self.get_parameter("sync_slop").value)
        sync.registerCallback(self._on_frame)
        # Ids 0 and 1 are BACKGROUND and UNLABELLED and cover most of the frame;
        # the rest name a prim. Without this map they read as two enormous
        # flowers in every single image.
        self.create_subscription(String, self.get_parameter("label_topic").value,
                                 self._on_labels, qos)
        self.create_subscription(String, "/perception/scan_event", self._on_scan_event, qos)

        self.get_logger().info(
            f"capturing split {self.split!r} into {self.out_root}, every {self.stride} frames")

    def _on_frame(self, image_msg, instance_msg):
        if not self.recording:
            return
        self.seen += 1
        if self.seen % self.stride or self.kept >= self.max_frames:
            return

        masks = self._masks(instance_msg)
        if len(masks) < self.min_instances:
            return

        rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        name = f"{self.split}_{self.kept:06d}.png"
        cv2.imwrite(str(self.image_dir / name), rgb)

        image_id = self.kept + 1
        height, width = rgb.shape[:2]
        self.images.append({"id": image_id, "file_name": name,
                            "height": height, "width": width})
        for mask in masks:
            self.annotations.append(self._annotation(mask, image_id))
        self.kept += 1

        if self.kept % self.flush_every == 0:
            self.flush()
            self.get_logger().info(
                f"{self.kept} frames, {len(self.annotations)} instances "
                f"({len(self.annotations) / self.kept:.1f} per frame)")

    def _on_scan_event(self, msg):
        if not self.gated:
            return
        self.recording = msg.data.split()[0] == "scan_sweeping"
        if not self.recording:
            self.flush()

    def _on_labels(self, msg):
        labels = json.loads(msg.data)
        self.flower_ids = {int(key) for key, name in labels.items()
                           if self.label_filter in name}

    def _masks(self, instance_msg):
        """One boolean mask per instance id in the simulator's segmentation."""
        from scipy import ndimage

        raw = self.bridge.imgmsg_to_cv2(instance_msg, desired_encoding="passthrough")
        if raw.ndim == 3:
            flat = np.zeros(raw.shape[:2], dtype=np.uint64)
            for channel in range(raw.shape[2]):
                flat = (flat << np.uint64(8)) | raw[..., channel].astype(np.uint64)
            raw = flat

        if not self.flower_ids:
            return []

        ids, inverse = np.unique(raw, return_inverse=True)
        labels = (inverse.reshape(raw.shape) + 1).astype(np.int32)

        masks = []
        for index, window in enumerate(ndimage.find_objects(labels), start=1):
            if window is None or int(ids[index - 1]) not in self.flower_ids:
                continue
            patch = labels[window] == index
            if patch.sum() < self.min_mask_pixels:
                continue
            full = np.zeros(labels.shape, dtype=np.uint8)
            full[window] = patch
            masks.append(full)
        return masks

    def _annotation(self, mask, image_id):
        from pycocotools import mask as mask_utils

        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        x, y, w, h = mask_utils.toBbox(rle).tolist()
        return {"id": len(self.annotations) + 1,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": [x, y, w, h],
                "area": float(mask_utils.area(rle)),
                "iscrowd": 0}

    def flush(self):
        self.annotation_path.write_text(json.dumps({
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": 1, "name": "appleflower"}]}))


def main():
    rclpy.init()
    node = DatasetCaptureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.flush()
        node.get_logger().info(
            f"wrote {node.kept} frames and {len(node.annotations)} instances "
            f"to {node.annotation_path}")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
