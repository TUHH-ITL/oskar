#!/usr/bin/env python3
"""Segment objects, place them in 3D, publish them in the map frame.

Phase 1 spreads this over five nodes joined by topics, and one of those topics
carries a full-frame sensor_msgs/Image mask per detected flower: a hundred
flowers at 1280x720 is roughly 90 MB per frame on the wire, which no amount of
model speed recovers. Here segmentation, depth lookup and back-projection share
one process, masks stay numpy, and only the 3D points are published.

Masks come from one of two sources, selected by the `model` parameter and
sharing every step after them:

  a model file  any oskar_detection model, by name (flowers_maskdino,
                flowers_maskrcnn_sim, ...); the offline perception loads the
                same files, and the model's class label rides on every detection
  oracle        the simulator's own instance segmentation, perfect masks on the
                real rendered image

The oracle exists to separate failures. Everything downstream of the masks --
deprojection, the transform lookup, synchronisation, the tracker under real
camera motion -- can then be brought up and measured without the detector in
the loop, and the detector afterwards measured against a known ceiling instead
of against an unknown.
"""

import time
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from oskar_msgs.msg import Detection3D, Detections3D


class Detection3DNode(Node):
    def __init__(self):
        super().__init__("detection_3d_node")

        self.declare_parameter("image_topic", "/wrist_cam/left/image_raw")
        self.declare_parameter("depth_topic", "/wrist_cam/left/depth")
        self.declare_parameter("camera_info_topic", "/wrist_cam/left/camera_info")
        self.declare_parameter("optical_frame", "wrist_cam_left")
        self.declare_parameter("map_frame", "map")
        # An oskar_detection model file, or "oracle".
        self.declare_parameter("model", "flowers_maskdino")
        # Only for the oracle, which has no model file to name its class.
        self.declare_parameter("oracle_class_label", "flower")
        self.declare_parameter("instance_topic", "/wrist_cam/left/instance_segmentation")
        self.declare_parameter("label_topic", "/wrist_cam/left/instance_labels")
        self.declare_parameter("label_filter", "Flower")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("sync_slop", 0.05)
        # Depth and the mask source arrive separately; a shallow queue drops a
        # pair whenever they are a frame apart, which halved the usable rate.
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("tf_cache_s", 60.0)
        self.declare_parameter("erosion_radius_px", 2)
        self.declare_parameter("min_mask_pixels", 12)
        # Offline the depth of a flower is taken high in the mask's distribution,
        # because foliage in front of it drags the median toward the camera. The
        # simulator renders exact depth, so the only contamination is mask bleed
        # onto the background, which the median already rejects.
        self.declare_parameter("depth_percentile", 50.0)
        self.declare_parameter("max_depth_spread_m", 0.10)
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 3.0)
        self.declare_parameter("report_interval_s", 5.0)
        # A guard on the oracle: far more ids than flowers in view means the
        # background id was misread and the frame is noise, not detections.
        self.declare_parameter("max_instances", 2000)
        self.declare_parameter("max_detections", 300)
        # Drawn only while something subscribes, so watching costs nothing
        # when nobody is.
        self.declare_parameter("publish_debug_image", True)
        # The tracker links a flower to its previous sighting only within this,
        # so the camera must not travel further between processed frames. The
        # node measures the travel and says so rather than assuming it.
        self.declare_parameter("match_radius_m", 0.025)

        self.optical_frame = self.get_parameter("optical_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.erosion_radius = self.get_parameter("erosion_radius_px").value
        self.min_mask_pixels = self.get_parameter("min_mask_pixels").value
        self.depth_percentile = self.get_parameter("depth_percentile").value
        self.max_depth_spread = self.get_parameter("max_depth_spread_m").value
        self.min_depth = self.get_parameter("min_depth_m").value
        self.max_instances = self.get_parameter("max_instances").value
        self.match_radius = self.get_parameter("match_radius_m").value
        self.use_amp = self.get_parameter("device").value.startswith("cuda")
        self.erosion_kernel = None
        if self.erosion_radius > 0:
            import cv2

            size = 2 * self.erosion_radius + 1
            self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        self.max_depth = self.get_parameter("max_depth_m").value

        model = self.get_parameter("model").value
        self.bridge = CvBridge()
        self.camera_info = None
        if model == "oracle":
            self.detector = "oracle"
            self.model = None
            self.class_label = self.get_parameter("oracle_class_label").value
        else:
            from oskar_detection import load_detector

            self.model = load_detector(
                model, self.confidence_threshold, device=self.get_parameter("device").value,
                max_detections=self.get_parameter("max_detections").value,
                use_amp=self.use_amp)
            self.detector = self.model.spec.architecture
            self.class_label = self.model.class_label
            self.get_logger().info(f"model {model}: {self.detector} detecting {self.class_label}")

        # A slow detector pulls frames from the queue long after they were
        # taken: at 0.65 s a frame with a queue of 20, up to 13 s stale, past
        # tf2's 10 s default cache. Every one of those looks up a transform the
        # buffer has already dropped.
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(
                seconds=self.get_parameter("tf_cache_s").value))
        # Own thread: the node's callbacks share one mutually exclusive group,
        # so without it the buffer stops filling for the whole inference.
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value,
                                 self._camera_info_callback, qos)
        # The arm swings into position at full speed before a sweep; leaving
        # those frames in the window made the travel look many times worse than
        # the sweep itself.
        self.create_subscription(String, "/perception/scan_event", self._scan_event_callback, qos)
        self.pub = self.create_publisher(Detections3D, "/perception/detections_3d", qos)
        self.debug_pub = (self.create_publisher(Image, "/perception/debug_image", qos)
                          if self.get_parameter("publish_debug_image").value else None)

        self.label_filter = self.get_parameter("label_filter").value
        self.flower_ids = set()
        if self.detector == "oracle":
            self.create_subscription(String, self.get_parameter("label_topic").value,
                                     self._labels_callback, qos)

        source_topic = (self.get_parameter("instance_topic").value
                        if self.detector == "oracle"
                        else self.get_parameter("image_topic").value)
        source_sub = Subscriber(self, Image, source_topic, qos_profile=qos)
        depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value, qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer(
            [source_sub, depth_sub],
            queue_size=self.get_parameter("sync_queue_size").value,
            slop=self.get_parameter("sync_slop").value)
        self.sync.registerCallback(self._frame_callback)

        self.frames = 0
        self.dropped_no_tf = 0
        self.latencies = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        self.camera_steps = deque(maxlen=100)
        self.frame_periods = deque(maxlen=100)
        self.last_camera_position = None
        self.last_frame_time = None
        self.create_timer(self.get_parameter("report_interval_s").value, self._report)
        self.get_logger().info(
            f"perception ready: detector={self.detector}, source={source_topic}, "
            f"{self.optical_frame} -> {self.map_frame}, depth p{self.depth_percentile:.0f}")

    def _scan_event_callback(self, msg):
        if msg.data.startswith("scan_sweeping"):
            self.camera_steps.clear()
            self.frame_periods.clear()
            self.latencies.clear()
            self.detection_counts.clear()
            self.last_camera_position = None
            self.get_logger().info("sweep started, statistics reset")

    def _camera_info_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().info(
                f"camera info: {msg.width}x{msg.height}, "
                f"fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}")
        self.camera_info = msg

    def _frame_callback(self, source_msg, depth_msg):
        if self.camera_info is None:
            return
        started = time.perf_counter()

        transform = self._lookup_transform(source_msg.header.stamp)
        if transform is None:
            self.dropped_no_tf += 1
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        pixels, scores, shape = (self._pixels_from_model(source_msg)
                                 if self.detector != "oracle"
                                 else self._pixels_from_renderer(source_msg))
        if pixels is None:
            return
        if shape != depth.shape[:2]:
            self.get_logger().warn(
                f"depth {depth.shape[:2]} does not match the mask source {shape}", once=True)
            return

        points, kept_scores, spreads, diameters = self._locate(pixels, scores, depth)
        out = Detections3D()
        out.header.stamp = source_msg.header.stamp
        out.header.frame_id = self.map_frame

        if len(points):
            world = self._to_map(points, transform)
            t = transform.transform.translation
            towards_camera = np.array([t.x, t.y, t.z]) - world
            towards_camera /= np.linalg.norm(towards_camera, axis=1, keepdims=True)
            for i in range(len(world)):
                d = Detection3D()
                d.instance_id = i
                d.class_label = self.class_label
                d.position.x, d.position.y, d.position.z = (float(v) for v in world[i])
                d.confidence = float(kept_scores[i])
                d.depth_std_ratio = float(spreads[i])
                d.low_confidence = bool(kept_scores[i] < self.confidence_threshold)
                d.view_direction.x, d.view_direction.y, d.view_direction.z = (
                    float(v) for v in towards_camera[i])
                d.diameter = float(diameters[i])
                out.detections.append(d)
        self.pub.publish(out)
        self._publish_debug(source_msg, pixels, len(out.detections))

        self.frames += 1
        self.latencies.append(time.perf_counter() - started)
        self.detection_counts.append(len(out.detections))

        t = transform.transform.translation
        position = np.array([t.x, t.y, t.z])
        stamp = source_msg.header.stamp.sec + source_msg.header.stamp.nanosec * 1e-9
        if self.last_camera_position is not None:
            self.camera_steps.append(float(np.linalg.norm(position - self.last_camera_position)))
            self.frame_periods.append(stamp - self.last_frame_time)
        self.last_camera_position = position
        self.last_frame_time = stamp

    def _publish_debug(self, source_msg, pixels, located):
        """The frame with every mask painted on it, for watching a run."""
        if self.debug_pub is None or not self.debug_pub.get_subscription_count():
            return
        import cv2

        if self.detector == "oracle":
            raw = self.bridge.imgmsg_to_cv2(source_msg, desired_encoding="passthrough")
            canvas = np.zeros(raw.shape[:2] + (3,), dtype=np.uint8)
        else:
            canvas = self.bridge.imgmsg_to_cv2(source_msg, desired_encoding="bgr8").copy()

        for rows, cols in pixels:
            canvas[rows, cols] = (0, 0, 255)
            if len(rows):
                cv2.rectangle(canvas, (int(cols.min()) - 3, int(rows.min()) - 3),
                              (int(cols.max()) + 3, int(rows.max()) + 3), (0, 255, 0), 1)
        cv2.putText(canvas, f"{len(pixels)} masks, {located} placed in 3D",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        message = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        message.header = source_msg.header
        self.debug_pub.publish(message)

    def _erode(self, mask):
        """Shrink a mask off its edges, unless that would erase it."""
        if self.erosion_kernel is None:
            return mask
        import cv2

        eroded = cv2.erode(mask, self.erosion_kernel)
        return eroded if eroded.sum() >= self.min_mask_pixels else mask

    def _pixels_from_model(self, image_msg):
        """Pixel coordinates per detected object, from the model."""
        rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        masks, scores, _ = self.model.detect(rgb)
        pixels = [np.nonzero(self._erode(mask.astype(np.uint8))) for mask in masks]
        return pixels, scores, rgb.shape[:2]

    def _labels_callback(self, msg):
        import json

        labels = json.loads(msg.data)
        self.flower_ids = {int(key) for key, name in labels.items()
                           if self.label_filter in name}

    def _pixels_from_renderer(self, instance_msg):
        """Pixel coordinates per instance in the simulator's segmentation image.

        Each instance is handled inside its own bounding box. Testing the whole
        frame once per instance instead costs a full-resolution pass per flower,
        which measured 567 ms a frame with 45 flowers in view.

        The encoding varies with the renderer's build, so the ids are read out
        of the raw buffer rather than assumed; a colour image is folded into a
        single integer per pixel so distinct colours stay distinct ids. Which
        ids are flowers comes from the label topic: ids 0 and 1 are BACKGROUND
        and UNLABELLED, and taking anything but them for a flower puts two
        frame-sized detections in every image.
        """
        from scipy import ndimage

        raw = self.bridge.imgmsg_to_cv2(instance_msg, desired_encoding="passthrough")
        if raw.ndim == 3:
            flat = np.zeros(raw.shape[:2], dtype=np.uint64)
            for channel in range(raw.shape[2]):
                flat = (flat << np.uint64(8)) | raw[..., channel].astype(np.uint64)
            raw = flat

        if not self.flower_ids:
            return [], np.empty(0), raw.shape

        ids, inverse = np.unique(raw, return_inverse=True)
        if len(ids) - 1 > self.max_instances:
            self.get_logger().warn(
                f"{len(ids) - 1} instance ids in one frame, above max_instances "
                f"({self.max_instances})", once=True)
            return None, None, raw.shape

        labels = (inverse.reshape(raw.shape) + 1).astype(np.int32)

        pixels = []
        for index, window in enumerate(ndimage.find_objects(labels), start=1):
            if window is None or int(ids[index - 1]) not in self.flower_ids:
                continue
            patch = self._erode((labels[window] == index).astype(np.uint8))
            rows, cols = np.nonzero(patch)
            pixels.append((rows + window[0].start, cols + window[1].start))
        return pixels, np.ones(len(pixels)), raw.shape

    def _lookup_transform(self, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                self.map_frame, self.optical_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=0.05))
        except tf2_ros.TransformException:
            return None

    def _locate(self, pixels, scores, depth):
        """One 3D point per flower, in the optical frame."""
        k = self.camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        points, kept, spreads, diameters = [], [], [], []
        # The masks were eroded, so the radius gets the erosion back.
        erosion = float(self.erosion_radius) if self.erosion_kernel is not None else 0.0
        for (v, u), score in zip(pixels, scores):
            if len(u) < self.min_mask_pixels:
                continue

            d = depth[v, u].astype(np.float64)
            valid = np.isfinite(d) & (d > self.min_depth) & (d < self.max_depth)
            if valid.sum() < self.min_mask_pixels:
                continue
            d, u, v = d[valid], u[valid], v[valid]

            z = float(np.percentile(d, self.depth_percentile))
            spread = float(np.percentile(d, 84) - np.percentile(d, 16))
            if spread > self.max_depth_spread:
                # The mask straddles a depth discontinuity, so no single point
                # represents it; the pixels near the chosen depth still do.
                near = np.abs(d - z) < self.max_depth_spread
                if near.sum() < self.min_mask_pixels:
                    continue
                u, v, d = u[near], v[near], d[near]

            cu, cv_ = float(u.mean()), float(v.mean())
            points.append([(cu - cx) * z / fx, (cv_ - cy) * z / fy, z])
            kept.append(score)
            spreads.append(spread)
            radius_px = np.sqrt(len(u) / np.pi) + erosion
            diameters.append(2.0 * radius_px * z / np.sqrt(fx * fy))
        return (np.array(points).reshape(-1, 3), np.array(kept), np.array(spreads),
                np.array(diameters))

    @staticmethod
    def _to_map(points, transform):
        t = transform.transform.translation
        q = transform.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        return points @ R.T + np.array([t.x, t.y, t.z])

    def _report(self):
        if not self.latencies:
            if self.dropped_no_tf:
                self.get_logger().warn(
                    f"no frames processed; {self.dropped_no_tf} dropped waiting for "
                    f"{self.map_frame} -> {self.optical_frame}")
            return
        message = (
            f"{self.frames} frames, {np.mean(self.latencies) * 1e3:.0f} ms of compute per "
            f"frame ({1.0 / np.mean(self.latencies):.1f} Hz if nothing else limited it), "
            f"{np.mean(self.detection_counts):.1f} detections/frame, "
            f"{self.dropped_no_tf} dropped without TF")

        moving = [d for d in self.camera_steps if d > 1e-4]
        if moving:
            worst = float(np.percentile(moving, 95))
            rate = 1.0 / np.mean(self.frame_periods) if np.mean(self.frame_periods) > 0 else 0.0
            message += (f"; while moving, {rate:.1f} Hz in simulation time and "
                        f"{worst * 1000:.0f} mm of camera travel per frame at the 95th "
                        f"percentile, against a {self.match_radius * 1000:.0f} mm match "
                        f"radius (simulation time is what binds: wall clock only decides "
                        f"how long the scan takes to watch)")
            if worst > self.match_radius:
                message += " -- TOO FAST, flowers will fragment into separate landmarks"
        self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = Detection3DNode()
    # Inference blocks for the better part of a second, and on one thread the
    # transform listener cannot run while it does: transforms pile up behind the
    # frame callback, the buffer gains holes, and frames are dropped for want of
    # a pose that was published on time. Two threads keep the listener fed.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
