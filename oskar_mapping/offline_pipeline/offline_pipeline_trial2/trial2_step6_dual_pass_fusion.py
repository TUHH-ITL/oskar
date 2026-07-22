"""Step 6: Dual-Pass (Front + Back Pass) Fusion & Deduplication Stage for Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

import os
import sys
import struct
import numpy as np
from typing import Dict, List, Tuple
from pipeline_types import FlowerDetection, TreeSummary

class DualPassFusionStage:
    """Combines Front Pass (West Side) and Back Pass (East Side) detections with 3D spatial deduplication."""

    def __init__(self, rec_path: str, r_match_m: float = 0.015):
        self.rec_path = rec_path
        self.r_match_m = r_match_m
        self.num_rec_frames = (os.path.getsize(rec_path) - 12) // 24551456

    def unpack_rec_frame(self, frame_idx: int) -> Tuple[float, np.ndarray]:
        """Unpacks timestamp and rectified BGR image for frame_idx from SAMSON3 .rec file."""
        import cv2
        frame_size = 24551456
        with open(self.rec_path, "rb") as f:
            f.seek(12 + frame_idx * frame_size)
            hdr = f.read(32)
            payloadbytes, imageid, secs, nsecs = struct.unpack("<QQQQ", hdr)
            ts = float(secs) + float(nsecs) * 1e-9
            img_bytes = f.read(payloadbytes)
            raw_img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(4608, 5328)
            bgr_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerBG2BGR)
        return ts, bgr_img

    def classify_rec_frames(self, interpolate_pose_func, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr):
        """Classifies .rec frames into Pass 1 (Front, ts < 1713171700) and Pass 2 (Back, ts > 1713172000)."""
        pass1_info = []
        pass2_info = []
        with open(self.rec_path, "rb") as f:
            for idx in range(self.num_rec_frames):
                f.seek(12 + idx * 24551456)
                hdr = f.read(32)
                _, _, secs, nsecs = struct.unpack("<QQQQ", hdr)
                ts = float(secs) + float(nsecs) * 1e-9
                pose = interpolate_pose_func(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
                yaw = pose["yaw"]
                if -0.6 <= yaw <= 0.2 and ts < 1713171700:
                    pass1_info.append((idx, ts, pose))
                elif (2.5 <= yaw <= 3.14 or -3.14 <= yaw <= -2.5) and ts > 1713172000:
                    pass2_info.append((idx, ts, pose))
        return pass1_info, pass2_info

    def deduplicate_front_back(self, pts_front: np.ndarray, pts_back: np.ndarray) -> Tuple[int, int]:
        """Returns (deduplicated_count, overlap_matches) for front and back 3D points."""
        n_front = len(pts_front)
        n_back = len(pts_back)
        n_sum = n_front + n_back

        if n_front == 0 or n_back == 0:
            return n_sum, 0

        dist_mat = np.linalg.norm(pts_front[:, np.newaxis, :] - pts_back[np.newaxis, :, :], axis=2)
        matched_count = int(np.sum(np.min(dist_mat, axis=1) <= self.r_match_m))
        n_dedup = n_sum - matched_count
        return n_dedup, matched_count
