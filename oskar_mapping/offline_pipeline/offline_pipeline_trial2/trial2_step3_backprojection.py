"""Stage 3: Backprojection Stage for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

import numpy as np
from typing import Dict, List

from pipeline_types import FlowerDetection, TreeSummary

class BackprojectionStage:
    """Stage 3: Backprojects 2D windowed detections of Frame A and Frame B into 3D ENU world coordinates."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, R_base_to_cam: np.ndarray, t_base_to_cam: np.ndarray, default_depth: float = 1.3):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.R_base_to_cam = R_base_to_cam
        self.t_base_to_cam = t_base_to_cam
        self.default_depth = default_depth

    def backproject_detections(self, dets: List[FlowerDetection], pose: dict) -> np.ndarray:
        """Backprojects a list of 2D detections given a camera pose into 3D ENU world array."""
        e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_map_to_base = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        t_map_to_base = np.array([e, n, u])

        pts_3d = []
        for det in dets:
            z_c = self.default_depth
            x_c = (det.centroid_u - self.cx) * z_c / self.fx
            y_c = (det.centroid_v - self.cy) * z_c / self.fy
            p_cam = np.array([x_c, y_c, z_c])
            p_base = self.R_base_to_cam @ p_cam + self.t_base_to_cam
            p_map = R_map_to_base @ p_base + t_map_to_base
            pts_3d.append(p_map)
        return np.array(pts_3d) if len(pts_3d) > 0 else np.zeros((0, 3))

    def _backproject_list(self, dets: List[FlowerDetection], frame_name: str, interpolate_pose_fn, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr):
        import utils
        ts = utils.parse_frame_timestamp(frame_name)
        pose = interpolate_pose_fn(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
        e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_map_to_base = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        t_map_to_base = np.array([e, n, u])

        landmarks = []
        for det in dets:
            z_c = self.default_depth
            x_c = (det.centroid_u - self.cx) * z_c / self.fx
            y_c = (det.centroid_v - self.cy) * z_c / self.fy

            det.cam_xyz = (x_c, y_c, z_c)
            p_cam = np.array([x_c, y_c, z_c])
            p_base = self.R_base_to_cam @ p_cam + self.t_base_to_cam
            p_map = R_map_to_base @ p_base + t_map_to_base

            world_pt = (float(p_map[0]), float(p_map[1]), float(p_map[2]))
            det.world_xyz = world_pt
            det.refined_world_xyz = world_pt
            landmarks.append(world_pt)
        return landmarks

    def run(self, summaries: Dict[int, TreeSummary], interpolate_pose_fn, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr) -> Dict[int, TreeSummary]:
        """Backprojects Frame A and Frame B windowed detections for each tree."""
        for tid, summary in summaries.items():
            win_a: List[FlowerDetection] = getattr(summary, "_windowed_a_dets", [])
            win_b: List[FlowerDetection] = getattr(summary, "_windowed_b_dets", [])
            frame_b_name = getattr(summary, "_frame_b_name", "")

            lms_a = self._backproject_list(win_a, summary.closest_frame, interpolate_pose_fn, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
            if win_b and frame_b_name:
                self._backproject_list(win_b, frame_b_name, interpolate_pose_fn, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)

            summary.landmarks_3d = lms_a

        return summaries
