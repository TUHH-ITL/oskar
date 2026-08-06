"""Stage 2: Pixel-Column Windowing Stage for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from pipeline_types import FlowerDetection, TreeSummary

class PixelColumnWindowingStage:
    """Stage 2: Projects tree trunks to compute [u_min, u_max] and LOCKS count N_i per tree from Frame A."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, raw_w: float, R_base_to_cam: np.ndarray, t_base_to_cam: np.ndarray):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.raw_w = raw_w
        self.R_base_to_cam = R_base_to_cam
        self.t_base_to_cam = t_base_to_cam

    def _project_enu_to_pixel(self, enu_pt: np.ndarray, pose: Dict[str, float]) -> Tuple[Optional[float], Optional[float], float]:
        e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        dx, dy, dz = enu_pt[0] - e, enu_pt[1] - n, enu_pt[2] - u
        p_base = np.array([cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy, dz])
        p_cam = self.R_base_to_cam.T @ (p_base - self.t_base_to_cam)
        x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]
        if z_c <= 0:
            return None, None, z_c
        pixel_u = self.fx * (x_c / z_c) + self.cx
        pixel_v = self.fy * (y_c / z_c) + self.cy
        return pixel_u, pixel_v, z_c

    def _compute_window(self, tid: int, tree_map: Dict[int, Dict[str, float]], pose: Dict[str, float]) -> Tuple[float, float]:
        u_curr, u_prev, u_next = None, None, None
        if tid in tree_map:
            pu, _, _ = self._project_enu_to_pixel(np.array([tree_map[tid]["x"], tree_map[tid]["y"], 0.5]), pose)
            u_curr = pu
        if tid - 1 in tree_map:
            pu, _, _ = self._project_enu_to_pixel(np.array([tree_map[tid-1]["x"], tree_map[tid-1]["y"], 0.5]), pose)
            u_prev = pu
        if tid + 1 in tree_map:
            pu, _, _ = self._project_enu_to_pixel(np.array([tree_map[tid+1]["x"], tree_map[tid+1]["y"], 0.5]), pose)
            u_next = pu

        u_left = (u_prev + u_curr) / 2.0 if u_prev is not None else 0.0
        u_right = (u_curr + u_next) / 2.0 if u_next is not None else self.raw_w

        u_min = max(0.0, min(u_left, u_right))
        u_max = min(self.raw_w, max(u_left, u_right))
        return u_min, u_max

    def run(
        self,
        tree_map: Dict[int, Dict[str, float]],
        gt_map: Dict[int, float],
        tree_frame_map: Dict[int, Tuple[str, str]],  # (Frame A, Frame B)
        frame_detections_map: Dict[str, List[FlowerDetection]],
        interpolate_pose_fn,
        t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr
    ) -> Dict[int, TreeSummary]:
        """Calculates 2D column boundaries, crops detections, and LOCKS count per tree from Frame A."""
        summaries = {}
        import utils

        for tid in sorted(list(tree_map.keys())):
            frame_tuple = tree_frame_map.get(tid)
            if not frame_tuple:
                continue
            frame_a, frame_b = frame_tuple

            # Frame A processing (Primary, Locks Count N_i)
            ts_a = utils.parse_frame_timestamp(frame_a)
            pose_a = interpolate_pose_fn(ts_a, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
            u_min_a, u_max_a = self._compute_window(tid, tree_map, pose_a)

            frame_a_dets = frame_detections_map.get(frame_a, [])
            windowed_a_dets = []
            for det in frame_a_dets:
                if u_min_a <= det.centroid_u <= u_max_a:
                    det_copy = FlowerDetection(
                        instance_id=det.instance_id, confidence=det.confidence,
                        centroid_u=det.centroid_u, centroid_v=det.centroid_v,
                        frame_name=frame_a, tree_id=tid
                    )
                    windowed_a_dets.append(det_copy)

            # Frame B processing (Secondary candidate observations of SAME tree_id)
            ts_b = utils.parse_frame_timestamp(frame_b)
            pose_b = interpolate_pose_fn(ts_b, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
            u_min_b, u_max_b = self._compute_window(tid, tree_map, pose_b)

            frame_b_dets = frame_detections_map.get(frame_b, [])
            windowed_b_dets = []
            for det in frame_b_dets:
                if u_min_b <= det.centroid_u <= u_max_b:
                    det_copy = FlowerDetection(
                        instance_id=det.instance_id, confidence=det.confidence,
                        centroid_u=det.centroid_u, centroid_v=det.centroid_v,
                        frame_name=frame_b, tree_id=tid
                    )
                    windowed_b_dets.append(det_copy)

            raw_count = len(windowed_a_dets)
            pred_buschel = raw_count / 4.5
            gt_val = gt_map.get(tid, np.nan)

            summary = TreeSummary(
                tree_id=tid,
                gt_buschel=gt_val,
                closest_frame=frame_a,
                raw_windowed_count=raw_count,
                predicted_buschel=pred_buschel,
                column_window=(u_min_a, u_max_a)
            )
            summary._windowed_a_dets = windowed_a_dets
            summary._windowed_b_dets = windowed_b_dets
            summary._frame_b_name = frame_b
            summaries[tid] = summary

        return summaries
