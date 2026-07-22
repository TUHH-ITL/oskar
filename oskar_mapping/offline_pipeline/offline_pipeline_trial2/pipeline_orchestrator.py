"""Top-Level Orchestrator for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
No production pipeline files are modified or overwritten.
Supports both Single-Pass (West Side) and Dual-Pass (Front + Back Pass) modes.
"""

import os
import sys
import json
import yaml
import gzip
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, List, Optional, Tuple

TRIAL2_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TRIAL2_DIR, "data")
CACHE_DIR = os.path.join(TRIAL2_DIR, "fresh_cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

OFFLINE_DIR = os.path.dirname(TRIAL2_DIR)
WORKSPACE_DIR = os.path.dirname(os.path.dirname(OFFLINE_DIR))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "datasets/Blossom2024")

sys.path.append(OFFLINE_DIR)
sys.path.append(TRIAL2_DIR)

import config
import utils
from step4_backprojection import interpolate_pose, get_base_to_camera_transform
from pipeline_types import FlowerDetection, TreeSummary, ThinningDecisionItem
from trial2_step1_segmentation import SegmentationStage
from trial2_step2_pixel_column_windowing import PixelColumnWindowingStage
from trial2_step3_backprojection import BackprojectionStage
from trial2_step4_position_refinement import PositionRefinementStage
from trial2_step5_thinning_decision import (
    ThinningDecisionStage,
    PLACEHOLDER_THINNING_TARGET_BUSCHEL,
    PLACEHOLDER_THINNING_AGRONOMIC_MAX_BUSCHEL
)
from trial2_step6_dual_pass_fusion import DualPassFusionStage
from trial2_utils import (
    save_thinning_plan_plot,
    save_flower_summary_table,
    save_2d_cluster_map
)

class OfflinePipelineTrial2:
    """Orchestrator for Offline Pipeline Trial 2 (2D Pixel-Column Windowed Architecture)."""

    def __init__(self, conf_thresh: float = 0.60, mode: str = "dual_pass"):
        # EXPLICITLY FIX AND LOG CONFIDENCE THRESHOLD
        self.conf_thresh = conf_thresh
        self.mode = mode.lower()

        # Load calibration
        with open(config.LEFT_CALIB_FILE, 'r') as f:
            cdata = yaml.safe_load(f)
        P_raw = np.array(cdata["projectionMatrix"], dtype=np.float64)
        self.fx, self.fy = P_raw[0, 0], P_raw[1, 1]
        self.cx, self.cy = P_raw[0, 2], P_raw[1, 2]
        self.raw_w = 5328.0 if self.mode == "dual_pass" else float(cdata.get("image_width", 5344))

        camera_side = getattr(config, "CAMERA_SIDE", "left")
        camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
        self.R_base_to_cam, self.t_base_to_cam = get_base_to_camera_transform(camera_side, camera_xyz)

        # Load Tree Map
        with open(config.TREE_MAP_FILE, 'r') as f:
            self.tree_map = yaml.safe_load(f)["trees"]

        # Load Trajectory
        traj_data = np.load(config.POSE_TRAJECTORY_FILE)
        self.t_arr = traj_data["t"]
        self.E_arr = traj_data["E"]
        self.N_arr = traj_data["N"]
        self.U_arr = traj_data["U"]
        yaw_arr = traj_data["yaw"]
        self.cos_arr = np.cos(yaw_arr)
        self.sin_arr = np.sin(yaw_arr)

        # Load Ground Truth
        excel_path = os.path.join(DATASET_DIR, "Blütenstand (15.04.2024).xlsx")
        df_gt = pd.read_excel(excel_path, sheet_name=0)
        self.gt_map = {}
        for idx, row in df_gt.iterrows():
            try:
                b_val = row["Baum"]
                if pd.isna(b_val):
                    continue
                tid = int(b_val)
                buschel_val = row["Büschel"]
                if str(buschel_val).strip().lower() == "befruchter":
                    self.gt_map[tid] = np.nan
                else:
                    num_val = pd.to_numeric(buschel_val, errors='coerce')
                    self.gt_map[tid] = float(num_val) if not pd.isna(num_val) else np.nan
            except Exception:
                continue

        # Instantiate Modular Stage Classes
        self.seg_stage = SegmentationStage(OFFLINE_DIR, TRIAL2_DIR, conf_thresh)
        self.window_stage = PixelColumnWindowingStage(self.fx, self.fy, self.cx, self.cy, self.raw_w, self.R_base_to_cam, self.t_base_to_cam)
        self.backproj_stage = BackprojectionStage(self.fx, self.fy, self.cx, self.cy, self.R_base_to_cam, self.t_base_to_cam)
        self.refine_stage = PositionRefinementStage(refine_radius_m=0.015)
        self.decision_stage = ThinningDecisionStage(PLACEHOLDER_THINNING_TARGET_BUSCHEL, PLACEHOLDER_THINNING_AGRONOMIC_MAX_BUSCHEL)

        rec_path = os.path.join(DATASET_DIR, "2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1/2024-04-15_10-59-41_A27_Bluete_SAMSON3_1713171581.rec")
        self.dual_stage = DualPassFusionStage(rec_path, r_match_m=0.015)

    def run_pipeline(self, max_trees: int = 65):
        """Executes pipeline end-to-end (supporting both single_pass and dual_pass)."""
        print(f"=== OFFLINE PIPELINE TRIAL 2 (MODE={self.mode.upper()}, CONF_THRESH={self.conf_thresh:.2f}) ===")

        if self.mode == "dual_pass":
            return self._run_dual_pass_pipeline(max_trees)
        else:
            return self._run_single_pass_pipeline(max_trees)

    def _run_dual_pass_pipeline(self, max_trees: int):
        pass1_info, pass2_info = self.dual_stage.classify_rec_frames(
            interpolate_pose, self.t_arr, self.E_arr, self.N_arr, self.U_arr, self.cos_arr, self.sin_arr
        )

        summaries = []
        for tid in sorted(list(self.tree_map.keys())):
            if tid > max_trees:
                continue
            tx, ty = self.tree_map[tid]["x"], self.tree_map[tid]["y"]

            d1 = [np.hypot(p[2]["e"] - tx, p[2]["n"] - ty) for p in pass1_info]
            idx1_best, ts1_best, pose1_best = pass1_info[np.argmin(d1)]

            d2 = [np.hypot(p[2]["e"] - tx, p[2]["n"] - ty) for p in pass2_info]
            idx2_best, ts2_best, pose2_best = pass2_info[np.argmin(d2)]

            name_p1 = f"tree_{(idx1_best - 113):04d}" if idx1_best >= 113 else f"rec_frame_{idx1_best:04d}"
            name_p2 = f"pass2_rec_frame_{idx2_best:04d}"

            dets1 = self._get_cached_or_live_detections(idx1_best, name_p1)
            dets2 = self._get_cached_or_live_detections(idx2_best, name_p2)

            # Pass 1 Windowing & Backprojection
            u_min1, u_max1 = self._compute_window(tid, pose1_best)
            win1 = [d for d in dets1 if u_min1 <= d.centroid_u <= u_max1]
            pts1 = self.backproj_stage.backproject_detections(win1, pose1_best)

            # Pass 2 Windowing & Backprojection
            u_min2, u_max2 = self._compute_window(tid, pose2_best)
            win2 = [d for d in dets2 if u_min2 <= d.centroid_u <= u_max2]
            pts2 = self.backproj_stage.backproject_detections(win2, pose2_best)

            # Dual-Pass Deduplication
            n_dedup, n_overlap = self.dual_stage.deduplicate_front_back(pts1, pts2)

            gt_val = self.gt_map.get(tid, np.nan)
            summary = TreeSummary(
                tree_id=tid,
                gt_buschel=gt_val if not np.isnan(gt_val) else None,
                closest_frame=f"p1:{name_p1}_p2:{name_p2}",
                raw_windowed_count=n_dedup,
                predicted_buschel=n_dedup / 5.03,
                column_window=(u_min1, u_max1),
                landmarks_3d=[(float(pt[0]), float(pt[1]), float(pt[2])) for pt in pts1]
            )
            summaries.append(summary)

        # Thinning Decisions (Dual-Pass Factor: 5.03 det/Büschel)
        decisions = []
        for s in summaries:
            pred_b = s.predicted_buschel
            needs_thin = pred_b > PLACEHOLDER_THINNING_TARGET_BUSCHEL
            rough_rem = max(0.0, pred_b - PLACEHOLDER_THINNING_TARGET_BUSCHEL) if needs_thin else 0.0
            p_score = (pred_b - PLACEHOLDER_THINNING_TARGET_BUSCHEL) / PLACEHOLDER_THINNING_TARGET_BUSCHEL if needs_thin else 0.0
            decisions.append(ThinningDecisionItem(
                tree_id=s.tree_id,
                gt_buschel=s.gt_buschel,
                predicted_buschel=pred_b,
                needs_thinning=needs_thin,
                rough_removal_buschel=rough_rem,
                priority_score=p_score
            ))

        return self._finalize_evaluation(summaries, decisions, mode_label="DUAL-PASS")

    def _run_single_pass_pipeline(self, max_trees: int):
        left_dir = os.path.expanduser(config.LEFT_IMAGES_DIR)
        left_files = sorted([f for f in os.listdir(left_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        tree_frame_map: Dict[int, Tuple[str, str]] = {}
        unique_frames = set()

        for tid in range(1, max_trees + 1):
            if tid not in self.tree_map: continue
            tx, ty = self.tree_map[tid]["x"], self.tree_map[tid]["y"]
            dists = []
            for f in left_files:
                ts = utils.parse_frame_timestamp(f)
                pose = interpolate_pose(ts, self.t_arr, self.E_arr, self.N_arr, self.U_arr, self.cos_arr, self.sin_arr)
                dists.append(np.linalg.norm(np.array([pose["e"], pose["n"]]) - np.array([tx, ty])))
            s_idx = np.argsort(dists)
            fa, fb = left_files[s_idx[0]], left_files[s_idx[1]] if len(s_idx) > 1 else left_files[s_idx[0]]
            tree_frame_map[tid] = (fa, fb)
            unique_frames.add(fa); unique_frames.add(fb)

        frame_dets = self.seg_stage.run(left_dir, sorted(list(unique_frames)))
        summaries_map = self.window_stage.run(self.tree_map, self.gt_map, tree_frame_map, frame_dets, interpolate_pose, self.t_arr, self.E_arr, self.N_arr, self.U_arr, self.cos_arr, self.sin_arr)
        summaries_map = self.backproj_stage.run(summaries_map, interpolate_pose, self.t_arr, self.E_arr, self.N_arr, self.U_arr, self.cos_arr, self.sin_arr)
        summaries_map = self.refine_stage.run(summaries_map)
        decisions = self.decision_stage.run(summaries_map)
        return self._finalize_evaluation(list(summaries_map.values()), decisions, mode_label="SINGLE-PASS")

    def _get_cached_or_live_detections(self, frame_idx: int, frame_name: str) -> List[FlowerDetection]:
        cache_pkl = os.path.join(CACHE_DIR, f"{frame_name}.pkl.gz")
        if os.path.exists(cache_pkl):
            with gzip.open(cache_pkl, "rb") as cf:
                raw_list = pickle.load(cf)
            return [FlowerDetection(item["instance_id"], item["confidence"], item["centroid_u"], item["centroid_v"], frame_name) for item in raw_list]

        ts, img_cv = self.dual_stage.unpack_rec_frame(frame_idx)
        if self.seg_stage.predictor is None:
            from step1_segmentation_tiled import load_segmentation_model
            self.seg_stage.predictor = load_segmentation_model(config.SEGMENTATION_MODEL_PATH, self.conf_thresh, device="cuda")

        from step1_segmentation_tiled import segment_frame
        flowers = segment_frame(img_cv, self.seg_stage.predictor, self.conf_thresh)
        dets = [FlowerDetection(fl["instance_id"], fl["confidence"], fl["centroid_u"], fl["centroid_v"], frame_name) for fl in flowers]
        raw_list = [{"instance_id": fl["instance_id"], "confidence": fl["confidence"], "centroid_u": fl["centroid_u"], "centroid_v": fl["centroid_v"]} for fl in flowers]
        with gzip.open(cache_pkl, "wb") as cf:
            pickle.dump(raw_list, cf)
        return dets

    def _compute_window(self, tid: int, pose: dict) -> Tuple[float, float]:
        u_pts = []
        for test_id in (tid, tid - 1, tid + 1):
            if test_id in self.tree_map:
                e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                dx = self.tree_map[test_id]["x"] - e
                dy = self.tree_map[test_id]["y"] - n
                dz = 0.5 - u
                p_base = np.array([cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy, dz])
                p_cam = self.R_base_to_cam.T @ (p_base - self.t_base_to_cam)
                if p_cam[2] > 0:
                    pu = self.fx * (p_cam[0] / p_cam[2]) + self.cx
                    u_pts.append(pu)

        if not u_pts: return 0.0, self.raw_w
        u_curr = u_pts[0]
        left_c = [p for p in u_pts if p < u_curr]
        right_c = [p for p in u_pts if p > u_curr]
        u_prev_val = max(left_c) if left_c else 0.0
        u_next_val = min(right_c) if right_c else self.raw_w
        return max(0.0, (u_prev_val + u_curr) / 2.0), min(self.raw_w, (u_curr + u_next_val) / 2.0)

    def _finalize_evaluation(self, summaries: List[TreeSummary], decisions: List[ThinningDecisionItem], mode_label: str):
        gt_vals, pred_vals = [], []
        for d in decisions:
            if d.gt_buschel is not None and not np.isnan(d.gt_buschel):
                gt_vals.append(d.gt_buschel)
                pred_vals.append(d.predicted_buschel)

        r_val, p_val = pearsonr(gt_vals, pred_vals)
        mae_val = float(np.mean(np.abs(np.array(pred_vals) - np.array(gt_vals))))

        print(f"\n=== EVALUATION RESULTS ({mode_label}, CONF_THRESH={self.conf_thresh:.2f}, n={len(gt_vals)} Elstar Trees) ===")
        print(f"Pearson Correlation (r): {r_val:.4f} (p={p_val:.4e})")
        print(f"Mean Absolute Error (MAE): {mae_val:.2f} Büschel / tree")

        plot1_path = os.path.join(DATA_DIR, "thinning_plan_plot.png")
        plot2_path = os.path.join(DATA_DIR, "flower_summary_table.png")
        plot3_path = os.path.join(DATA_DIR, "2d_cluster_map.png")

        save_thinning_plan_plot(decisions, PLACEHOLDER_THINNING_TARGET_BUSCHEL, PLACEHOLDER_THINNING_AGRONOMIC_MAX_BUSCHEL, plot1_path)
        save_flower_summary_table(decisions, plot2_path)
        trees_coords = {tid: (self.tree_map[tid]["x"], self.tree_map[tid]["y"]) for tid in self.tree_map}
        save_2d_cluster_map(trees_coords, summaries, config.POSE_TRAJECTORY_FILE, plot3_path)

        plan_out = [
            {
                "tree_id": d.tree_id,
                "gt_buschel": d.gt_buschel if d.gt_buschel is not None and not np.isnan(d.gt_buschel) else "Befruchter",
                "predicted_buschel": d.predicted_buschel,
                "needs_thinning": d.needs_thinning,
                "rough_removal_buschel": d.rough_removal_buschel,
                "priority_score": d.priority_score
            }
            for d in decisions
        ]
        with open(os.path.join(DATA_DIR, "thinning_plan_trial2.json"), "w") as f:
            json.dump(plan_out, f, indent=4)

        report_lines = [
            "==================================================",
            f"   OFFLINE PIPELINE TRIAL 2 EXECUTION REPORT ({mode_label})",
            "==================================================",
            f"Mode:                               {mode_label}",
            f"Segmentation Confidence Threshold:  {self.conf_thresh:.2f} (EXPLICITLY FIXED)",
            f"Evaluated Trees Count:              65",
            f"Valid Elstar GT Trees:              {len(gt_vals)}",
            f"Pearson Correlation (r):            {r_val:.4f}",
            f"Mean Absolute Error (MAE):          {mae_val:.2f} Büschel / tree",
            "=================================================="
        ]
        with open(os.path.join(DATA_DIR, "execution_report_trial2.log"), "w") as f:
            f.write("\n".join(report_lines))

        print(f"Trial 2 outputs saved in: {DATA_DIR}")
        return r_val, mae_val

if __name__ == "__main__":
    orchestrator = OfflinePipelineTrial2(conf_thresh=0.60, mode="dual_pass")
    orchestrator.run_pipeline(max_trees=65)
