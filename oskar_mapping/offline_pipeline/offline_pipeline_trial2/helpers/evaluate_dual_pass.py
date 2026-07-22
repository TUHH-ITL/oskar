"""Dual-Pass (Front + Back Pass) Evaluation Script for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/helpers/.
Unpacks Pass 2 frames directly from SAMSON3 .rec container on the fly.
"""

import os
import sys
import cv2
import yaml
import struct
import gzip
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Dict, List, Tuple

TRIAL2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_DIR = os.path.dirname(TRIAL2_DIR)
WORKSPACE_DIR = os.path.dirname(os.path.dirname(OFFLINE_DIR))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "datasets/Blossom2024")

sys.path.append(OFFLINE_DIR)
sys.path.append(TRIAL2_DIR)

import config
import utils
from step4_backprojection import interpolate_pose, get_base_to_camera_transform
from pipeline_types import FlowerDetection
from trial2_step1_segmentation import SegmentationStage

REC_PATH = os.path.join(DATASET_DIR, "2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1/2024-04-15_10-59-41_A27_Bluete_SAMSON3_1713171581.rec")
CACHE_DIR = os.path.join(TRIAL2_DIR, "fresh_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def unpack_rec_frame(frame_idx: int) -> Tuple[float, np.ndarray]:
    """Unpacks timestamp and BGR image for any frame_idx from SAMSON3 .rec file."""
    frame_size = 24551456
    with open(REC_PATH, "rb") as f:
        f.seek(12 + frame_idx * frame_size)
        hdr = f.read(32)
        payloadbytes, imageid, secs, nsecs = struct.unpack("<QQQQ", hdr)
        ts = float(secs) + float(nsecs) * 1e-9
        img_bytes = f.read(payloadbytes)
        raw_img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(4608, 5328)
        bgr_img = cv2.cvtColor(raw_img, cv2.COLOR_BayerBG2BGR)
    return ts, bgr_img

def get_segmented_detections(frame_idx: int, frame_name: str, seg_stage: SegmentationStage) -> List[FlowerDetection]:
    """Runs GPU Mask R-CNN segmentation or retrieves from cache for a specific frame."""
    cache_pkl = os.path.join(CACHE_DIR, f"{frame_name}.pkl.gz")
    if os.path.exists(cache_pkl):
        with gzip.open(cache_pkl, "rb") as cf:
            raw_list = pickle.load(cf)
        dets = []
        for item in raw_list:
            dets.append(FlowerDetection(
                instance_id=item["instance_id"], confidence=item["confidence"],
                centroid_u=item["centroid_u"], centroid_v=item["centroid_v"],
                frame_name=frame_name
            ))
        return dets

    # Unpack from .rec and segment live on GPU
    ts, img_cv = unpack_rec_frame(frame_idx)
    if seg_stage.predictor is None:
        from step1_segmentation_tiled import load_segmentation_model
        seg_stage.predictor = load_segmentation_model(config.SEGMENTATION_MODEL_PATH, seg_stage.conf_thresh, device="cuda")

    from step1_segmentation_tiled import segment_frame
    flowers = segment_frame(img_cv, seg_stage.predictor, seg_stage.conf_thresh)
    
    dets = []
    raw_list = []
    for fl in flowers:
        det = FlowerDetection(
            instance_id=fl["instance_id"], confidence=fl["confidence"],
            centroid_u=fl["centroid_u"], centroid_v=fl["centroid_v"],
            frame_name=frame_name
        )
        dets.append(det)
        raw_list.append({
            "instance_id": fl["instance_id"], "confidence": fl["confidence"],
            "centroid_u": fl["centroid_u"], "centroid_v": fl["centroid_v"]
        })

    with gzip.open(cache_pkl, "wb") as cf:
        pickle.dump(raw_list, cf)

    return dets

def run_dual_pass_experiment(conf_thresh: float = 0.60):
    print(f"=== DUAL-PASS FRONT/BACK EVALUATION (conf_thresh={conf_thresh:.2f}) ===", flush=True)

    # Load Calibration
    with open(config.LEFT_CALIB_FILE, 'r') as f:
        cdata = yaml.safe_load(f)
    P_raw = np.array(cdata["projectionMatrix"], dtype=np.float64)
    fx, fy = P_raw[0, 0], P_raw[1, 1]
    cx, cy = P_raw[0, 2], P_raw[1, 2]
    raw_w = 5328.0

    camera_side = getattr(config, "CAMERA_SIDE", "left")
    camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
    R_base_to_cam, t_base_to_cam = get_base_to_camera_transform(camera_side, camera_xyz)

    # Load Tree Map
    with open(config.TREE_MAP_FILE, 'r') as f:
        tree_map = yaml.safe_load(f)["trees"]

    # Load Trajectory
    traj_data = np.load(config.POSE_TRAJECTORY_FILE)
    t_arr, E_arr, N_arr, U_arr, yaw_arr = traj_data["t"], traj_data["E"], traj_data["N"], traj_data["U"], traj_data["yaw"]
    cos_arr, sin_arr = np.cos(yaw_arr), np.sin(yaw_arr)

    # Load GT
    excel_path = os.path.join(DATASET_DIR, "Blütenstand (15.04.2024).xlsx")
    df_gt = pd.read_excel(excel_path, sheet_name=0)
    gt_map = {}
    for idx, row in df_gt.iterrows():
        try:
            b_val = row["Baum"]
            if pd.isna(b_val):
                continue
            tid = int(b_val)
            buschel_val = row["Büschel"]
            if str(buschel_val).strip().lower() == "befruchter":
                gt_map[tid] = np.nan
            else:
                num_val = pd.to_numeric(buschel_val, errors='coerce')
                gt_map[tid] = float(num_val) if not pd.isna(num_val) else np.nan
        except Exception:
            continue

    # Extract all timestamps from .rec
    num_rec_frames = (os.path.getsize(REC_PATH) - 12) // 24551456
    rec_timestamps = []
    with open(REC_PATH, "rb") as f:
        for i in range(num_rec_frames):
            f.seek(12 + i * 24551456)
            hdr = f.read(32)
            _, _, secs, nsecs = struct.unpack("<QQQQ", hdr)
            rec_timestamps.append(float(secs) + float(nsecs) * 1e-9)

    # Classify frames into Pass 1 (Front, ts < 1713171700) and Pass 2 (Back, ts > 1713172000)
    pass1_info = []
    pass2_info = []
    for idx, ts in enumerate(rec_timestamps):
        pose = interpolate_pose(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
        yaw = pose["yaw"]
        if -0.6 <= yaw <= 0.2 and ts < 1713171700:
            pass1_info.append((idx, ts, pose))
        elif (2.5 <= yaw <= 3.14 or -3.14 <= yaw <= -2.5) and ts > 1713172000:
            pass2_info.append((idx, ts, pose))

    tree_best_frames = {}
    seg_stage = SegmentationStage(OFFLINE_DIR, TRIAL2_DIR, conf_thresh)

    # Process per-tree front and back frames
    tree_dets_map = {}

    for tid in sorted(list(tree_map.keys())):
        tx, ty = tree_map[tid]["x"], tree_map[tid]["y"]

        d1 = [np.hypot(p[2]["e"] - tx, p[2]["n"] - ty) for p in pass1_info]
        idx1_best, ts1_best, pose1_best = pass1_info[np.argmin(d1)]

        d2 = [np.hypot(p[2]["e"] - tx, p[2]["n"] - ty) for p in pass2_info]
        idx2_best, ts2_best, pose2_best = pass2_info[np.argmin(d2)]

        name_p1 = f"tree_{(idx1_best - 113):04d}" if idx1_best >= 113 else f"rec_frame_{idx1_best:04d}"
        name_p2 = f"pass2_rec_frame_{idx2_best:04d}"

        tree_best_frames[tid] = {
            "p1_idx": idx1_best, "p1_name": name_p1, "p1_pose": pose1_best,
            "p2_idx": idx2_best, "p2_name": name_p2, "p2_pose": pose2_best
        }

        dets1 = get_segmented_detections(idx1_best, name_p1, seg_stage)
        dets2 = get_segmented_detections(idx2_best, name_p2, seg_stage)

        tree_dets_map[tid] = (dets1, dets2)

    def project_enu_to_pixel(enu_pt, pose):
        e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        dx, dy, dz = enu_pt[0] - e, enu_pt[1] - n, enu_pt[2] - u
        p_base = np.array([cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy, dz])
        p_cam = R_base_to_cam.T @ (p_base - t_base_to_cam)
        x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]
        if z_c <= 0:
            return None, None
        pixel_u = fx * (x_c / z_c) + cx
        pixel_v = fy * (y_c / z_c) + cy
        return pixel_u, pixel_v

    def compute_window(tid, pose):
        u_pts = []
        for test_id in (tid, tid - 1, tid + 1):
            if test_id in tree_map:
                pu, _ = project_enu_to_pixel(np.array([tree_map[test_id]["x"], tree_map[test_id]["y"], 0.5]), pose)
                if pu is not None:
                    u_pts.append(pu)

        if not u_pts:
            return 0.0, raw_w

        u_curr = u_pts[0]
        u_left_candidates = [p for p in u_pts if p < u_curr]
        u_right_candidates = [p for p in u_pts if p > u_curr]

        u_prev_val = max(u_left_candidates) if u_left_candidates else 0.0
        u_next_val = min(u_right_candidates) if u_right_candidates else raw_w

        u_min = max(0.0, (u_prev_val + u_curr) / 2.0)
        u_max = min(raw_w, (u_curr + u_next_val) / 2.0)
        return u_min, u_max

    def backproject_dets(dets, pose):
        e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_map_to_base = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        t_map_to_base = np.array([e, n, u])

        pts_3d = []
        for det in dets:
            z_c = 1.3
            x_c = (det.centroid_u - cx) * z_c / fx
            y_c = (det.centroid_v - cy) * z_c / fy
            p_cam = np.array([x_c, y_c, z_c])
            p_base = R_base_to_cam @ p_cam + t_base_to_cam
            p_map = R_map_to_base @ p_base + t_map_to_base
            pts_3d.append(p_map)
        return np.array(pts_3d) if len(pts_3d) > 0 else np.zeros((0, 3))

    sweep_radii = [0.015, 0.03, 0.05, 0.08, 0.12]
    eval_results = []

    for r_match in sweep_radii:
        gt_vals = []
        front_counts = []
        back_counts = []
        sum_counts = []
        dedup_counts = []

        for tid in sorted(list(tree_map.keys())):
            gt_val = gt_map.get(tid, np.nan)
            if np.isnan(gt_val):
                continue

            info = tree_best_frames[tid]
            dets1, dets2 = tree_dets_map[tid]

            # Pass 1 Windowing
            u_min1, u_max1 = compute_window(tid, info["p1_pose"])
            win1 = [d for d in dets1 if u_min1 <= d.centroid_u <= u_max1]

            # Pass 2 Windowing
            u_min2, u_max2 = compute_window(tid, info["p2_pose"])
            win2 = [d for d in dets2 if u_min2 <= d.centroid_u <= u_max2]

            pts1 = backproject_dets(win1, info["p1_pose"])
            pts2 = backproject_dets(win2, info["p2_pose"])

            n_front = len(pts1)
            n_back = len(pts2)
            n_sum = n_front + n_back

            matched_count = 0
            if len(pts1) > 0 and len(pts2) > 0:
                dist_mat = np.linalg.norm(pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :], axis=2)
                matched_count = int(np.sum(np.min(dist_mat, axis=1) <= r_match))

            n_dedup = n_sum - matched_count

            gt_vals.append(gt_val)
            front_counts.append(n_front)
            back_counts.append(n_back)
            sum_counts.append(n_sum)
            dedup_counts.append(n_dedup)

        gt_arr = np.array(gt_vals)
        front_arr = np.array(front_counts)
        back_arr = np.array(back_counts)
        sum_arr = np.array(sum_counts)
        dedup_arr = np.array(dedup_counts)

        # Evaluate Front Pass only
        factor_front = float(np.sum(front_arr) / np.sum(gt_arr))
        pred_front = front_arr / factor_front
        r_front, _ = pearsonr(gt_arr, pred_front)
        mae_front = float(np.mean(np.abs(pred_front - gt_arr)))

        # Evaluate Dual-Pass Sum (no dedup)
        factor_sum = float(np.sum(sum_arr) / np.sum(gt_arr))
        pred_sum = sum_arr / factor_sum
        r_sum, _ = pearsonr(gt_arr, pred_sum)
        mae_sum = float(np.mean(np.abs(pred_sum - gt_arr)))

        # Evaluate Dual-Pass Deduplicated
        factor_dedup = float(np.sum(dedup_arr) / np.sum(gt_arr))
        pred_dedup = dedup_arr / factor_dedup
        r_dedup, p_dedup = pearsonr(gt_arr, pred_dedup)
        mae_dedup = float(np.mean(np.abs(pred_dedup - gt_arr)))

        eval_results.append({
            "r_match_cm": r_match * 100,
            "r_front": r_front, "mae_front": mae_front, "factor_front": factor_front,
            "r_sum": r_sum, "mae_sum": mae_sum, "factor_sum": factor_sum,
            "r_dedup": r_dedup, "mae_dedup": mae_dedup, "factor_dedup": factor_dedup,
            "tot_front": int(np.sum(front_arr)),
            "tot_back": int(np.sum(back_arr)),
            "tot_sum": int(np.sum(sum_arr)),
            "tot_dedup": int(np.sum(dedup_arr))
        })

    print("\n==========================================================================================", flush=True)
    print("                 DUAL-PASS (FRONT PASS + BACK PASS) DEDUPLICATION RESULTS                 ", flush=True)
    print("==========================================================================================", flush=True)
    print(f"Front Pass Only (West Side)  : r = {eval_results[0]['r_front']:.4f} | MAE = {eval_results[0]['mae_front']:.2f} Büschel | Factor = {eval_results[0]['factor_front']:.2f} det/Büschel", flush=True)
    print(f"Dual-Pass Sum (Front + Back) : r = {eval_results[0]['r_sum']:.4f} | MAE = {eval_results[0]['mae_sum']:.2f} Büschel | Factor = {eval_results[0]['factor_sum']:.2f} det/Büschel", flush=True)
    print("-" * 90, flush=True)
    for res in eval_results:
        print(f"Match Radius r = {res['r_match_cm']:4.1f} cm | Deduplicated Dual-Pass: r = {res['r_dedup']:.4f} | MAE = {res['mae_dedup']:.2f} Büschel | Factor = {res['factor_dedup']:.2f} det/Büschel", flush=True)
        print(f"   Detections -> Front: {res['tot_front']} | Back: {res['tot_back']} | Sum: {res['tot_sum']} | Dedup: {res['tot_dedup']}\n", flush=True)

if __name__ == "__main__":
    run_dual_pass_experiment(conf_thresh=0.60)
