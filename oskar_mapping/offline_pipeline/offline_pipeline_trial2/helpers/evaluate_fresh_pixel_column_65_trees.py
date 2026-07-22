#!/usr/bin/env python3
"""Fresh 65-Tree Pixel-Column Attribution Evaluation (Rerunning Mask R-CNN from Scratch with Caching).

Placed in: offline_pipeline_trial2/
Reruns Mask R-CNN segmentation live on GPU for each of the 65 GT trees (no stale cache reads).
Saves fresh results into offline_pipeline_trial2/fresh_cache/ for fast resumption.
Calculates pixel-column windowed attribution [u_min, u_max] and recomputes Pearson r and MAE.
"""

import os
import sys
import cv2
import yaml
import gzip
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

WORKSPACE_DIR = "/home/workstation/ros2_ws/src/oskar"
OFFLINE_DIR = os.path.join(WORKSPACE_DIR, "oskar_mapping/offline_pipeline")
DATASET_DIR = os.path.join(WORKSPACE_DIR, "datasets/Blossom2024")
TRIAL2_DIR = os.path.join(OFFLINE_DIR, "offline_pipeline_trial2")
FRESH_CACHE_DIR = os.path.join(TRIAL2_DIR, "fresh_cache")
os.makedirs(FRESH_CACHE_DIR, exist_ok=True)

sys.path.append(OFFLINE_DIR)
import config
import utils
from step1_segmentation_tiled import load_segmentation_model, segment_frame
from step4_backprojection import interpolate_pose, get_base_to_camera_transform

# 1. Read Excel GT Sheet
excel_path = os.path.join(DATASET_DIR, "Blütenstand (15.04.2024).xlsx")
df_gt = pd.read_excel(excel_path, sheet_name=0)

gt_map = {}
befruchter_gt_set = set()

for idx, row in df_gt.iterrows():
    try:
        baum_val = row["Baum"]
        if pd.isna(baum_val):
            continue
        baum_id = int(baum_val)
        buschel_val = row["Büschel"]
        
        if str(buschel_val).strip().lower() == "befruchter":
            befruchter_gt_set.add(baum_id)
            gt_map[baum_id] = np.nan
        else:
            num_val = pd.to_numeric(buschel_val, errors='coerce')
            gt_map[baum_id] = float(num_val) if not pd.isna(num_val) else np.nan
    except Exception:
        continue

# 2. Load Tree Map & Trajectory
with open(config.TREE_MAP_FILE, 'r') as f:
    tree_map = yaml.safe_load(f)["trees"]

traj_data = np.load(config.POSE_TRAJECTORY_FILE)
t_arr, E_arr, N_arr, U_arr = traj_data["t"], traj_data["E"], traj_data["N"], traj_data["U"]
cos_arr, sin_arr = np.cos(traj_data["yaw"]), np.sin(traj_data["yaw"])

left_dir = os.path.expanduser(config.LEFT_IMAGES_DIR)
left_files = sorted([f for f in os.listdir(left_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# 3. Camera Calibration & Intrinsics
with open(config.LEFT_CALIB_FILE, 'r') as f:
    cdata = yaml.safe_load(f)
P_raw = np.array(cdata["projectionMatrix"], dtype=np.float64)
fx, fy = P_raw[0, 0], P_raw[1, 1]
cx, cy = P_raw[0, 2], P_raw[1, 2]
raw_w = cdata.get("image_width", 5344)

camera_side = getattr(config, "CAMERA_SIDE", "left")
camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
R_base_to_cam, t_base_to_cam = get_base_to_camera_transform(camera_side, camera_xyz)

def project_enu_to_pixel(enu_pt, pose):
    e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    dx, dy, dz = enu_pt[0] - e, enu_pt[1] - n, enu_pt[2] - u
    p_base = np.array([cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy, dz])
    p_cam = R_base_to_cam.T @ (p_base - t_base_to_cam)
    x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]
    if z_c <= 0:
        return None, None, z_c
    pixel_u = fx * (x_c / z_c) + cx
    pixel_v = fy * (y_c / z_c) + cy
    return pixel_u, pixel_v, z_c

# 4. Load Mask R-CNN Predictor
conf_thresh = getattr(config, "SEG_CONFIDENCE_THRESHOLD", 0.6)
model_path = getattr(config, "SEGMENTATION_MODEL_PATH")

tree_frame_map = {}
unique_frames = set()

for tid in range(1, 66):
    if tid not in tree_map:
        continue
    tx, ty = tree_map[tid]["x"], tree_map[tid]["y"]
    dists = []
    for f in left_files:
        ts = utils.parse_frame_timestamp(f)
        pose = interpolate_pose(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
        cam_dist = np.linalg.norm(np.array([pose["e"], pose["n"]]) - np.array([tx, ty]))
        dists.append(cam_dist)
    best_idx = int(np.argmin(dists))
    best_frame = left_files[best_idx]
    tree_frame_map[tid] = best_frame
    unique_frames.add(best_frame)

print(f"Total unique image frames to process for 65 trees: {len(unique_frames)}")

# Predictor lazy load
predictor = None

fresh_segmentations = {}
sorted_unique = sorted(list(unique_frames))

for idx, frame_fname in enumerate(sorted_unique):
    base_name = os.path.splitext(frame_fname)[0]
    cache_pkl = os.path.join(FRESH_CACHE_DIR, f"{base_name}.pkl.gz")
    
    if os.path.exists(cache_pkl):
        with gzip.open(cache_pkl, "rb") as cf:
            flowers = pickle.load(cf)
        fresh_segmentations[frame_fname] = flowers
        print(f"[{idx+1}/{len(sorted_unique)}] Loaded from fresh cache {frame_fname}: {len(flowers)} flowers")
    else:
        if predictor is None:
            print(f"Loading Mask R-CNN model from {model_path} (conf_thresh={conf_thresh})...")
            predictor = load_segmentation_model(model_path, conf_thresh, device="cuda")
            
        img_path = os.path.join(left_dir, frame_fname)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"Error loading image {img_path}")
            continue
        flowers = segment_frame(img_cv, predictor, conf_thresh)
        
        # Save to fresh_cache without unneeded masks to keep fast
        flowers_slim = []
        for fl in flowers:
            flowers_slim.append({
                "instance_id": fl["instance_id"],
                "confidence": fl["confidence"],
                "centroid_u": fl["centroid_u"],
                "centroid_v": fl["centroid_v"]
            })
            
        with gzip.open(cache_pkl, "wb") as cf:
            pickle.dump(flowers_slim, cf)
            
        fresh_segmentations[frame_fname] = flowers_slim
        print(f"[{idx+1}/{len(sorted_unique)}] Live Segmented & Saved {frame_fname}: {len(flowers_slim)} raw flowers detected")

print("\n=== FRESH MASK R-CNN PIXEL-COLUMN ATTRIBUTION (TREES 1..65) ===")
print(f"{'Tree ID':<8} | {'GT':<10} | {'Closest Frame':<28} | {'Fresh Frame Raw':<16} | {'Fresh Windowed Raw':<18} | {'Fresh Büschel (÷4.5)':<20} | {'Column Window [u_min, u_max] px':<32}")
print("-" * 140)

results = []

for tid in range(1, 66):
    if tid not in tree_map:
        continue
    best_frame = tree_frame_map[tid]
    ts = utils.parse_frame_timestamp(best_frame)
    pose = interpolate_pose(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
    
    # Project trunks
    u_curr, u_prev, u_next = None, None, None
    if tid in tree_map:
        pu, _, _ = project_enu_to_pixel(np.array([tree_map[tid]["x"], tree_map[tid]["y"], 0.5]), pose)
        u_curr = pu
    if tid - 1 in tree_map:
        pu, _, _ = project_enu_to_pixel(np.array([tree_map[tid-1]["x"], tree_map[tid-1]["y"], 0.5]), pose)
        u_prev = pu
    if tid + 1 in tree_map:
        pu, _, _ = project_enu_to_pixel(np.array([tree_map[tid+1]["x"], tree_map[tid+1]["y"], 0.5]), pose)
        u_next = pu
        
    u_left = (u_prev + u_curr) / 2.0 if u_prev is not None else 0.0
    u_right = (u_curr + u_next) / 2.0 if u_next is not None else float(raw_w)
    
    u_min = max(0.0, min(u_left, u_right))
    u_max = min(float(raw_w), max(u_left, u_right))
    
    flowers = fresh_segmentations.get(best_frame, [])
    whole_raw = len(flowers)
    windowed_raw = 0
    for fl in flowers:
        u_c = fl.get("centroid_u")
        if u_c is not None and u_min <= u_c <= u_max:
            windowed_raw += 1
            
    gt_val = gt_map.get(tid, np.nan)
    is_befruchter = tid in befruchter_gt_set or pd.isna(gt_val)
    gt_str = "Befruchter" if is_befruchter else f"{gt_val:.0f}"
    
    windowed_buschel = windowed_raw / 4.5
    
    results.append({
        "tree_id": tid,
        "gt": gt_val,
        "best_frame": best_frame,
        "whole_raw": whole_raw,
        "windowed_raw": windowed_raw,
        "windowed_buschel": windowed_buschel,
        "is_befruchter": is_befruchter,
        "u_min": u_min,
        "u_max": u_max
    })
    
    win_str = f"[{u_min:.1f}, {u_max:.1f}]"
    print(f"{tid:<8} | {gt_str:<10} | {best_frame:<28} | {whole_raw:<16} | {windowed_raw:<18} | {windowed_buschel:<20.2f} | {win_str:<32}")

print("-" * 140)
print()

df = pd.DataFrame(results)
df_valid = df.dropna(subset=["gt"]).copy()
gt_v = df_valid["gt"].values

pred_fresh_win = df_valid["windowed_buschel"].values
r_fresh, p_fresh = pearsonr(gt_v, pred_fresh_win)
mae_fresh = np.mean(np.abs(pred_fresh_win - gt_v))

print("=== FRESH VS CACHED EVALUATION COMPARISON (n=60 Elstar Trees) ===")
print(f"Cached Windowed Pixel-Column  : Pearson r = 0.9373, MAE = 23.19 Büschel / tree")
print(f"FRESH Live Mask R-CNN Windowed : Pearson r = {r_fresh:.4f} (p={p_fresh:.4e}), MAE = {mae_fresh:.2f} Büschel / tree")
