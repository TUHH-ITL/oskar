#!/usr/bin/env python3
"""Updated Evaluation Script comparing Whole-Frame vs Windowed Pixel-Column Attribution across 65 Trees.

Fixes:
1. Applies 2D column boundaries [u_left, u_right] computed from adjacent trunk ENU projections.
2. Correctly filters Befruchter vs Main Elstar trees.
3. Computes Pearson r and MAE for both unwindowed (whole frame) and windowed (cropped column).
"""

import os
import sys
import yaml
import gzip
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

WORKSPACE_DIR = "/home/workstation/ros2_ws/src/oskar"
OFFLINE_DIR = os.path.join(WORKSPACE_DIR, "oskar_mapping/offline_pipeline")
DATASET_DIR = os.path.join(WORKSPACE_DIR, "datasets/Blossom2024")

sys.path.append(OFFLINE_DIR)
import config
import utils
from step4_backprojection import interpolate_pose, get_base_to_camera_transform

# 1. Read Excel GT
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

# Tree map & trajectory
with open(config.TREE_MAP_FILE, 'r') as f:
    tree_map = yaml.safe_load(f)["trees"]

traj_data = np.load(config.POSE_TRAJECTORY_FILE)
t_arr, E_arr, N_arr, U_arr = traj_data["t"], traj_data["E"], traj_data["N"], traj_data["U"]
cos_arr, sin_arr = np.cos(traj_data["yaw"]), np.sin(traj_data["yaw"])

left_dir = os.path.expanduser(config.LEFT_IMAGES_DIR)
left_files = sorted([f for f in os.listdir(left_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

# Camera calibration
with open(config.LEFT_CALIB_FILE, 'r') as f:
    cdata = yaml.safe_load(f)
P_raw = np.array(cdata["projectionMatrix"], dtype=np.float64)
# Mask R-CNN detections are in raw image pixel coordinates (5344 x 3744)
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

cache_dir = os.path.join(config.OUTPUT_DIR, "cache")
results = []

print("=== PIXEL-COLUMN ATTRIBUTION EVALUATION (TREES 1..65) ===")
print(f"{'Tree ID':<8} | {'GT':<10} | {'Closest Frame':<28} | {'Whole Frame Raw':<16} | {'Windowed Raw':<14} | {'Windowed Büschel':<18} | {'Column Window [u_min, u_max] px':<32}")
print("-" * 135)

for tid in range(1, 66):
    if tid not in tree_map:
        continue
    tx, ty = tree_map[tid]["x"], tree_map[tid]["y"]
    
    # Find closest frame
    dists = []
    for f in left_files:
        ts = utils.parse_frame_timestamp(f)
        pose = interpolate_pose(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
        cam_dist = np.linalg.norm(np.array([pose["e"], pose["n"]]) - np.array([tx, ty]))
        dists.append(cam_dist)
        
    best_idx = int(np.argmin(dists))
    best_frame = left_files[best_idx]
    ts = utils.parse_frame_timestamp(best_frame)
    pose = interpolate_pose(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
    
    # Project trunks for tid-1, tid, tid+1
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
    
    # Load cache
    base_name = os.path.splitext(best_frame)[0]
    pkl_path = os.path.join(cache_dir, f"{base_name}.pkl.gz")
    
    whole_frame_raw = 0
    windowed_raw = 0
    
    if os.path.exists(pkl_path):
        with gzip.open(pkl_path, "rb") as cf:
            cache_data = pickle.load(cf)
        flowers = cache_data.get("flowers", [])
        whole_frame_raw = len(flowers)
        
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
        "whole_raw": whole_frame_raw,
        "windowed_raw": windowed_raw,
        "windowed_buschel": windowed_buschel,
        "is_befruchter": is_befruchter,
        "u_min": u_min,
        "u_max": u_max
    })
    
    win_str = f"[{u_min:.1f}, {u_max:.1f}]"
    print(f"{tid:<8} | {gt_str:<10} | {best_frame:<28} | {whole_frame_raw:<16} | {windowed_raw:<14} | {windowed_buschel:<18.2f} | {win_str:<32}")

print("-" * 135)
print()

df = pd.DataFrame(results)

# 1. WHOLE FRAME (UNWINDOWED) EVALUATION
df_valid = df.dropna(subset=["gt"]).copy()
gt_v = df_valid["gt"].values
pred_whole = df_valid["whole_raw"].values / 4.5
r_w, _ = pearsonr(gt_v, pred_whole)
mae_w = np.mean(np.abs(pred_whole - gt_v))

# 2. WINDOWED EVALUATION (PROPER PIXEL COLUMN)
pred_win = df_valid["windowed_buschel"].values
r_win, _ = pearsonr(gt_v, pred_win)
mae_win = np.mean(np.abs(pred_win - gt_v))

print("=== COMPARATIVE EVALUATION (n=60 Elstar Trees with Ground Truth) ===")
print(f"1. Whole Frame (Unwindowed)  : Pearson r = {r_w:.4f}, MAE = {mae_w:.2f} Büschel / tree")
print(f"2. Pixel-Column Windowed    : Pearson r = {r_win:.4f}, MAE = {mae_win:.2f} Büschel / tree")
print()

# TREE 39, TREE 10, TREE 46 COMPARISON
for t_check in [10, 39, 46]:
    r_c = df[df["tree_id"] == t_check].iloc[0]
    print(f"Tree {t_check:<2} (GT={r_c['gt']}): Whole-Frame Raw = {r_c['whole_raw']:<4} | Windowed Raw = {r_c['windowed_raw']:<4} ({r_c['windowed_buschel']:.1f} Büschel) | Window = [{r_c['u_min']:.1f}, {r_c['u_max']:.1f}]")
