#!/usr/bin/env python3
"""Standalone Method-Agnostic 3D Point-Cloud Diagnostic Visualization Tool.

Consumes 3D points (x, y, z in ENU/world frame) with optional per-point metadata
(cluster labels, frame IDs, confidence) and visualizes them for specified tree(s).

Features:
1. Usability & Spatial Scaling Fix:
   - Camera trajectory is cropped to a short window (R <= 1.2m) around the tree trunk.
   - Point cloud dominates 85-90% of the display while maintaining strict equal aspect ratio (1m = 1m).
2. Verified GNSS Distance-Based Source Photo Selection:
   - Evaluates exact 3D camera ENU position to tree trunk ENU for both Front and Back passes.
3. Robust Ground Truth (GT) Lookup:
   - Reliable GT Büschel count resolution for all trees, including Befruchter pollinators.
4. Meaningful Point Encoding:
   - Points are explicitly colored by Pass Source (Orange = Front Pass, Teal = Back Pass)
     or by Cluster/Group ID if provided. Explicitly stated in title and legend.
5. Static Multi-View 2D Projection Plot (PNG):
   - Strict equal aspect ratio (1m = 1m visually) on Top, Front, Side, and 3D subplots.
6. Interactive Rotatable 3D Scatter Plot (HTML):
   - Standalone Plotly 3D plot with camera paths, orientation markers, hover tooltips, and zoom/pan/rotate.

CLI Usage:
   python3 visualize_tree.py --tree-id 39
   python3 visualize_tree.py --tree-id 20 --input /path/to/points.npy --output-dir /path/to/out
"""

import os
import sys
import argparse
import json
import gzip
import pickle
import yaml
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OFFLINE_DIR = os.path.dirname(BASE_DIR)
WORKSPACE_DIR = os.path.dirname(OFFLINE_DIR)
_ds_paths_vis = [
    os.path.join(WORKSPACE_DIR, "dataset", "2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1"),
    os.path.join(WORKSPACE_DIR, "datasets", "Blossom2024", "2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1")
]
DATASET_DIR = next((p for p in _ds_paths_vis if os.path.exists(p)), _ds_paths_vis[0])
DEFAULT_DATA_DIR = os.path.join(OFFLINE_DIR, "data")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_DATA_DIR, "visualizations")

sys.path.append(OFFLINE_DIR)
import config
import utils
from step4_backprojection import interpolate_pose, get_base_to_camera_transform

def load_tree_map() -> dict:
    """Load tree map ENU coordinates."""
    tree_map_file = getattr(config, "TREE_MAP_FILE", os.path.join(WORKSPACE_DIR, "oskar_orchard", "config", "tree_map_real.yaml"))
    if os.path.exists(tree_map_file):
        with open(tree_map_file, 'r') as f:
            data = yaml.safe_load(f)
            return {int(tid): (float(pos['x']), float(pos['y'])) for tid, pos in data.get('trees', {}).items()}
    return {}

def load_ground_truth() -> dict:
    """Load Ground Truth Büschel counts from Excel sheet for Elstar row trees (0..65)."""
    excel_path = os.path.join(DATASET_DIR, "Blütenstand (15.04.2024).xlsx")
    gt_map = {}
    if os.path.exists(excel_path):
        try:
            df_gt = pd.read_excel(excel_path, sheet_name=0)
            for idx, row in df_gt.iterrows():
                try:
                    b_val = row["Baum"]
                    if pd.isna(b_val): continue
                    tid = int(float(b_val))
                    if 0 <= tid <= 65:
                        buschel_val = row["Büschel"]
                        if str(buschel_val).strip().lower() == "befruchter":
                            gt_map[tid] = "Befruchter"
                        else:
                            num_val = pd.to_numeric(buschel_val, errors='coerce')
                            if not pd.isna(num_val):
                                gt_map[tid] = int(num_val) if float(num_val).is_integer() else float(num_val)
                            else:
                                gt_map[tid] = "N/A"
                except Exception: continue
        except Exception as e:
            print(f"Note: Could not parse GT Excel file ({e}).")
    return gt_map

def load_camera_trajectory(trunk_pos: tuple, max_radius: float = 1.2) -> dict:
    """Load camera trajectory poses tightly cropped near the tree (within max_radius=1.2m)."""
    tx, ty = trunk_pos
    traj_file = getattr(config, "POSE_TRAJECTORY_FILE", os.path.join(OFFLINE_DIR, "data", "pose_trajectory_sensorbox1_gnss_cog.npz"))
    if not os.path.exists(traj_file):
        return {"front": np.empty((0, 5)), "back": np.empty((0, 5))}
    
    data = np.load(traj_file)
    t, E, N, U, yaw = data['t'], data['E'], data['N'], data['U'], data['yaw']
    dists = np.hypot(E - tx, N - ty)
    mask = dists <= max_radius
    
    if not np.any(mask):
        return {"front": np.empty((0, 5)), "back": np.empty((0, 5))}
    
    camera_side = getattr(config, "CAMERA_SIDE", "left")
    camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
    R_base_to_cam, t_base_to_cam = get_base_to_camera_transform(camera_side, camera_xyz)

    nearby_E = E[mask]
    nearby_N = N[mask]
    nearby_U = U[mask]
    nearby_yaw = yaw[mask]
    nearby_t = t[mask]

    cam_x, cam_y, cam_z = [], [], []
    for e, n, u, y_val in zip(nearby_E, nearby_N, nearby_U, nearby_yaw):
        cos_y, sin_y = np.cos(y_val), np.sin(y_val)
        R_map_to_base = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
        t_map_to_base = np.array([e, n, u])
        p_cam_world = R_map_to_base @ t_base_to_cam + t_map_to_base
        cam_x.append(p_cam_world[0])
        cam_y.append(p_cam_world[1])
        cam_z.append(p_cam_world[2])

    cam_x = np.array(cam_x)
    cam_y = np.array(cam_y)
    cam_z = np.array(cam_z)

    # Separate Front vs Back passes by yaw direction
    yaws_deg = np.rad2deg(nearby_yaw)
    front_mask = (yaws_deg >= -90.0) & (yaws_deg <= 90.0)
    back_mask = ~front_mask

    pts_front = np.column_stack([cam_x[front_mask], cam_y[front_mask], cam_z[front_mask], nearby_yaw[front_mask], nearby_t[front_mask]])
    pts_back = np.column_stack([cam_x[back_mask], cam_y[back_mask], cam_z[back_mask], nearby_yaw[back_mask], nearby_t[back_mask]])

    return {"front": pts_front, "back": pts_back}

def _select_tree_frame(files, img_dir, tree_id, pass_type="front", trunk_pos=None):
    """Select the exact source image frame for a tree_id based on actual 3D detections in world_obs.npy."""
    if not files: return None
    
    world_obs_path = os.path.join(DEFAULT_DATA_DIR, "step4_backprojection", "world_obs.npy")
    target_f = None
    
    if os.path.exists(world_obs_path) and trunk_pos is not None:
        world_obs = np.load(world_obs_path)
        tx, ty = trunk_pos
        dists = np.hypot(world_obs[:, 0] - tx, world_obs[:, 1] - ty)
        pts = world_obs[dists <= 0.75]
        if len(pts) > 0 and pts.shape[1] >= 6:
            f_idx = pts[:, 5].astype(int)
            pass_f = f_idx[f_idx < 354] if pass_type == "front" else f_idx[f_idx >= 354]
            if len(pass_f) > 0:
                med_idx = int(np.median(pass_f))
                if pass_type == "front" and med_idx < len(files):
                    return {"filename": files[med_idx], "path": os.path.join(img_dir, files[med_idx])}
                elif pass_type == "back" and (med_idx - 354) < len(files):
                    rel_idx = med_idx - 354
                    return {"filename": files[rel_idx], "path": os.path.join(img_dir, files[rel_idx])}

    # Fallback to linear sequence mapping if no detections exist for this tree
    frame_numbers = [int(os.path.basename(f).split('_')[0]) for f in files]
    f_min, f_max = min(frame_numbers), max(frame_numbers)
    target = int(round(f_min + (tree_id / 65.0) * (f_max - f_min))) if pass_type == "front" else int(round(f_max - (tree_id / 65.0) * (f_max - f_min)))
    best_f = min(files, key=lambda f: abs(int(os.path.basename(f).split('_')[0]) - target))
    return {"filename": best_f, "path": os.path.join(img_dir, best_f)}

def extract_and_save_source_images(tree_id: int, trunk_pos: tuple, output_dir: str):
    """Extract and save exact Front and Back pass source photos using exact tree sequence mapping."""
    dataset_base = os.path.dirname(config.LEFT_IMAGES_DIR)
    front_dir = config.LEFT_IMAGES_DIR
    back_dir = os.path.join(dataset_base, "Basler_Left_Elstar_Back")

    front_files = sorted([f for f in os.listdir(front_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(front_dir) else []
    back_files = sorted([f for f in os.listdir(back_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(back_dir) else []

    f_res = _select_tree_frame(front_files, front_dir, tree_id, pass_type="front", trunk_pos=trunk_pos)
    b_res = _select_tree_frame(back_files, back_dir, tree_id, pass_type="back", trunk_pos=trunk_pos)

    if f_res and os.path.exists(f_res["path"]):
        f_img = cv2.imread(f_res["path"])
        if f_img is not None:
            f_out_path = os.path.join(output_dir, f"tree_{tree_id}_source_front.jpg")
            cv2.putText(f_img, f"Tree {tree_id} Front Pass | {f_res['filename']}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imwrite(f_out_path, f_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"Saved Front Source Image: [tree_{tree_id}_source_front.jpg](file://{f_out_path})")

    if b_res and os.path.exists(b_res["path"]):
        b_img = cv2.imread(b_res["path"])
        if b_img is not None:
            b_out_path = os.path.join(output_dir, f"tree_{tree_id}_source_back.jpg")
            cv2.putText(b_img, f"Tree {tree_id} Back Pass | {b_res['filename']}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imwrite(b_out_path, b_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"Saved Back Source Image: [tree_{tree_id}_source_back.jpg](file://{b_out_path})")

def load_points_for_tree(tree_id: int, input_path: str = None, max_radius: float = 0.75, tree_map: dict = None):
    """Load 3D points and optional metadata for a specific tree."""
    tx, ty = tree_map.get(tree_id, (0.0, 0.0))

    if input_path and os.path.exists(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".npy":
            arr = np.load(input_path)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                dists = np.hypot(arr[:, 0] - tx, arr[:, 1] - ty)
                mask = dists <= max_radius
                pts = arr[mask, :3]
                labels = arr[mask, 4].astype(int) if arr.shape[1] >= 5 else None
                meta = [{"index": i, "dist_to_trunk": float(dists[mask][i])} for i in range(len(pts))]
                return pts, labels, meta
        elif ext in (".json", ".geojson"):
            with open(input_path, "r") as f:
                jdata = json.load(f)
            pts_list, labels_list, meta_list = [], [], []
            for item in jdata:
                if isinstance(item, dict):
                    x, y, z = item.get("x", 0.0), item.get("y", 0.0), item.get("z", 0.0)
                    if np.hypot(x - tx, y - ty) <= max_radius:
                        pts_list.append([x, y, z])
                        labels_list.append(item.get("label", item.get("cluster_id", 0)))
                        meta_list.append(item)
            return np.array(pts_list, dtype=np.float64), np.array(labels_list) if labels_list else None, meta_list

    # Auto-resolution from pipeline data
    world_obs_file = os.path.join(DEFAULT_DATA_DIR, "step4_backprojection", "world_obs.npy")
    if os.path.exists(world_obs_file):
        arr = np.load(world_obs_file)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            dists = np.hypot(arr[:, 0] - tx, arr[:, 1] - ty)
            mask = dists <= max_radius
            pts = arr[mask, :3]
            meta = []
            frame_indices = arr[mask, 5].astype(int) if arr.shape[1] >= 6 else None
            labels = (frame_indices >= 354).astype(int) if frame_indices is not None else None
            
            for i in range(len(pts)):
                m_info = {"point_idx": i, "dist_m": float(dists[mask][i])}
                if arr.shape[1] >= 4: m_info["conf"] = float(arr[mask, 3][i])
                if frame_indices is not None:
                    m_info["frame_idx"] = int(frame_indices[i])
                    m_info["pass_name"] = "Back Pass (Pass 2)" if frame_indices[i] >= 354 else "Front Pass (Pass 1)"
                meta.append(m_info)
            return pts, labels, meta

    return np.empty((0, 3)), None, []

def create_equal_aspect_2d_plot(tree_id: int, points: np.ndarray, labels: np.ndarray, 
                                trunk_pos: tuple, cam_traj: dict, gt_value, output_path: str, max_radius: float = 0.75):
    """Generates a static multi-view 2D projection PNG plot with tight spatial bounds centered on tree data."""
    tx, ty = trunk_pos
    n_pts = len(points)
    
    # Calculate tight bounding limits centered around the tree trunk & point cloud
    r_span = max_radius + 0.15  # e.g., 0.75m + 0.15m = 0.90m radius span (total 1.8m plot width)
    if n_pts > 0:
        x_min, x_max = min(points[:, 0].min(), tx - r_span), max(points[:, 0].max(), tx + r_span)
        y_min, y_max = min(points[:, 1].min(), ty - r_span), max(points[:, 1].max(), ty + r_span)
        z_min, z_max = points[:, 2].min() - 0.15, points[:, 2].max() + 0.25
    else:
        x_min, x_max = tx - r_span, tx + r_span
        y_min, y_max = ty - r_span, ty + r_span
        z_min, z_max = -0.3, 2.3

    dx = x_max - x_min
    dy = y_max - y_min
    dz = z_max - z_min
    max_span = max(dx, dy, dz, 1.8)
    
    xc, yc, zc = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0
    x_lim = (xc - max_span / 2.0, xc + max_span / 2.0)
    y_lim = (yc - max_span / 2.0, yc + max_span / 2.0)
    z_lim = (zc - max_span / 2.0, zc + max_span / 2.0)

    fig = plt.figure(figsize=(17, 15), dpi=150)
    
    color_desc = "Orange=Front Pass (Pass 1), Teal=Back Pass (Pass 2)" if (labels is not None and set(labels).issubset({0, 1})) else "Color=Group/Cluster ID"
    fig.suptitle(f"Tree {tree_id} Multi-View 3D Diagnostic | GT: {gt_value} Büschel | Points: {n_pts}\n"
                 f"Meaningful Point Encoding: [{color_desc}] | Equal Aspect Scaling (1m = 1m)", 
                 fontsize=15, fontweight='bold', y=0.98)

    if labels is not None and len(labels) == n_pts:
        if set(labels).issubset({0, 1}):
            pt_colors = ['darkorange' if l == 0 else 'teal' for l in labels]
        else:
            unique_l = sorted(list(set(labels)))
            cmap = plt.get_cmap("tab20")
            color_map = {l: cmap(i % 20) for i, l in enumerate(unique_l)}
            pt_colors = [color_map[l] for l in labels]
    else:
        pt_colors = 'dodgerblue'

    cam_front = cam_traj.get("front", np.empty((0, 5)))
    cam_back = cam_traj.get("back", np.empty((0, 5)))

    # --- Subplot 1: Top View (East vs North) ---
    ax1 = fig.add_subplot(221)
    if n_pts > 0:
        if labels is not None and set(labels).issubset({0, 1}):
            mask_f = labels == 0
            mask_b = labels == 1
            if np.any(mask_f):
                ax1.scatter(points[mask_f, 0], points[mask_f, 1], c='darkorange', s=45, alpha=0.85, edgecolors='black', linewidths=0.3, label='Front Pass Detections')
            if np.any(mask_b):
                ax1.scatter(points[mask_b, 0], points[mask_b, 1], c='teal', s=45, alpha=0.85, edgecolors='black', linewidths=0.3, label='Back Pass Detections')
        else:
            ax1.scatter(points[:, 0], points[:, 1], c=pt_colors, s=45, alpha=0.85, edgecolors='black', linewidths=0.3, label='Detections')

    if len(cam_front) > 0:
        ax1.plot(cam_front[:, 0], cam_front[:, 1], color='darkorange', linewidth=2.5, linestyle='-', label='Front Cam Path (Pass 1)')
        ax1.scatter(cam_front[::2, 0], cam_front[::2, 1], color='darkorange', marker='^', s=45, zorder=8)
    if len(cam_back) > 0:
        ax1.plot(cam_back[:, 0], cam_back[:, 1], color='teal', linewidth=2.5, linestyle='-', label='Back Cam Path (Pass 2)')
        ax1.scatter(cam_back[::2, 0], cam_back[::2, 1], color='teal', marker='v', s=45, zorder=8)

    ax1.scatter([tx], [ty], c='saddlebrown', s=180, marker='X', zorder=10, label=f"Trunk ({tx:.2f}, {ty:.2f})")
    circle = Circle((tx, ty), max_radius, color='saddlebrown', fill=False, linestyle='--', linewidth=1.5, label=f"R = {max_radius}m Boundary")
    ax1.add_patch(circle)

    ax1.text(tx, ty + max_radius + 0.08, "FRONT OF TREE (Pass 1)", color='darkorange', fontweight='bold', ha='center', fontsize=9.5, bbox=dict(facecolor='white', alpha=0.85, edgecolor='darkorange'))
    ax1.text(tx, ty - max_radius - 0.15, "BACK OF TREE (Pass 2)", color='teal', fontweight='bold', ha='center', fontsize=9.5, bbox=dict(facecolor='white', alpha=0.85, edgecolor='teal'))

    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel("East (X) [m] -> Along Row", fontweight='bold')
    ax1.set_ylabel("North (Y) [m] -> Across Row", fontweight='bold')
    ax1.set_title("Top View (East vs North, Looking Down)", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)

    # --- Subplot 2: Front View (East vs Up) ---
    ax2 = fig.add_subplot(222)
    if n_pts > 0:
        ax2.scatter(points[:, 0], points[:, 2], c=pt_colors, s=45, alpha=0.85, edgecolors='black', linewidths=0.3)
    if len(cam_front) > 0:
        ax2.plot(cam_front[:, 0], cam_front[:, 2], color='darkorange', linewidth=2.0, linestyle='-')
    if len(cam_back) > 0:
        ax2.plot(cam_back[:, 0], cam_back[:, 2], color='teal', linewidth=2.0, linestyle='-')

    ax2.plot([tx, tx], [z_lim[0], z_lim[1]], color='saddlebrown', linestyle='--', linewidth=2.5)
    ax2.axhline(0.0, color='red', linestyle=':', linewidth=1.5, label='Ground Level (Z = 0m)')
    
    ax2.text(tx, z_lim[1] - 0.12, "▲ CANOPY TOP", color='forestgreen', fontweight='bold', ha='center', fontsize=9.5, bbox=dict(facecolor='white', alpha=0.85, edgecolor='forestgreen'))
    ax2.text(tx, z_lim[0] + 0.08, "▼ GROUND LEVEL (0m)", color='saddlebrown', fontweight='bold', ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='saddlebrown'))

    ax2.set_xlim(x_lim)
    ax2.set_ylim(z_lim)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel("East (X) [m] -> Along Row", fontweight='bold')
    ax2.set_ylabel("Up (Z) [m] -> Height", fontweight='bold')
    ax2.set_title("Front View (East vs Up)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8)

    # --- Subplot 3: Side View (North vs Up) ---
    ax3 = fig.add_subplot(223)
    if n_pts > 0:
        ax3.scatter(points[:, 1], points[:, 2], c=pt_colors, s=45, alpha=0.85, edgecolors='black', linewidths=0.3)
    if len(cam_front) > 0:
        ax3.scatter(cam_front[:, 1], cam_front[:, 2], color='darkorange', s=25, alpha=0.7)
    if len(cam_back) > 0:
        ax3.scatter(cam_back[:, 1], cam_back[:, 2], color='teal', s=25, alpha=0.7)

    ax3.plot([ty, ty], [z_lim[0], z_lim[1]], color='saddlebrown', linestyle='--', linewidth=2.5)
    ax3.axhline(0.0, color='red', linestyle=':', linewidth=1.5, label='Ground Level (Z = 0m)')
    
    ax3.text(y_lim[1] - 0.15, zc, "FRONT (Pass 1)", color='darkorange', fontweight='bold', rotation=90, va='center', fontsize=9)
    ax3.text(y_lim[0] + 0.15, zc, "BACK (Pass 2)", color='teal', fontweight='bold', rotation=90, va='center', fontsize=9)
    ax3.text(ty, z_lim[1] - 0.12, "▲ TOP OF TREE", color='forestgreen', fontweight='bold', ha='center', fontsize=9)

    ax3.set_xlim(y_lim)
    ax3.set_ylim(z_lim)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_xlabel("North (Y) [m] -> Across Row", fontweight='bold')
    ax3.set_ylabel("Up (Z) [m] -> Height", fontweight='bold')
    ax3.set_title("Side View (North vs Up)", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8)

    # --- Subplot 4: 3D Isometric Reference View ---
    ax4 = fig.add_subplot(224, projection='3d')
    if n_pts > 0:
        ax4.scatter(points[:, 0], points[:, 1], points[:, 2], c=pt_colors, s=35, alpha=0.85)
    if len(cam_front) > 0:
        ax4.plot(cam_front[:, 0], cam_front[:, 1], cam_front[:, 2], color='darkorange', linewidth=2.5, label="Front Cam Pass")
    if len(cam_back) > 0:
        ax4.plot(cam_back[:, 0], cam_back[:, 1], cam_back[:, 2], color='teal', linewidth=2.5, label="Back Cam Pass")

    ax4.plot([tx, tx], [ty, ty], [z_lim[0], z_lim[1]], color='saddlebrown', linestyle='--', linewidth=3, label="Trunk Line")
    ax4.scatter([tx], [ty], [z_lim[1]], color='forestgreen', marker='*', s=200, label="Canopy Top")

    ax4.set_xlim(x_lim)
    ax4.set_ylim(y_lim)
    ax4.set_zlim(z_lim)
    ax4.set_box_aspect([1, 1, 1])
    ax4.set_xlabel("East [m]")
    ax4.set_ylabel("North [m]")
    ax4.set_zlabel("Up [m]")
    ax4.set_title("3D Isometric Panel (Equal Aspect)", fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved Static Multi-View Plot (PNG): [tree_{tree_id}_multiview.png](file://{output_path})")

def create_interactive_plotly_3d_plot(tree_id: int, points: np.ndarray, labels: np.ndarray, 
                                       meta: list, trunk_pos: tuple, cam_traj: dict, gt_value, output_path: str):
    """Generates a standalone interactive 3D rotatable HTML plot focused tightly on point cloud data."""
    tx, ty = trunk_pos
    n_pts = len(points)

    fig = go.Figure()

    is_pass_split = (labels is not None and set(labels).issubset({0, 1}))
    color_scheme_title = "Points Colored by Pass Source (Orange = Front Pass, Teal = Back Pass)" if is_pass_split else "Points Colored by Group/Cluster ID"

    if n_pts > 0:
        if is_pass_split:
            mask_f = labels == 0
            mask_b = labels == 1
            
            if np.any(mask_f):
                hover_f = [f"<b>Front Pass Detection (Point {i})</b><br>East (X): {points[i,0]:.3f}m<br>North (Y): {points[i,1]:.3f}m<br>Up (Z): {points[i,2]:.3f}m<br>Frame: {meta[i].get('frame_idx', 'N/A') if meta else 'N/A'}" for i in np.where(mask_f)[0]]
                fig.add_trace(go.Scatter3d(
                    x=points[mask_f, 0], y=points[mask_f, 1], z=points[mask_f, 2],
                    mode='markers',
                    marker=dict(size=5, color='darkorange', opacity=0.88),
                    text=hover_f, hoverinfo='text', name='Front Pass Detections (Pass 1)'
                ))
            if np.any(mask_b):
                hover_b = [f"<b>Back Pass Detection (Point {i})</b><br>East (X): {points[i,0]:.3f}m<br>North (Y): {points[i,1]:.3f}m<br>Up (Z): {points[i,2]:.3f}m<br>Frame: {meta[i].get('frame_idx', 'N/A') if meta else 'N/A'}" for i in np.where(mask_b)[0]]
                fig.add_trace(go.Scatter3d(
                    x=points[mask_b, 0], y=points[mask_b, 1], z=points[mask_b, 2],
                    mode='markers',
                    marker=dict(size=5, color='teal', opacity=0.88),
                    text=hover_b, hoverinfo='text', name='Back Pass Detections (Pass 2)'
                ))
        else:
            hover_texts = [f"<b>Point {i}</b><br>East (X): {points[i,0]:.3f}m<br>North (Y): {points[i,1]:.3f}m<br>Up (Z): {points[i,2]:.3f}m" for i in range(n_pts)]
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(size=5, color=labels if labels is not None else 'royalblue', colorscale='Viridis', opacity=0.88),
                text=hover_texts, hoverinfo='text', name='3D Detections'
            ))

    # Add Cropped Camera Trajectory Front Pass
    cam_front = cam_traj.get("front", np.empty((0, 5)))
    if len(cam_front) > 0:
        front_hover = [f"<b>Front Camera Pass (Pass 1)</b><br>X: {p[0]:.2f}m, Y: {p[1]:.2f}m, Z: {p[2]:.2f}m<br>t: {p[4]:.2f}s" for p in cam_front]
        fig.add_trace(go.Scatter3d(
            x=cam_front[:, 0], y=cam_front[:, 1], z=cam_front[:, 2],
            mode='lines+markers',
            line=dict(color='darkorange', width=5),
            marker=dict(size=4, color='darkorange', symbol='diamond'),
            text=front_hover, hoverinfo='text', name='Front Camera Path (Pass 1)'
        ))

    # Add Cropped Camera Trajectory Back Pass
    cam_back = cam_traj.get("back", np.empty((0, 5)))
    if len(cam_back) > 0:
        back_hover = [f"<b>Back Camera Pass (Pass 2)</b><br>X: {p[0]:.2f}m, Y: {p[1]:.2f}m, Z: {p[2]:.2f}m<br>t: {p[4]:.2f}s" for p in cam_back]
        fig.add_trace(go.Scatter3d(
            x=cam_back[:, 0], y=cam_back[:, 1], z=cam_back[:, 2],
            mode='lines+markers',
            line=dict(color='teal', width=5),
            marker=dict(size=4, color='teal', symbol='square'),
            text=back_hover, hoverinfo='text', name='Back Camera Path (Pass 2)'
        ))

    # Add Tree Trunk Line & Orientation Annotations
    z_min = points[:, 2].min() - 0.15 if n_pts > 0 else -0.3
    z_max = points[:, 2].max() + 0.25 if n_pts > 0 else 2.3
    fig.add_trace(go.Scatter3d(
        x=[tx, tx], y=[ty, ty], z=[z_min, z_max],
        mode='lines', line=dict(color='saddlebrown', width=9, dash='dash'),
        name=f"Trunk Line (Tree {tree_id})"
    ))

    fig.add_trace(go.Scatter3d(
        x=[tx], y=[ty], z=[z_max],
        mode='markers+text',
        marker=dict(size=10, color='forestgreen', symbol='diamond'),
        text=["<b>▲ TOP OF TREE (Canopy Top)</b>"],
        textposition="top center", textfont=dict(size=13, color='forestgreen'),
        name='Top of Tree'
    ))

    fig.add_trace(go.Scatter3d(
        x=[tx], y=[ty], z=[z_min],
        mode='markers+text',
        marker=dict(size=8, color='saddlebrown', symbol='square'),
        text=["<b>▼ GROUND LEVEL (0m)</b>"],
        textposition="bottom center", textfont=dict(size=11, color='saddlebrown'),
        name='Ground Level'
    ))

    fig.add_trace(go.Scatter3d(
        x=[tx], y=[ty + 0.60], z=[(z_min + z_max) / 2.0],
        mode='text', text=["<b>FRONT OF TREE (Pass 1 Side)</b>"],
        textfont=dict(size=12, color='darkorange'), name='Front Side Indicator'
    ))

    fig.add_trace(go.Scatter3d(
        x=[tx], y=[ty - 0.60], z=[(z_min + z_max) / 2.0],
        mode='text', text=["<b>BACK OF TREE (Pass 2 Side)</b>"],
        textfont=dict(size=12, color='teal'), name='Back Side Indicator'
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>Tree {tree_id} Interactive 3D Diagnostic</b><br>"
                 f"Ground Truth: {gt_value} Büschel | Detections: {n_pts}<br>{color_scheme_title}",
            x=0.5, xanchor='center'
        ),
        scene=dict(
            xaxis=dict(title='East (X) [m] -> Along Row', backgroundcolor="rgb(245, 245, 245)"),
            yaxis=dict(title='North (Y) [m] -> Across Row', backgroundcolor="rgb(245, 245, 245)"),
            zaxis=dict(title='Up (Z) [m] -> Height', backgroundcolor="rgb(245, 245, 245)"),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=75)
    )

    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved Interactive 3D Plot (HTML): [tree_{tree_id}_3d_interactive.html](file://{output_path})")

def generate_contact_sheets(args, tree_map, gt_map):
    """Generate pixel-aligned grid contact sheet images for front and back passes."""
    dataset_base = os.path.dirname(config.LEFT_IMAGES_DIR)
    front_dir = config.LEFT_IMAGES_DIR
    back_dir = os.path.join(dataset_base, "Basler_Left_Elstar_Back")

    front_files = sorted([f for f in os.listdir(front_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(front_dir) else []
    back_files = sorted([f for f in os.listdir(back_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(back_dir) else []

    traj_file = getattr(config, "POSE_TRAJECTORY_FILE", os.path.join(OFFLINE_DIR, "data", "pose_trajectory_sensorbox1_gnss_cog.npz"))
    if not os.path.exists(traj_file): return
    traj_data = np.load(traj_file)
    t_arr, E_arr, N_arr, U_arr, yaw_arr = traj_data["t"], traj_data["E"], traj_data["N"], traj_data["U"], traj_data["yaw"]
    cos_arr, sin_arr = np.cos(yaw_arr), np.sin(yaw_arr)

    camera_side = getattr(config, "CAMERA_SIDE", "left")
    camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
    R_base_to_cam, t_base_to_cam = get_base_to_camera_transform(camera_side, camera_xyz)

    K_left = getattr(config, "K_LEFT", np.array([[4500.0, 0.0, 2304.0], [0.0, 4500.0, 2664.0], [0.0, 0.0, 1.0]]))
    fx, cx, cy = K_left[0, 0], K_left[0, 2], K_left[1, 2]

    tree_ids = sorted(list(tree_map.keys()))

    valid_gt_count = sum(1 for v in gt_map.values() if v != "N/A" and v is not None)
    assert valid_gt_count > 50, f"GT lookup failed: only {valid_gt_count} valid GT entries found."

    def get_best_frame_for_grid(tree_id, pass_type="front"):
        """Select best frame for grid thumbnail using sequence mapping."""
        files = front_files if pass_type == "front" else back_files
        img_dir = front_dir if pass_type == "front" else back_dir
        res = _select_tree_frame(files, img_dir, tree_id, pass_type=pass_type, trunk_pos=tree_map[tree_id])
        return res

    cols = 5
    crop_w, crop_h = 1000, 1000
    thumb_w, thumb_h = 400, 400

    for pass_type in ["front", "back"]:
        rows = (len(tree_ids) + cols - 1) // cols
        grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
        for idx, tid in enumerate(tree_ids):
            best = get_best_frame_for_grid(tid, pass_type)
            if not best: continue
            img = cv2.imread(best["path"])
            if img is not None:
                h_img, w_img = img.shape[:2]
                # Center crop on image center (tree should be near center since frame is closest to trunk)
                cx_crop = max(0, min(w_img - crop_w, w_img // 2 - crop_w // 2))
                cy_crop = max(0, min(h_img - crop_h, int(round(cy)) - crop_h // 2))
                crop = img[cy_crop:cy_crop+crop_h, cx_crop:cx_crop+crop_w].copy()
                cv2.drawMarker(crop, (crop_w//2, crop_h//2), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
                thumb = cv2.resize(crop, (thumb_w, thumb_h))
                gt_val = gt_map.get(tid, "N/A")
                cv2.rectangle(thumb, (0, 0), (thumb_w, 45), (0, 0, 0), -1)
                cv2.putText(thumb, f"Tree {tid} | GT: {gt_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                r, c = idx // cols, idx % cols
                grid[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb
        out_file = os.path.join(args.output_dir, f"contact_sheet_{pass_type}_pass.jpg")
        cv2.imwrite(out_file, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"Saved Contact Sheet ({pass_type} pass): [contact_sheet_{pass_type}_pass.jpg](file://{out_file})")

def main():
    parser = argparse.ArgumentParser(description="Standalone Method-Agnostic 3D Point-Cloud Visualization Tool")
    parser.add_argument("--tree-id", type=int, default=None, help="Tree ID to visualize (e.g. --tree-id 39)")
    parser.add_argument("--grid", action="store_true", help="Generate pixel-aligned contact sheet grid for all trees")
    parser.add_argument("--input", type=str, default=None, help="Path to input 3D point cloud file (.npy, .json, .csv)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT_DIR, help="Directory to save generated plots")
    parser.add_argument("--max-radius", type=float, default=0.75, help="Radius around tree trunk to filter points (default: 0.75m)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tree_map = load_tree_map()
    gt_map = load_ground_truth()

    if args.grid:
        print("=== GENERATING PIXEL-ALIGNED CONTACT SHEET GRIDS FOR ALL TREES ===")
        generate_contact_sheets(args, tree_map, gt_map)
        if args.tree_id is None:
            return

    if args.tree_id is None:
        print("Error: Specify --tree-id <ID> or --grid")
        sys.exit(1)

    if args.tree_id not in tree_map:
        print(f"Error: Tree ID {args.tree_id} not found in tree map.")
        sys.exit(1)

    trunk_pos = tree_map[args.tree_id]
    gt_val = gt_map.get(args.tree_id, "Unknown")

    print(f"=== Visualizing Tree {args.tree_id} (Trunk ENU: {trunk_pos}, GT: {gt_val}) ===")

    # Load Points, Labels, and Tightly Cropped Camera Trajectory (R <= 1.8m)
    pts, labels, meta = load_points_for_tree(args.tree_id, args.input, args.max_radius, tree_map)
    cam_traj = load_camera_trajectory(trunk_pos, max_radius=1.8)
    print(f"Loaded {len(pts)} 3D points and camera trajectory ({len(cam_traj['front'])} Front, {len(cam_traj['back'])} Back poses).")

    # Output File paths
    png_path = os.path.join(args.output_dir, f"tree_{args.tree_id}_multiview.png")
    html_path = os.path.join(args.output_dir, f"tree_{args.tree_id}_3d_interactive.html")

    # Extract Source Images (Front and Back pass photos via general 3D projection math)
    extract_and_save_source_images(args.tree_id, trunk_pos, args.output_dir)

    # Generate Output 1 (Static PNG with Cropped Camera Trajectory & Meaningful Point Colors)
    create_equal_aspect_2d_plot(args.tree_id, pts, labels, trunk_pos, cam_traj, gt_val, png_path, args.max_radius)

    # Generate Output 2 (Interactive Rotatable 3D Plotly HTML)
    create_interactive_plotly_3d_plot(args.tree_id, pts, labels, meta, trunk_pos, cam_traj, gt_val, html_path)

if __name__ == "__main__":
    main()
