#!/usr/bin/env python3
"""Reusable DBSCAN 3D Diagnostic Tool for Apple Flower Mapping.

Generates 3D multi-frame comparisons (Frame A vs Frame B) and DBSCAN cluster plots.
Includes:
- Frame selection: Frame A = best/closest frame; Frame B = frame offset by ~+5 frames.
- Structural reference: Vertical line segment at the tree trunk (x, y) coordinates.
- Reporting: Raw point counts, cluster counts, noise counts, and largest-cluster stats.
"""

import os
import argparse
import pickle
import gzip
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

WORKSPACE_DIR = "/home/workstation/ros2_ws/src/oskar"
OFFLINE_DIR = os.path.join(WORKSPACE_DIR, "oskar_mapping/offline_pipeline")
ARTIFACTS_DIR = "/home/workstation/.gemini/antigravity-ide/brain/fc8795bb-cef3-4fe2-b749-cc8c1221f41e"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

import sys
sys.path.append(OFFLINE_DIR)
import config
import utils
from step3_depth_fusion import depth_fusion_frame
from step4_backprojection import get_base_to_camera_transform, interpolate_pose

# Load tree map
with open(config.TREE_MAP_FILE, 'r') as f:
    t_map = yaml.safe_load(f)
trees = {int(tid): (float(pos['x']), float(pos['y'])) for tid, pos in t_map['trees'].items()}

# Load ground truth sheet if available
gt_counts = {
    27: 6,
    28: 5,
    26: 16,
    20: 34,
    57: 79,
    64: 119,
    42: 181,
    39: 277,
    61: 214
}

# Load trajectory
traj_data = np.load(config.POSE_TRAJECTORY_FILE)
t_arr, E_arr, N_arr, U_arr = traj_data["t"], traj_data["E"], traj_data["N"], traj_data["U"]
cos_arr, sin_arr = np.cos(traj_data["yaw"]), np.sin(traj_data["yaw"])

# Calibration & Transforms
lc = utils.load_calibration_yaml(config.LEFT_CALIB_FILE)
K = np.array(lc["cameraMatrix"], dtype=np.float64).reshape(3, 3)
camera_side = getattr(config, "CAMERA_SIDE", "right")
camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
R_base_to_cam, t_base_to_cam = get_base_to_camera_transform(camera_side, camera_xyz)

erosion_radius = config.DEPTH_FUSION_EROSION_RADIUS
erosion_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2 * erosion_radius + 1, 2 * erosion_radius + 1),
)

left_files = sorted([
    f for f in os.listdir(config.LEFT_IMAGES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

def get_frame_selection(tid, offset_frames=5):
    """Select Frame A (closest) and Frame B (+offset_frames shift)."""
    t_pos = np.array(trees[tid])
    dists = []
    for idx, f in enumerate(left_files):
        ts = utils.parse_frame_timestamp(f)
        pose = interpolate_pose(ts, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
        cam_pos = np.array([pose['e'], pose['n']])
        dist = np.linalg.norm(cam_pos - t_pos)
        dists.append(dist)
        
    best_a_idx = int(np.argmin(dists))
    dist_a = dists[best_a_idx]
    
    # Target frame B offset
    idx_b = best_a_idx + offset_frames
    if idx_b >= len(left_files):
        idx_b = best_a_idx - offset_frames
    if idx_b < 0:
        idx_b = best_a_idx
        
    dist_b = dists[idx_b]
    
    return best_a_idx, left_files[best_a_idx], dist_a, idx_b, left_files[idx_b], dist_b

def get_frame_3d_points(frame_name):
    """Load cache and return 3D ENU world coordinates."""
    base_name = os.path.splitext(frame_name)[0]
    pkl_path = os.path.join(config.OUTPUT_DIR, "cache", f"{base_name}.pkl.gz")
    if not os.path.exists(pkl_path):
        return np.empty((0, 4))
        
    with gzip.open(pkl_path, "rb") as f:
        cache_data = pickle.load(f)
    flowers = cache_data["flowers"]
    depth = cache_data["depth"]
    
    frame_detections = depth_fusion_frame(depth, flowers, K, erosion_kernel)
    
    img_path = os.path.join(config.LEFT_IMAGES_DIR, frame_name)
    timestamp = utils.parse_frame_timestamp(img_path)
    pose = interpolate_pose(timestamp, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
    e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]
    
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_map_to_base = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    t_map_to_base = np.array([e, n, u])
    
    points = []
    for det in frame_detections:
        p_cam = np.array([det["x_cam"], det["y_cam"], det["z_cam"]])
        p_base = R_base_to_cam @ p_cam + t_base_to_cam
        p_map = R_map_to_base @ p_base + t_map_to_base
        points.append([p_map[0], p_map[1], p_map[2], det["confidence"]])
        
    return np.array(points) if points else np.empty((0, 4))

def analyze_tree(tid, eps=0.08, min_samples=3, max_radius=0.7, output_dir=ARTIFACTS_DIR):
    """Run DBSCAN diagnostic for a single tree and output figures and stats."""
    tx, ty = trees[tid]
    gt_val = gt_counts.get(tid, "Unknown")
    
    idx_a, fname_a, dist_a, idx_b, fname_b, dist_b = get_frame_selection(tid, offset_frames=5)
    
    pts_a = get_frame_3d_points(fname_a)
    pts_b = get_frame_3d_points(fname_b)
    
    # Filter to tree 0.7m radius
    if len(pts_a) > 0:
        dists_a = np.sqrt((pts_a[:, 0] - tx)**2 + (pts_a[:, 1] - ty)**2)
        pts_a_f = pts_a[dists_a < max_radius]
    else:
        pts_a_f = np.empty((0, 4))
        
    if len(pts_b) > 0:
        dists_b = np.sqrt((pts_b[:, 0] - tx)**2 + (pts_b[:, 1] - ty)**2)
        pts_b_f = pts_b[dists_b < max_radius]
    else:
        pts_b_f = np.empty((0, 4))
        
    combined_pts = np.vstack([pts_a_f, pts_b_f]) if (len(pts_a_f) > 0 or len(pts_b_f) > 0) else np.empty((0, 4))
    
    total_in = len(combined_pts)
    n_clusters = 0
    n_noise = 0
    largest_size = 0
    bounding_radius = 0.0
    a_in_largest = 0
    b_in_largest = 0
    
    if total_in > 0:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(combined_pts[:, :3])
        labels = db.labels_
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        largest_label = None
        for l in unique_labels:
            if l == -1:
                continue
            sz = list(labels).count(l)
            if sz > largest_size:
                largest_size = sz
                largest_label = l
                
        if largest_label is not None:
            l_pts = combined_pts[labels == largest_label]
            centroid = l_pts[:, :3].mean(axis=0)
            bounding_radius = float(np.linalg.norm(l_pts[:, :3] - centroid, axis=1).max())
            
            idx_in_combined = np.where(labels == largest_label)[0]
            a_in_largest = int(np.sum(idx_in_combined < len(pts_a_f)))
            b_in_largest = int(np.sum(idx_in_combined >= len(pts_a_f)))
            
        # Generate 3D Plot
        fig = plt.figure(figsize=(15, 6))
        
        # Subplot 1: Frame Comparison
        ax1 = fig.add_subplot(121, projection='3d')
        if len(pts_a_f) > 0:
            ax1.scatter(pts_a_f[:, 0], pts_a_f[:, 1], pts_a_f[:, 2], c='blue', marker='^', s=45, label=f'Frame A ({fname_a}, idx {idx_a})')
        if len(pts_b_f) > 0:
            ax1.scatter(pts_b_f[:, 0], pts_b_f[:, 1], pts_b_f[:, 2], c='red', marker='s', s=45, label=f'Frame B ({fname_b}, idx {idx_b})')
            
        # Vertical Trunk Reference Line
        z_min = combined_pts[:, 2].min() - 0.2 if total_in > 0 else 0
        z_max = combined_pts[:, 2].max() + 0.2 if total_in > 0 else 2.0
        ax1.plot([tx, tx], [ty, ty], [z_min, z_max], color='saddlebrown', linestyle='--', linewidth=3, label=f'Tree {tid} Trunk ({tx:.2f}, {ty:.2f})')
        
        ax1.set_xlabel('East [m]')
        ax1.set_ylabel('North [m]')
        ax1.set_zlabel('Up [m]')
        ax1.set_title(f'Tree {tid} (GT={gt_val}): Frame Overlap (A vs B)')
        ax1.legend(loc='upper right')
        
        # Subplot 2: Cluster IDs
        ax2 = fig.add_subplot(122, projection='3d')
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters + 1, 2)))
        
        noise_mask = labels == -1
        if np.any(noise_mask):
            ax2.scatter(combined_pts[noise_mask, 0], combined_pts[noise_mask, 1], combined_pts[noise_mask, 2], c='black', marker='x', s=25, label='Noise')
            
        c_idx = 0
        for l in sorted(unique_labels):
            if l == -1:
                continue
            c_mask = labels == l
            c_color = colors[c_idx % 20]
            c_idx += 1
            ax2.scatter(combined_pts[c_mask, 0], combined_pts[c_mask, 1], combined_pts[c_mask, 2], color=c_color, marker='o', s=50, label=f'Cluster {l}')
            
        ax2.plot([tx, tx], [ty, ty], [z_min, z_max], color='saddlebrown', linestyle='--', linewidth=3, label='Tree Trunk')
        
        ax2.set_xlabel('East [m]')
        ax2.set_ylabel('North [m]')
        ax2.set_zlabel('Up [m]')
        ax2.set_title(f'Tree {tid} (GT={gt_val}): DBSCAN Clusters (eps=0.08m)')
        if n_clusters < 10:
            ax2.legend(loc='upper right')
            
        plt.suptitle(f'Tree {tid} Diagnostic (GT: {gt_val} Büschel) | Frame A (idx {idx_a}, d={dist_a:.2f}m) vs Frame B (idx {idx_b}, d={dist_b:.2f}m)', fontsize=13, fontweight='bold')
        plot_path = os.path.join(output_dir, f"tree_{tid}_3d_analysis.jpg")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    stats = {
        "tid": tid,
        "gt": gt_val,
        "fname_a": fname_a,
        "idx_a": idx_a,
        "dist_a": dist_a,
        "fname_b": fname_b,
        "idx_b": idx_b,
        "dist_b": dist_b,
        "pts_a_f": len(pts_a_f),
        "pts_b_f": len(pts_b_f),
        "total_in": total_in,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "largest_size": largest_size,
        "bounding_radius": bounding_radius,
        "a_in_largest": a_in_largest,
        "b_in_largest": b_in_largest
    }
    return stats

def main():
    parser = argparse.ArgumentParser(description="DBSCAN 3D Diagnostic Tool")
    parser.add_argument("--trees", nargs="+", type=int, default=[27, 26, 20, 64, 39], help="List of Tree IDs to process")
    parser.add_argument("--eps", type=float, default=0.08, help="DBSCAN epsilon radius in meters (default: 0.08)")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples threshold (default: 3)")
    args = parser.parse_args()
    
    all_stats = []
    print("\n" + "="*80, flush=True)
    print(f"RUNNING DBSCAN 3D DIAGNOSTIC TOOL (eps={args.eps}m, min_samples={args.min_samples})", flush=True)
    print("="*80, flush=True)
    
    for tid in args.trees:
        if tid not in trees:
            print(f"Warning: Tree {tid} not found in tree map. Skipping.", flush=True)
            continue
        print(f"Processing Tree {tid}...", flush=True)
        stats = analyze_tree(tid, eps=args.eps, min_samples=args.min_samples)
        all_stats.append(stats)
        
    print("\n" + "="*110)
    print(f"{'Tree ID':<8} | {'GT':<5} | {'Frame A (idx, dist)':<25} | {'Frame B (idx, dist)':<25} | {'Total In':<8} | {'Clusters':<8} | {'Noise':<6} | {'Max Cluster':<11} | {'Radius (m)':<10}")
    print("="*110)
    for s in all_stats:
        str_a = f"{s['fname_a']} ({s['idx_a']}, {s['dist_a']:.2f}m)"
        str_b = f"{s['fname_b']} ({s['idx_b']}, {s['dist_b']:.2f}m)"
        print(f"{s['tid']:<8} | {s['gt']:<5} | {str_a:<25} | {str_b:<25} | {s['total_in']:<8} | {s['n_clusters']:<8} | {s['n_noise']:<6} | {s['largest_size']:<11} | {s['bounding_radius']:<10.3f}")
    print("="*110 + "\n")

if __name__ == "__main__":
    main()
