import os
import cv2
import numpy as np
import yaml

def load_calibration_yaml(filepath):
    """Load calibration file."""
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data

def build_rectification_maps(left_calib_path, right_calib_path):
    """Build left and right rectification maps using camera calibration."""
    lc = load_calibration_yaml(left_calib_path)
    rc = load_calibration_yaml(right_calib_path)
    
    K_l = np.array(lc["cameraMatrix"], dtype=np.float64)
    D_l = np.array(lc["distCoeffs"], dtype=np.float64)
    R_l = np.array(lc["rotation"], dtype=np.float64)
    P_l = np.array(lc["projectionMatrix"], dtype=np.float64)[:, :3]
    
    K_r = np.array(rc["cameraMatrix"], dtype=np.float64)
    D_r = np.array(rc["distCoeffs"], dtype=np.float64)
    R_r = np.array(rc["rotation"], dtype=np.float64)
    P_r = np.array(rc["projectionMatrix"], dtype=np.float64)[:, :3]
    
    width, height = lc["imageSize"]
    size = (width, height)
    
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, size, cv2.CV_32F)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, size, cv2.CV_32F)
    return (map_lx, map_ly), (map_rx, map_ry)

def parse_frame_timestamp(path):
    """Parse '003396_1713171581-898.jpg' -> timestamp float."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    _, ts = name.split("_", 1)
    sec_str, ms_str = ts.split("-")
    return float(sec_str) + float(ms_str) / 1000.0

def clear_dir(dir_path):
    """Deletes all files in a directory, creating it if it does not exist."""
    import shutil
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def save_debug_segmentation(image_path, flowers, output_path, map_l=None):
    """Saves a debug image with transparent segmentation mask overlays and centroids."""
    img = cv2.imread(image_path)
    if img is None:
        return
    if map_l is not None:
        img = cv2.remap(img, map_l[0], map_l[1], cv2.INTER_LINEAR)

    overlay = img.copy()
    for f in flowers:
        mask = f["mask_bool"]
        # Draw transparent green mask
        overlay[mask] = [0, 255, 0]
        # Draw centroid dot and ID
        cx, cy = int(f["centroid_u"]), int(f["centroid_v"])
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, f"ID {f['instance_id']}", (cx + 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Combine original and overlay
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.imwrite(output_path, img)

def save_debug_disparity(depth_map, output_path):
    """Saves a colormapped jet image representing depth/disparity values."""
    if depth_map is None or depth_map.size == 0:
        return
    # Convert depth back to a pseudo-disparity for pretty visualization, handling NaNs
    valid = ~np.isnan(depth_map)
    if not np.any(valid):
        # Save a black image if no valid depth
        cv2.imwrite(output_path, np.zeros_like(depth_map, dtype=np.uint8))
        return

    disp_pseudo = np.zeros_like(depth_map)
    disp_pseudo[valid] = 1.0 / depth_map[valid]
    
    # Scale to 0-255
    min_val, max_val = disp_pseudo[valid].min(), disp_pseudo[valid].max()
    if max_val > min_val:
        scaled = ((disp_pseudo - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        scaled = np.zeros_like(disp_pseudo, dtype=np.uint8)

    # Colorize valid pixels, keep background black
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colored[~valid] = [0, 0, 0]
    cv2.imwrite(output_path, colored)

def save_thinning_plot(plan, output_path):
    """Saves a bar plot visualizing the thinning plan per tree (only showing scanned trees)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    active_plan = [item for item in plan if item["total_count"] > 0]
    if not active_plan:
        return

    tree_ids = [item["tree_id"] for item in active_plan]
    totals = [item["total_count"] for item in active_plan]
    needs_thin = [item["needs_thinning"] for item in active_plan]

    if not tree_ids:
        return

    plt.figure(figsize=(10, 5))
    colors = ['#ff4d4d' if t else '#4da6ff' for t in needs_thin]
    plt.bar(tree_ids, totals, color=colors, edgecolor='black', alpha=0.85)

    # Target line
    import config
    plt.axhline(y=config.THINNING_TARGET_FLOWERS_PER_TREE, color='#00cc44', linestyle='--', linewidth=2, label='Target (20)')
    plt.axhline(y=config.THINNING_AGRONOMIC_MAX, color='#e68a00', linestyle=':', linewidth=1.5, label='Max (25)')

    plt.title("Flower Counts & Thinning Decisions per Tree", fontsize=14, fontweight='bold')
    plt.xlabel("Tree ID", fontsize=12)
    plt.ylabel("Total Flower Count (Confirmed + Inferred)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_2d_cluster_map(trees, per_tree_flowers, output_path):
    """Saves a detailed 2D farm map zoomed tightly in Y to distinguish flowers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np
    import os

    # 1. Gather active trees and rotate
    tree_ids = sorted(list(trees.keys()))
    active_tree_ids = []
    for tid in tree_ids:
        tree_data = per_tree_flowers.get(str(tid)) or per_tree_flowers.get(int(tid)) or {}
        if tree_data.get("confirmed", 0) + tree_data.get("inferred", 0) > 0:
            active_tree_ids.append(tid)
    if not active_tree_ids:
        active_tree_ids = tree_ids

    # Rotate row to align horizontally
    if len(active_tree_ids) >= 2:
        pt_first = np.array(trees[active_tree_ids[0]])
        pt_last = np.array(trees[active_tree_ids[-1]])
        dx, dy = pt_last - pt_first
        theta = np.arctan2(dy, dx)
    else:
        theta = 0.0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    def rotate_pt(x, y):
        if x is None or y is None:
            return None, None
        pts = np.vstack([x, y]).T
        pts_rot = pts @ R.T
        return pts_rot[:, 0], pts_rot[:, 1]

    trees_rot = {}
    for tid in active_tree_ids:
        tx, ty = trees[tid]
        rx, ry = rotate_pt(tx, ty)
        trees_rot[tid] = (rx[0], ry[0])
    tx_vals = [trees_rot[tid][0] for tid in active_tree_ids]
    ty_vals = [trees_rot[tid][1] for tid in active_tree_ids]

    # Load trajectory
    import config
    import json
    # Try loading metadata to crop trajectory
    meta_path = os.path.join(config.OUTPUT_DIR, "metadata.json")
    ts_start, ts_end = None, None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                ts_start = meta.get("ts_start")
                ts_end = meta.get("ts_end")
        except Exception:
            pass

    trajectory_x, trajectory_y = None, None
    if os.path.exists(config.POSE_TRAJECTORY_FILE):
        try:
            traj_data = np.load(config.POSE_TRAJECTORY_FILE)
            t_arr = traj_data["t"]
            tx_raw = traj_data["E"]
            ty_raw = traj_data["N"]
            
            # Crop raw trajectory to only include processed frame time range
            if ts_start is not None and ts_end is not None:
                # Add 0.5s padding to trajectory to ensure start/end are covered
                mask = (t_arr >= ts_start - 0.5) & (t_arr <= ts_end + 0.5)
                if np.any(mask):
                    tx_raw = tx_raw[mask]
                    ty_raw = ty_raw[mask]
                    
            trajectory_x, trajectory_y = rotate_pt(tx_raw, ty_raw)
        except Exception:
            pass

    max_radius = getattr(config, "TREE_ASSIGNMENT_MAX_RADIUS_M", 0.8)

    # Setup figure for a single clean map plot
    fig, ax = plt.subplots(figsize=(15, 7))

    # Track Y coordinates of trees, landmarks, and trajectory for dynamic Y scaling
    all_y = list(ty_vals)
    if trajectory_y is not None and len(trajectory_y) > 0:
        all_y.extend(list(trajectory_y))

    # Plot Robot Path
    if trajectory_x is not None and trajectory_y is not None and len(trajectory_x) > 0:
        ax.plot(trajectory_x, trajectory_y, color='#7F8C8D', linestyle='--', linewidth=1.5, alpha=0.6, label='Robot Path', zorder=1)
        ax.scatter(trajectory_x[0], trajectory_y[0], color='#27AE60', s=100, marker='o', edgecolors='black', label='Start Point', zorder=6)
        ax.scatter(trajectory_x[-1], trajectory_y[-1], color='#E67E22', s=100, marker='X', edgecolors='black', label='End Point', zorder=6)

    # Colors for flower landmarks
    cmap = plt.colormaps.get_cmap('tab10')
    colors = {tid: cmap(i % 10) for i, tid in enumerate(active_tree_ids)}

    for tree_id in active_tree_ids:
        tx, ty = trees_rot[tree_id]
        color = colors[tree_id]
        tree_data = per_tree_flowers.get(str(tree_id)) or per_tree_flowers.get(int(tree_id)) or {}
        
        # Draw the 0.8m canopy assignment boundary as a light shaded ellipse
        ellipse = Ellipse((tx, ty), width=max_radius*2, height=max_radius*2, facecolor=color, alpha=0.08, edgecolor=color, linestyle=':', linewidth=1.0, zorder=2)
        ax.add_patch(ellipse)

        # Plot Tree Location: prominent triangle with tree label
        ax.scatter(tx, ty, color='#27AE60', s=180, marker='^', edgecolors='black', linewidth=1.2, zorder=5,
                   label='Tree' if tree_id == active_tree_ids[0] else "")
        ax.text(tx, ty + 0.18, f"T{tree_id}", fontsize=10, ha='center', color='#2C3E50', fontweight='bold', zorder=6)

        # Plot assigned clusters (rotated)
        landmarks = tree_data.get("landmarks", [])
        if landmarks:
            lms = np.array(landmarks)
            lx_rot, ly_rot = rotate_pt(lms[:, 0], lms[:, 1])
            # Plot flower points slightly larger and with high contrast
            ax.scatter(lx_rot, ly_rot, s=25, alpha=0.7, color=color, edgecolors='black', linewidths=0.3, zorder=4,
                       label='Flower Cluster' if tree_id == active_tree_ids[0] else "")
            # Draw lines connecting flowers to the tree center
            for lxi, lyi in zip(lx_rot, ly_rot):
                ax.plot([tx, lxi], [ty, lyi], color=color, alpha=0.15, linewidth=0.8, zorder=3)
            all_y.extend(list(ly_rot))

    ax.set_title("Global Orchard Row Mapping & Flower Clusters Distribution", fontsize=15, fontweight='bold', color='#2C3E50', pad=15)
    ax.set_xlabel("Distance along Orchard Row [meters]", fontsize=12, fontweight='semibold', color='#34495E')
    ax_map_y_label = "Lateral Offset from Row [meters]\n(Stretched vertically to distinguish details)"
    ax.set_ylabel(ax_map_y_label, fontsize=12, fontweight='semibold', color='#34495E')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Set X limits with 8 meters padding
    ax.set_xlim(min(tx_vals) - 8.0, max(tx_vals) + 8.0)
    
    # Set Y limits dynamically zoomed in on the canopy range
    min_y = min(all_y)
    max_y = max(all_y)
    padding_y = 0.2
    ax.set_ylim(min_y - padding_y, max_y + padding_y)
    
    # CRITICAL: Do NOT set aspect equal! Stretches the Y-axis vertically to clearly separate flower details.
    ax.set_aspect('auto')
    
    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    # 2. Table Summary Plot (Saved as a separate image)
    fig_table, ax_table = plt.subplots(figsize=(6, 8))
    ax_table.axis('off')
    
    cell_text = []
    total_confirmed = 0
    total_inferred = 0
    
    for tid in active_tree_ids:
        tree_data = per_tree_flowers.get(str(tid)) or per_tree_flowers.get(int(tid)) or {}
        conf = tree_data.get("confirmed", 0)
        inf = tree_data.get("inferred", 0)
        tot = conf + inf
        cell_text.append([f"Tree {tid}", str(conf), str(inf), str(tot)])
        total_confirmed += conf
        total_inferred += inf
        
    cell_text.append(["Total", str(total_confirmed), str(total_inferred), str(total_confirmed + total_inferred)])
    
    col_labels = ["Tree ID", "Confirmed", "Inferred", "Total Flowers"]
    
    table = ax_table.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Bold the headers and total row, and color them nicely
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#404040') # Dark gray header
        elif row == len(cell_text):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0') # Light gray total row
            
    ax_table.set_title("Flower Counts Summary Table", fontsize=14, fontweight='bold', pad=15)
    
    # Save the table in the same output directory
    table_output_path = os.path.join(os.path.dirname(output_path), "flower_summary_table.png")
    plt.tight_layout()
    plt.savefig(table_output_path, dpi=200)
    plt.close()
