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

def parse_tree_stamp_from_rec(tree_num):
    """Dynamically lookup exact timestamp from SAMSON3 .rec file using direct seek."""
    import struct
    try:
        # Resolve config inside function to avoid circular import at module load
        import config
        rec_path = os.path.join(os.path.dirname(config.LEFT_IMAGES_DIR), "2024-04-15_10-59-41_A27_Bluete_SAMSON3_1713171581.rec")
        if os.path.exists(rec_path):
            frame_idx = 113 + tree_num
            with open(rec_path, "rb") as f:
                f.seek(12 + frame_idx * 24551456)
                headerdata = f.read(32)
                if len(headerdata) == 32:
                    _, _, secs, nsecs = struct.unpack("<QQQQ", headerdata)
                    return float(secs) + float(nsecs) * 1e-9
    except Exception:
        pass
    # Fallback to linear formula (5 FPS starting at first frame epoch)
    return 1713171581.898 + (113 + tree_num) * 0.200

def parse_frame_timestamp(path):
    """Parse '003396_1713171581-898.jpg' or 'tree_0006.png' -> timestamp float."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    if name.startswith("tree_"):
        tree_num = int(name.split("_")[1])
        return parse_tree_stamp_from_rec(tree_num)
    else:
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

def save_2d_cluster_map(trees, per_tree_flowers, output_path, x_window_m=None):
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

    # Track Y coordinates of trees, landmarks, and trajectory for dynamic Y scaling
    all_y = list(ty_vals)
    if trajectory_y is not None and len(trajectory_y) > 0:
        all_y.extend(list(trajectory_y))

    # Pre-gather all flower landmarks to calculate accurate Y bounds
    for tree_id in active_tree_ids:
        tree_data = per_tree_flowers.get(str(tree_id)) or per_tree_flowers.get(int(tree_id)) or {}
        landmarks = tree_data.get("landmarks", [])
        if landmarks:
            lms = np.array(landmarks)
            lx_rot, ly_rot = rotate_pt(lms[:, 0], lms[:, 1])
            all_y.extend(list(ly_rot))

    min_y = min(all_y)
    max_y = max(all_y)
    padding_y = 0.2

    # Scale tree marker size and label spacing dynamically
    num_active = len(active_tree_ids)
    if num_active > 150:
        marker_size = 35
        label_interval = 20
    elif num_active > 50:
        marker_size = 70
        label_interval = 10
    elif num_active > 20:
        marker_size = 110
        label_interval = 5
    else:
        marker_size = 180
        label_interval = 1

    # Colors for flower landmarks
    cmap = plt.colormaps.get_cmap('tab10')
    colors = {tid: cmap(i % 10) for i, tid in enumerate(active_tree_ids)}

    # Helper function to render the map on a given Axes
    def draw_on_ax(ax, x_lim=None):
        # Plot Robot Path
        if trajectory_x is not None and trajectory_y is not None and len(trajectory_x) > 0:
            ax.plot(trajectory_x, trajectory_y, color='#7F8C8D', linestyle='--', linewidth=1.5, alpha=0.6, label='Robot Path', zorder=1)
            ax.scatter(trajectory_x[0], trajectory_y[0], color='#27AE60', s=100, marker='o', edgecolors='black', label='Start Point', zorder=6)
            ax.scatter(trajectory_x[-1], trajectory_y[-1], color='#E67E22', s=100, marker='X', edgecolors='black', label='End Point', zorder=6)

        # Plot trees and landmarks
        for tree_id in active_tree_ids:
            tx, ty = trees_rot[tree_id]
            if x_lim is not None:
                if tx < x_lim[0] - 2.0 or tx > x_lim[1] + 2.0:
                    continue

            color = colors[tree_id]
            tree_data = per_tree_flowers.get(str(tree_id)) or per_tree_flowers.get(int(tree_id)) or {}
            
            # Draw the 0.8m canopy assignment boundary as a light shaded ellipse
            ellipse = Ellipse((tx, ty), width=max_radius*2, height=max_radius*2, facecolor=color, alpha=0.08, edgecolor=color, linestyle=':', linewidth=1.0, zorder=2)
            ax.add_patch(ellipse)

            # Plot Tree Location
            ax.scatter(tx, ty, color='#27AE60', s=marker_size, marker='^', edgecolors='black', linewidth=1.0, zorder=5,
                       label='Tree' if tree_id == active_tree_ids[0] else "")
            
            # Label tree
            if x_lim is not None or (tree_id % label_interval == 0):
                ax.text(tx, ty + 0.18, f"T{tree_id}", fontsize=9, ha='center', color='#2C3E50', fontweight='bold', zorder=6)

            # Plot assigned clusters (rotated)
            landmarks = tree_data.get("landmarks", [])
            if landmarks:
                lms = np.array(landmarks)
                lx_rot, ly_rot = rotate_pt(lms[:, 0], lms[:, 1])
                ax.scatter(lx_rot, ly_rot, s=25, alpha=0.7, color=color, edgecolors='black', linewidths=0.3, zorder=4,
                           label='Flower Cluster' if tree_id == active_tree_ids[0] else "")
                for lxi, lyi in zip(lx_rot, ly_rot):
                    ax.plot([tx, lxi], [ty, lyi], color=color, alpha=0.15, linewidth=0.8, zorder=3)

        ax.set_xlabel("Distance along Orchard Row [meters]", fontsize=12, fontweight='semibold', color='#34495E')
        ax_map_y_label = "Lateral Offset from Row [meters]\n(Stretched vertically to distinguish details)"
        ax.set_ylabel(ax_map_y_label, fontsize=12, fontweight='semibold', color='#34495E')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(min(tx_vals) - 8.0, max(tx_vals) + 8.0)
            
        ax.set_ylim(min_y - padding_y, max_y + padding_y)
        ax.set_aspect('auto')
        
        # Clean legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left', framealpha=0.9)

    # Plot 1: Save full-row map with dynamic width
    row_span_m = max(tx_vals) - min(tx_vals)
    fig_width = max(15.0, row_span_m * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    draw_on_ax(ax)
    ax.set_title("Global Orchard Row Mapping & Flower Clusters Distribution", fontsize=15, fontweight='bold', color='#2C3E50', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    # Plot 2: Save paginated maps if x_window_m is provided
    if x_window_m is not None and x_window_m > 0:
        start_x = min(tx_vals) - 4.0
        end_x = max(tx_vals) + 4.0
        x_current = start_x
        part_idx = 1
        
        base, ext = os.path.splitext(output_path)
        while x_current < end_x:
            w_start = x_current
            w_end = x_current + x_window_m
            
            # Verify if there are actually any trees or trajectory points in this range
            has_data = False
            for tid in active_tree_ids:
                if w_start <= trees_rot[tid][0] <= w_end:
                    has_data = True
                    break
            if not has_data and trajectory_x is not None:
                if np.any((trajectory_x >= w_start) & (trajectory_x <= w_end)):
                    has_data = True
                    
            if has_data:
                fig_slice, ax_slice = plt.subplots(figsize=(15, 7))
                draw_on_ax(ax_slice, x_lim=(w_start, w_end))
                ax_slice.set_title(f"Orchard Row Mapping - Part {part_idx} ({w_start:.0f}m to {w_end:.0f}m)", fontsize=15, fontweight='bold', color='#2C3E50', pad=15)
                part_path = f"{base}_part{part_idx}{ext}"
                plt.tight_layout()
                plt.savefig(part_path, dpi=200)
                plt.close()
                part_idx += 1
            
            x_current += x_window_m

    # 2. Table Summary Plot (Saved as a separate image, sizing is dynamic based on number of trees)
    num_rows = len(active_tree_ids) + 2
    fig_height = max(6.0, num_rows * 0.26)
    fig_table, ax_table = plt.subplots(figsize=(6, fig_height))
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
    table.scale(1.0, 1.3)
    
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
