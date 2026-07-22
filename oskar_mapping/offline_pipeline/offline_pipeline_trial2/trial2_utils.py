"""Adapted Visualization Utilities for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def save_thinning_plan_plot(decision_items, target_buschel, max_buschel, output_path):
    """Plot 1: Bar chart of flower counts (in Büschel) and thinning decisions per tree."""
    active_items = [item for item in decision_items if item.predicted_buschel > 0 or (item.gt_buschel is not None and item.gt_buschel > 0)]
    if not active_items:
        active_items = decision_items

    tree_ids = [item.tree_id for item in active_items]
    counts = [item.predicted_buschel for item in active_items]
    needs_thin = [item.needs_thinning for item in active_items]

    plt.figure(figsize=(12, 5))
    colors = ['#ff4d4d' if t else '#4da6ff' for t in needs_thin]
    plt.bar(tree_ids, counts, color=colors, edgecolor='black', alpha=0.85)

    plt.axhline(y=target_buschel, color='#00cc44', linestyle='--', linewidth=2, label=f'Placeholder Target ({target_buschel:.0f} Büschel)')
    plt.axhline(y=max_buschel, color='#e68a00', linestyle=':', linewidth=1.5, label=f'Placeholder Max ({max_buschel:.0f} Büschel)')

    plt.title("Per-Tree Flower Counts & Thinning Decisions (2D Pixel-Column Windowed)", fontsize=13, fontweight='bold')
    plt.xlabel("Tree ID", fontsize=11)
    plt.ylabel("Estimated Count [Büschel] (Raw Detections ÷ 4.5)", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_flower_summary_table(decision_items, output_path):
    """Plot 2: Clean summary table adapted for 2D Windowed Pipeline Schema."""
    active_items = [item for item in decision_items if item.predicted_buschel > 0 or (item.gt_buschel is not None and item.gt_buschel > 0)]
    if not active_items:
        active_items = decision_items

    num_rows = len(active_items) + 2
    fig_height = max(6.0, num_rows * 0.28)
    fig_table, ax_table = plt.subplots(figsize=(7, fig_height))
    ax_table.axis('off')

    cell_text = []
    total_pred = 0.0

    for item in active_items:
        gt_str = f"{item.gt_buschel:.0f}" if item.gt_buschel is not None and not np.isnan(item.gt_buschel) else "Befruchter"
        thin_str = "YES" if item.needs_thinning else "No"
        rem_str = f"{item.rough_removal_buschel:.1f}" if item.rough_removal_buschel > 0 else "0.0"
        cell_text.append([
            f"Tree {item.tree_id}",
            gt_str,
            f"{item.predicted_buschel:.1f}",
            thin_str,
            rem_str
        ])
        total_pred += item.predicted_buschel

    cell_text.append(["Total / Mean", "-", f"{total_pred:.1f}", "-", "-"])
    col_labels = ["Tree ID", "Ground Truth", "Predicted (Büschel)", "Needs Thinning?", "Removal Target"]

    table = ax_table.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.3)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')
        elif row == len(cell_text):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#EAECEE')

    ax_table.set_title("Trial 2 Pipeline Summary Table", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def save_2d_cluster_map(trees, tree_summaries, pose_traj_path, output_path):
    """Plot 3: 2D Farm Map reading count-anchored 3D landmarks with dynamic data-derived X-limits and non-overlapping tree labels."""
    active_tree_ids = [s.tree_id for s in tree_summaries if len(s.landmarks_3d) > 0 or (s.gt_buschel is not None and not np.isnan(s.gt_buschel))]
    if not active_tree_ids:
        active_tree_ids = sorted(list(trees.keys()))

    pt_first = np.array(trees[active_tree_ids[0]])
    pt_last = np.array(trees[active_tree_ids[-1]])
    dx, dy = pt_last - pt_first
    theta = np.arctan2(dy, dx)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

    def rotate_pt(x, y):
        pts = np.vstack([x, y]).T
        pts_rot = pts @ R.T
        return pts_rot[:, 0], pts_rot[:, 1]

    trees_rot = {tid: rotate_pt(trees[tid][0], trees[tid][1]) for tid in active_tree_ids}
    tx_vals = [trees_rot[tid][0][0] for tid in active_tree_ids]
    ty_vals = [trees_rot[tid][1][0] for tid in active_tree_ids]

    # Explicit data-derived horizontal bounds with 2.0m padding
    min_x = min(tx_vals) - 2.0
    max_x = max(tx_vals) + 2.0

    traj_x, traj_y = None, None
    if os.path.exists(pose_traj_path):
        try:
            td = np.load(pose_traj_path)
            raw_tx, raw_ty = rotate_pt(td["E"], td["N"])
            # Crop trajectory to active tree span to prevent axis stretching
            mask = (raw_tx >= min_x - 1.0) & (raw_tx <= max_x + 1.0)
            if np.any(mask):
                traj_x = raw_tx[mask]
                traj_y = raw_ty[mask]
        except Exception:
            pass

    cmap = plt.colormaps.get_cmap('tab10')
    colors = {tid: cmap(i % 10) for i, tid in enumerate(active_tree_ids)}

    row_span_m = max_x - min_x
    fig_width = max(15.0, row_span_m * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    if traj_x is not None and traj_y is not None:
        ax.plot(traj_x, traj_y, color='#7F8C8D', linestyle='--', linewidth=1.5, alpha=0.6, label='Robot Trajectory')

    summary_dict = {s.tree_id: s for s in tree_summaries}

    # Label spacing logic: draw label for every 5th tree or key trees to prevent overlap
    for tid in active_tree_ids:
        tx, ty = trees_rot[tid][0][0], trees_rot[tid][1][0]
        color = colors[tid]
        
        ellipse = Ellipse((tx, ty), width=1.4, height=1.4, facecolor=color, alpha=0.08, edgecolor=color, linestyle=':', linewidth=1.0)
        ax.add_patch(ellipse)

        ax.scatter(tx, ty, color='#27AE60', s=90, marker='^', edgecolors='black', label='Tree' if tid == active_tree_ids[0] else "")
        
        # Staggered label drawing (every 5th tree or end trees) to avoid label clutter
        if tid % 5 == 0 or tid == active_tree_ids[0] or tid == active_tree_ids[-1]:
            ax.text(tx, ty + 0.25, f"T{tid}", fontsize=8.5, ha='center', color='#2C3E50', fontweight='bold', rotation=30)

        summary = summary_dict.get(tid)
        if summary and summary.landmarks_3d:
            lms = np.array(summary.landmarks_3d)
            lx_rot, ly_rot = rotate_pt(lms[:, 0], lms[:, 1])
            ax.scatter(lx_rot, ly_rot, s=18, alpha=0.75, color=color, edgecolors='black', linewidths=0.3, label='Count-Anchored 3D Flower' if tid == active_tree_ids[0] else "")
            for lxi, lyi in zip(lx_rot, ly_rot):
                ax.plot([tx, lxi], [ty, lyi], color=color, alpha=0.15, linewidth=0.8)

    ax.set_xlabel("Distance along Orchard Row [meters]", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Lateral Offset [meters]", fontsize=11, fontweight='semibold')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title("Count-Anchored 3D Orchard Flower Map (Trial 2 Pipeline)", fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
