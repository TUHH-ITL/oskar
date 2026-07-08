#!/usr/bin/env python3
import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import utils
from step7_tree_assignment import load_tree_map
from step8_thinning_decision import process_thinning_decision

def main():
    per_tree_path = os.path.join(config.TREE_ASSIGN_OUT_DIR, "per_tree_flowers.json")
    landmarks_bio_path = os.path.join(config.BIO_SANITY_OUT_DIR, "landmarks_bio.npy")
    
    if not os.path.exists(per_tree_path):
        print(f"Error: {per_tree_path} not found.")
        sys.exit(1)
        
    with open(per_tree_path, "r") as f:
        per_tree_flowers = json.load(f)
        
    # Convert keys to int
    per_tree_flowers_int = {}
    for k, v in per_tree_flowers.items():
        try:
            per_tree_flowers_int[int(k)] = v
        except ValueError:
            per_tree_flowers_int[k] = v

    trees = load_tree_map(config.TREE_MAP_FILE)
    map_path = os.path.join(config.THINNING_OUT_DIR, "2d_cluster_map.png")
    
    # 1. Regenerate rotated 2D map (filtered inside save_2d_cluster_map)
    utils.save_2d_cluster_map(trees, per_tree_flowers_int, map_path)
    
    # 2. Regenerate thinning plan plot (filtered inside save_thinning_plot)
    plan = process_thinning_decision(
        per_tree_flowers_int,
        config.THINNING_TARGET_FLOWERS_PER_TREE,
        config.THINNING_AGRONOMIC_MIN,
        config.THINNING_AGRONOMIC_MAX
    )
    plot_path = os.path.join(config.THINNING_OUT_DIR, "thinning_plan_plot.png")
    utils.save_thinning_plot(plan, plot_path)
    
    # 3. Update execution report with new stats
    report_path = os.path.join(config.THINNING_OUT_DIR, "execution_report.log")
    
    total_flowers = 0
    if os.path.exists(landmarks_bio_path):
        landmarks_bio = np.load(landmarks_bio_path)
        total_flowers = len(landmarks_bio)
    else:
        # Fallback to sum of tree counts
        total_flowers = sum(v["confirmed"] + v["inferred"] for v in per_tree_flowers_int.values())
        
    scanned_trees = sum(1 for item in plan if item["total_count"] > 0)
    
    # Read existing report to preserve timing details
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            lines = f.readlines()
        
        # Insert stats right after Processed Frames Count
        new_lines = []
        for line in lines:
            if "Processed Frames Count" in line:
                new_lines.append(line)
                new_lines.append(f"Total Flower Landmarks Detected:   {total_flowers}\n")
                new_lines.append(f"Total Trees Scanned:               {scanned_trees}\n")
                continue
            if "Total Flower Landmarks Detected" in line or "Total Trees Scanned" in line:
                continue
            new_lines.append(line)
            
        with open(report_path, "w") as f:
            f.writelines(new_lines)
            
    print(f"Successfully regenerated clean cluster map at {map_path}")
    print(f"Successfully regenerated clean thinning bar plot at {plot_path}")
    print(f"Updated execution report at {report_path}")

if __name__ == "__main__":
    main()
