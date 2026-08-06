#!/usr/bin/env python3
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import utils
from step7_tree_assignment import load_tree_map

def main():
    # Paths to archived outputs
    archive_dir = "/home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline/data_archive/run_full2566_2026-07-02"
    per_tree_path = os.path.join(archive_dir, "step7_tree_assignment", "per_tree_flowers.json")
    
    if not os.path.exists(per_tree_path):
        print(f"Error: Archived file not found at {per_tree_path}")
        return
        
    print(f"Loading archived tree assignment from {per_tree_path}...")
    with open(per_tree_path, "r") as f:
        per_tree_flowers = json.load(f)
        
    # Convert keys to int for correct sorting/alignment
    per_tree_flowers_int = {}
    for k, v in per_tree_flowers.items():
        try:
            per_tree_flowers_int[int(k)] = v
        except ValueError:
            per_tree_flowers_int[k] = v

    trees = load_tree_map(config.TREE_MAP_FILE)
    
    # Save the new version plots alongside the old ones with _v2 suffix
    map_v2_path = os.path.join(archive_dir, "step8_thinning_decision", "2d_cluster_map_v2.png")
    
    print(f"Regenerating 2D maps (Full + 40m parts) into {os.path.dirname(map_v2_path)}...")
    utils.save_2d_cluster_map(
        trees, 
        per_tree_flowers_int, 
        map_v2_path, 
        x_window_m=40.0
    )
    
    print("\nRegeneration of v2 plots successfully completed!")
    print(f"Full v2 Map:  {map_v2_path}")
    print(f"Parts generated in the same directory.")

if __name__ == "__main__":
    main()
