#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

import config
import utils
from step7_tree_assignment import process_tree_assignment, count_overlapping_landmarks, count_orphan_landmarks
from step8_thinning_decision import process_thinning_decision

def main():
    # 1. Load the archived landmarks_bio.npy from the full 2566-frame run
    archive_dir = os.path.join(config.OUTPUT_DIR, "data_archive", "run_full2566_2026-07-02")
    landmarks_bio_path = os.path.join(archive_dir, "landmarks_bio.npy")
    
    if not os.path.exists(landmarks_bio_path):
        # Fallback to active output folder if the archive doesn't exist yet
        landmarks_bio_path = os.path.join(config.BIO_SANITY_OUT_DIR, "landmarks_bio.npy")
        
    if not os.path.exists(landmarks_bio_path):
        print(f"Error: Could not find landmarks_bio.npy at {landmarks_bio_path}")
        return
 
    landmarks_bio = np.load(landmarks_bio_path)
    total_landmarks = len(landmarks_bio)
    print(f"Loaded {total_landmarks} landmarks from {landmarks_bio_path}")

    radii = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
    results = []

    print("| Radius (m) | Overlap Count | Overlap % | Orphan Count | Orphan % | Trees Scanned (>0) | Zero-Flower Trees | Total Assigned Flowers |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in radii:
        overlap_cnt = count_overlapping_landmarks(landmarks_bio, config.TREE_MAP_FILE, r)
        orphan_cnt = count_orphan_landmarks(landmarks_bio, config.TREE_MAP_FILE, r)
        
        overlap_pct = (overlap_cnt / total_landmarks) * 100
        orphan_pct = (orphan_cnt / total_landmarks) * 100
        
        # Run step 7 tree assignment
        per_tree = process_tree_assignment(landmarks_bio, config.TREE_MAP_FILE, r)
        
        # Run step 8 thinning decision
        plan = process_thinning_decision(
            per_tree,
            target=getattr(config, "AGRONOMIC_TARGET_FLOWERS", 20),
            agronomic_min=getattr(config, "AGRONOMIC_MIN_FLOWERS", 15),
            agronomic_max=getattr(config, "AGRONOMIC_MAX_FLOWERS", 25)
        )
        
        total_trees = len(per_tree)
        scanned_trees = sum(1 for item in plan if item["total_count"] > 0)
        zero_trees = total_trees - scanned_trees
        
        # Sum total assigned flowers
        total_assigned = sum(item["total_count"] for item in plan)

        results.append({
            "radius_m": r,
            "overlap_count": overlap_cnt,
            "overlap_pct": float(overlap_pct),
            "orphan_count": orphan_cnt,
            "orphan_pct": float(orphan_pct),
            "trees_scanned": scanned_trees,
            "zero_flower_trees": zero_trees,
            "total_assigned_flowers": total_assigned,
            "total_trees": total_trees
        })

        print(f"| {r:.2f} | {overlap_cnt} | {overlap_pct:.2f}% | {orphan_cnt} | {orphan_pct:.2f}% | {scanned_trees} / {total_trees} | {zero_trees} | {total_assigned} |")

    # 3. Save sweep results to data_archive/radius_sweep_2026-07-02/radius_sweep_results.json
    sweep_archive_dir = os.path.join(config.OUTPUT_DIR, "data_archive", "radius_sweep_2026-07-02")
    os.makedirs(sweep_archive_dir, exist_ok=True)
    
    save_path = os.path.join(sweep_archive_dir, "radius_sweep_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved sweep results to {save_path}\n")

if __name__ == "__main__":
    main()
