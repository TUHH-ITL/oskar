#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np

import config
import utils
from step5_multiview_fusion import process_multiview_fusion
from step6_bio_sanity import process_bio_sanity
from step7_tree_assignment import process_tree_assignment
from step8_thinning_decision import process_thinning_decision

def main():
    parser = argparse.ArgumentParser(description="Test frame decimation on Step 5-8 offline mapping pipeline.")
    parser.add_argument("--decimation", type=int, default=1, help="Decimation factor (default: 1)")
    args = parser.parse_args()

    decimation = args.decimation

    # 1. Load world_obs.npy
    obs_path = os.path.join(config.BACKPROJ_OUT_DIR, "world_obs.npy")
    if not os.path.exists(obs_path):
        print(f"Error: Step 4 output {obs_path} not found. Please run run_pipeline.py first!")
        return

    world_obs = np.load(obs_path)
    print(f"Loaded {len(world_obs)} total observations from {obs_path}")

    # Check if frame_idx column exists (column 5)
    if world_obs.shape[1] < 6:
        print("Error: world_obs.npy does not have the 6th column (frame_idx). Re-run run_pipeline.py first!")
        return

    # 2. Filter observations by frame_idx % decimation == 0
    frame_indices = world_obs[:, 5].astype(int)
    mask = (frame_indices % decimation == 0)
    filtered_obs = world_obs[mask]
    print(f"Decimation N={decimation}: Filtered down to {len(filtered_obs)} observations (from {len(world_obs)})")

    # 3. Run Step 5: Multiview Fusion
    landmarks = process_multiview_fusion(
        filtered_obs,
        eps=config.MULTIVIEW_FUSION_EPS,
        min_samples=config.MULTIVIEW_FUSION_MIN_SAMPLES,
    )
    print(f"Fused into {len(landmarks)} unique landmarks.")

    # 4. Run Step 6: Bio Sanity Priors
    landmarks_bio = process_bio_sanity(
        landmarks,
        config.BIO_SANITY_EXPECTED_FLOWERS_PER_CORYMB,
        config.BIO_SANITY_INTRA_FLOWER_SPACING_M,
        config.BIO_SANITY_MERGE_DISTANCE_M,
        config.BIO_SANITY_CORYMB_RADIUS_M
    )
    print(f"Bio Sanity landmarks: {len(landmarks_bio)}")

    # 5. Run Step 7: Tree Assignment
    per_tree_flowers = process_tree_assignment(
        landmarks_bio,
        config.TREE_MAP_FILE,
        config.TREE_ASSIGNMENT_MAX_RADIUS_M
    )

    # 6. Run Step 8: Thinning Decision
    plan = process_thinning_decision(
        per_tree_flowers,
        target=getattr(config, "AGRONOMIC_TARGET_FLOWERS", 20),
        agronomic_min=getattr(config, "AGRONOMIC_MIN_FLOWERS", 15),
        agronomic_max=getattr(config, "AGRONOMIC_MAX_FLOWERS", 25)
    )

    # Archive outputs to data_archive/decim_test_N{N}_2026-07-02/
    archive_dir = os.path.join(config.OUTPUT_DIR, "data_archive", f"decim_test_N{decimation}_2026-07-02")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Save the files
    np.save(os.path.join(archive_dir, "landmarks_bio.npy"), landmarks_bio)
    
    with open(os.path.join(archive_dir, "per_tree_flowers.json"), "w") as f:
        json.dump(per_tree_flowers, f, indent=4)
        
    with open(os.path.join(archive_dir, "thinning_plan.json"), "w") as f:
        json.dump(plan, f, indent=4)

    # Write summary stats
    total_trees = len(per_tree_flowers)
    scanned_trees = sum(1 for item in plan if item["total_count"] > 0)
    zero_trees = total_trees - scanned_trees
    
    summary = {
        "decimation": decimation,
        "total_observations": len(filtered_obs),
        "fused_landmarks": len(landmarks),
        "bio_landmarks": len(landmarks_bio),
        "trees_scanned": scanned_trees,
        "zero_flower_trees": zero_trees,
        "total_trees": total_trees
    }
    
    with open(os.path.join(archive_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\n==========================================")
    print(f"DECIMATION N={decimation} SUMMARY REPORT")
    print("==========================================")
    print(f"Total Observations:   {len(filtered_obs)}")
    print(f"Fused Landmarks:      {len(landmarks)}")
    print(f"Bio-prior Landmarks:  {len(landmarks_bio)}")
    print(f"Trees Scanned (>0):   {scanned_trees} / {total_trees}")
    print(f"Zero-flower Trees:    {zero_trees}")
    print(f"Results archived to:  {archive_dir}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
