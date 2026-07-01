#!/usr/bin/env python3
"""Modular Step 7: Assign flower landmarks to the nearest known tree.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

import os
import yaml
import numpy as np

import config

import utils

def load_tree_map(filepath):
    """Load tree map coordinate positions from YAML."""
    if not os.path.exists(filepath):
        print(f"Warning: tree map file {filepath} not found. Returning empty map.")
        return {}

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    if not data or "trees" not in data:
        print("Warning: no 'trees' key in tree map yaml.")
        return {}

    trees = {}
    for tree_id, pos in data["trees"].items():
        trees[int(tree_id)] = (float(pos["x"]), float(pos["y"]))
    return trees

def process_tree_assignment(landmarks_bio, tree_map_filepath=None, max_radius=None):
    """Assigns each landmark to the nearest tree coordinate within max_radius.
    
    Returns a dictionary of: tree_id -> { 'confirmed': count, 'inferred': count, 'landmarks': [] }
    """
    if tree_map_filepath is None:
        tree_map_filepath = config.TREE_MAP_FILE

    if max_radius is None:
        max_radius = config.TREE_ASSIGNMENT_MAX_RADIUS_M

    trees = load_tree_map(tree_map_filepath)
    if not trees:
        print("Error: no trees loaded for assignment.")
        return {}

    # Initialize dictionary
    per_tree_flowers = {
        tree_id: {"confirmed": 0, "inferred": 0, "landmarks": []}
        for tree_id in trees
    }

    # Format of landmarks_bio: [E, N, U, confidence, obs_count, is_inferred]
    for lm in landmarks_bio:
        lm_x, lm_y, lm_z = lm[0], lm[1], lm[2]
        is_inferred = bool(lm[5] > 0.5)

        closest_tree_id = None
        closest_distance = max_radius

        for tree_id, (tree_x, tree_y) in trees.items():
            dist = np.sqrt((lm_x - tree_x) ** 2 + (lm_y - tree_y) ** 2)
            if dist < closest_distance:
                closest_distance = dist
                closest_tree_id = tree_id

        if closest_tree_id is not None:
            per_tree_flowers[closest_tree_id]["landmarks"].append(lm.tolist())
            if is_inferred:
                per_tree_flowers[closest_tree_id]["inferred"] += 1
            else:
                per_tree_flowers[closest_tree_id]["confirmed"] += 1

    return per_tree_flowers

if __name__ == "__main__":
    landmarks_bio_path = os.path.join(config.BIO_SANITY_OUT_DIR, "landmarks_bio.npy")
    if not os.path.exists(landmarks_bio_path):
        print(f"Error: Step 6 output {landmarks_bio_path} not found. Run previous steps first!")
        exit(1)

    landmarks_bio = np.load(landmarks_bio_path)
    per_tree_flowers = process_tree_assignment(
        landmarks_bio,
        config.TREE_MAP_FILE,
        config.TREE_ASSIGNMENT_MAX_RADIUS_M
    )

    # Save output as a JSON file
    import json
    utils.clear_dir(config.TREE_ASSIGN_OUT_DIR)
    save_path = os.path.join(config.TREE_ASSIGN_OUT_DIR, "per_tree_flowers.json")
    with open(save_path, "w") as f:
        json.dump(per_tree_flowers, f, indent=4)
    print(f"Assigned landmarks to trees. Saved counts to {save_path}")
