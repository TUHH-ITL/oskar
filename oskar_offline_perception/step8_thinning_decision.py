#!/usr/bin/env python3
"""Modular Step 8: Calculate thinning plan based on per-tree flower counts.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

import os
import json

import config

import utils

def process_thinning_decision(per_tree_flowers, target=20, agronomic_min=15, agronomic_max=25):
    """Computes thinning plan from per-tree flower counts.
    
    Returns a list of dicts: [ { 'tree_id': id, 'needs_thinning': bool, ... } ]
    """
    plan = []
    for tree_id, counts in sorted(per_tree_flowers.items()):
        confirmed = counts["confirmed"]
        inferred = counts["inferred"]
        total = confirmed + inferred

        needs_thinning = bool(total > agronomic_max)
        rough_removal = int(max(0, total - target))
        priority = float((total - target) / target if target > 0 else 0.0)

        plan.append({
            "tree_id": tree_id,
            "confirmed_count": confirmed,
            "inferred_count": inferred,
            "total_count": total,
            "needs_thinning": needs_thinning,
            "rough_removal_count": rough_removal,
            "priority_score": max(0.0, priority)
        })

    return plan

if __name__ == "__main__":
    per_tree_path = os.path.join(config.TREE_ASSIGN_OUT_DIR, "per_tree_flowers.json")
    if not os.path.exists(per_tree_path):
        print(f"Error: Step 7 output {per_tree_path} not found. Run previous steps first!")
        exit(1)

    with open(per_tree_path, "r") as f:
        per_tree_flowers = json.load(f)
    
    # JSON converts dict keys to string, so convert them back to int
    per_tree_flowers_int = {int(k): v for k, v in per_tree_flowers.items()}

    plan = process_thinning_decision(
        per_tree_flowers_int,
        config.THINNING_TARGET_FLOWERS_PER_TREE,
        config.THINNING_AGRONOMIC_MIN,
        config.THINNING_AGRONOMIC_MAX
    )

    utils.clear_dir(config.THINNING_OUT_DIR)
    save_path = os.path.join(config.THINNING_OUT_DIR, "thinning_plan.json")
    with open(save_path, "w") as f:
        json.dump(plan, f, indent=4)
    
    # Save the thinning plot
    plot_path = os.path.join(config.THINNING_OUT_DIR, "thinning_plan_plot.png")
    utils.save_thinning_plot(plan, plot_path)
    print(f"Thinning plan generated successfully. Saved to {save_path}")
