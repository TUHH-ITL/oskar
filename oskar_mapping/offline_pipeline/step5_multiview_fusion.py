#!/usr/bin/env python3
"""Modular Step 5: DBSCAN clustering to merge duplicate flower observations into unique landmarks.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

import os
import numpy as np
from sklearn.cluster import DBSCAN

import config

import utils

def process_multiview_fusion(world_observations, eps=None, min_samples=3):
    """Clusters world observations using DBSCAN to merge multi-view detections.
    
    Returns a list of unique flower landmarks: [x, y, z, confidence, observation_count]
    """
    if eps is None:
        eps = config.MULTIVIEW_FUSION_EPS

    if len(world_observations) < min_samples:
        print(f"Warning: not enough observations ({len(world_observations)}) to run DBSCAN.")
        return np.array([], dtype=np.float32)

    positions = world_observations[:, :3]
    confidences = world_observations[:, 3]

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label == -1:  # Noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    landmarks = []
    for cluster_id, indices in clusters.items():
        if len(indices) <= 1:  # Discard singletons
            continue

        cluster_positions = positions[indices]
        cluster_confidences = confidences[indices]

        # Compute confidence-weighted centroid
        total_conf = cluster_confidences.sum()
        if total_conf > 0:
            weights = cluster_confidences / total_conf
            centroid = (cluster_positions * weights[:, np.newaxis]).sum(axis=0)
        else:
            centroid = cluster_positions.mean(axis=0)

        mean_conf = cluster_confidences.mean()

        landmarks.append([
            centroid[0],      # East
            centroid[1],      # North
            centroid[2],      # Up
            float(mean_conf),
            len(indices)      # Observation count
        ])

    return np.array(landmarks, dtype=np.float32)

if __name__ == "__main__":
    obs_path = os.path.join(config.BACKPROJ_OUT_DIR, "world_obs.npy")
    if not os.path.exists(obs_path):
        print(f"Error: Step 4 output {obs_path} not found. Run previous steps first!")
        exit(1)

    world_obs = np.load(obs_path)
    landmarks = process_multiview_fusion(world_obs)
    utils.clear_dir(config.MULTIVIEW_FUSION_OUT_DIR)
    save_path = os.path.join(config.MULTIVIEW_FUSION_OUT_DIR, "landmarks.npy")
    np.save(save_path, landmarks)
    print(f"Fused {len(world_obs)} observations into {len(landmarks)} landmarks. Saved to {save_path}")
