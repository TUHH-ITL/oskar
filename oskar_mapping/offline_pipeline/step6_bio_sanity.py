#!/usr/bin/env python3
"""Modular Step 6: Apply apple flower biology priors to infer missing corymb members.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

import os
import numpy as np
from sklearn.cluster import DBSCAN

import config

import utils

def merge_detections(landmarks, merge_distance):
    """Merge landmarks within merge_distance, keeping the highest confidence detection."""
    if len(landmarks) < 2:
        return landmarks
    
    positions = landmarks[:, :3]
    clustering = DBSCAN(eps=merge_distance, min_samples=1).fit(positions)
    labels = clustering.labels_
    
    merged = {}
    for idx, label in enumerate(labels):
        if label not in merged:
            merged[label] = []
        merged[label].append(landmarks[idx])
        
    result = []
    for cluster_lms in merged.values():
        if len(cluster_lms) == 1:
            result.append(cluster_lms[0])
        else:
            # Merge: keep the observation with the highest confidence (index 3)
            best = max(cluster_lms, key=lambda lm: lm[3])
            result.append(best)
            
    return np.array(result, dtype=np.float32)

def group_into_corymbs(landmarks, corymb_radius):
    """Group landmarks into spatial clusters (corymb candidates) using DBSCAN."""
    if len(landmarks) == 0:
        return []
        
    positions = landmarks[:, :3]
    clustering = DBSCAN(eps=corymb_radius, min_samples=1).fit(positions)
    labels = clustering.labels_
    
    groups = {}
    for idx, label in enumerate(labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(landmarks[idx])
        
    return list(groups.values())

def infer_corymb(group, expected_per_corymb, intra_spacing):
    """Infer missing flowers inside a corymb group using radial layout."""
    output = []
    
    # Keep original confirmed flowers
    for lm in group:
        # Format: [E, N, U, confidence, observation_count, is_inferred]
        # We append a 5th index for is_inferred flag (0.0 for original, 1.0 for inferred)
        output.append([lm[0], lm[1], lm[2], lm[3], lm[4], 0.0])
        
    confirmed_count = len(group)
    if confirmed_count == 0:
        return output
    elif confirmed_count == 1:
        to_infer = expected_per_corymb - 1
    elif confirmed_count < expected_per_corymb:
        to_infer = expected_per_corymb - confirmed_count
    else:
        return output
        
    # Compute group centroid
    positions = np.array([[lm[0], lm[1], lm[2]] for lm in group])
    centroid = positions.mean(axis=0)
    
    # Generate inferred positions around centroid
    for i in range(to_infer):
        angle = 2 * np.pi * i / to_infer
        offset = intra_spacing * 0.5  # Radial offset
        dx = offset * np.cos(angle)
        dy = offset * np.sin(angle)
        
        output.append([
            float(centroid[0] + dx),  # East
            float(centroid[1] + dy),  # North
            float(centroid[2]),       # Up
            0.3,                      # Low confidence for inferred
            0.0,                      # 0 observations for inferred
            1.0                       # is_inferred = True (1.0)
        ])
        
    return output

def process_bio_sanity(landmarks, expected_per_corymb=5, intra_spacing=0.03, merge_distance=0.02, corymb_radius=0.06):
    """Applies biology priors to landmarks list."""
    if len(landmarks) == 0:
        return np.array([], dtype=np.float32)
        
    # Step 1: Merge double detections
    merged_lms = merge_detections(landmarks, merge_distance)
    
    # Step 2: Group into corymb candidates
    corymb_groups = group_into_corymbs(merged_lms, corymb_radius)
    
    # Step 3: Infer missing members
    output_landmarks = []
    for group in corymb_groups:
        inferred_group = infer_corymb(group, expected_per_corymb, intra_spacing)
        output_landmarks.extend(inferred_group)
        
    return np.array(output_landmarks, dtype=np.float32)

if __name__ == "__main__":
    landmarks_path = os.path.join(config.MULTIVIEW_FUSION_OUT_DIR, "landmarks.npy")
    if not os.path.exists(landmarks_path):
        print(f"Error: Step 5 output {landmarks_path} not found. Run previous steps first!")
        exit(1)
        
    landmarks = np.load(landmarks_path)
    landmarks_bio = process_bio_sanity(
        landmarks,
        config.BIO_SANITY_EXPECTED_FLOWERS_PER_CORYMB,
        config.BIO_SANITY_INTRA_FLOWER_SPACING_M,
        config.BIO_SANITY_MERGE_DISTANCE_M,
        config.BIO_SANITY_CORYMB_RADIUS_M
    )
    utils.clear_dir(config.BIO_SANITY_OUT_DIR)
    save_path = os.path.join(config.BIO_SANITY_OUT_DIR, "landmarks_bio.npy")
    np.save(save_path, landmarks_bio)
    print(f"Bio sanity: {len(landmarks)} -> {len(landmarks_bio)} landmarks after inference. Saved to {save_path}")
