#!/usr/bin/env python3
"""Modular Step 3: Depth and mask fusion to compute 3D coordinates in camera frame.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

import os
import cv2
import numpy as np

import config
import utils

def process_depth_fusion(depth_maps, segmentation_results, calibration_path=None):
    """Fuses depth maps and flower masks, backprojecting centroids to 3D camera coordinates.
    
    Returns a list of frames, each containing a list of 3D detections.
    """
def depth_fusion_frame(depth_cv, flowers, K, kernel):
    """Fuses depth map and segmentations for a single frame, backprojecting flower centroids to 3D."""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    detections = []
    if depth_cv is None or depth_cv.size == 0:
        return detections

    for flower in flowers:
        mask_bool = flower["mask_bool"]
        mask_binary = mask_bool.astype(np.uint8)

        # Erode mask
        mask_eroded = cv2.morphologyEx(mask_binary, cv2.MORPH_ERODE, kernel)
        
        # Sample depth values under the eroded mask
        depth_samples = depth_cv[mask_eroded > 0]
        valid_depths = depth_samples[~np.isnan(depth_samples)]

        if len(valid_depths) < config.DEPTH_FUSION_MIN_VALID_PIXELS:
            continue

        # Compute median depth and std dev
        median_depth = float(np.median(valid_depths))
        depth_std = float(np.std(valid_depths))
        depth_std_ratio = depth_std / median_depth if median_depth > 0 else np.inf
        low_confidence = depth_std_ratio > config.DEPTH_FUSION_STD_THRESHOLD

        # Backproject centroid to 3D Left Camera coordinate system
        u = flower["centroid_u"]
        v = flower["centroid_v"]

        x_cam = (u - cx) / fx * median_depth
        y_cam = (v - cy) / fy * median_depth
        z_cam = median_depth

        detections.append({
            "instance_id": flower["instance_id"],
            "confidence": flower["confidence"],
            "low_confidence": low_confidence,
            "depth_std_ratio": depth_std_ratio,
            "x_cam": x_cam,
            "y_cam": y_cam,
            "z_cam": z_cam
        })
    return detections

def process_depth_fusion(depth_maps, segmentation_results, calibration_path=None):
    """Fuses depth maps and flower masks, backprojecting centroids to 3D camera coordinates.
    
    Returns a list of frames, each containing a list of 3D detections.
    """
    if calibration_path is None:
        calibration_path = config.LEFT_CALIB_FILE

    lc = utils.load_calibration_yaml(calibration_path)
    K = np.array(lc["cameraMatrix"], dtype=np.float64).reshape(3, 3)

    # Create erosion kernel
    erosion_radius = config.DEPTH_FUSION_EROSION_RADIUS
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_radius + 1, 2 * erosion_radius + 1)
    )

    results_3d = []
    total_frames = len(depth_maps)

    for idx, depth_cv in enumerate(depth_maps):
        print(f"[{idx+1}/{total_frames}] Fusing depth for frame {idx}...")
        
        frame_seg = segmentation_results[idx]
        flowers = frame_seg["flowers"]
        detections = depth_fusion_frame(depth_cv, flowers, K, kernel)

        results_3d.append({
            "frame_idx": idx,
            "timestamp": frame_seg["timestamp"],
            "detections": detections
        })

    return results_3d

def save_intermediate_detections_3d(results_3d, output_dir):
    """Save 3D camera-frame detections to disk."""
    utils.clear_dir(output_dir)
    print(f"Saving intermediate 3D detections to {output_dir}...")
    for frame in results_3d:
        frame_idx = frame["frame_idx"]
        save_path = os.path.join(output_dir, f"detections_3d_{frame_idx:06d}.npz")
        
        detections = frame["detections"]
        if not detections:
            np.savez_compressed(
                save_path,
                instance_ids=np.array([]),
                confidences=np.array([]),
                low_confidences=np.array([]),
                depth_std_ratios=np.array([]),
                positions_cam=np.array([]),
                timestamp=frame["timestamp"]
            )
            continue

        instance_ids = np.array([d["instance_id"] for d in detections])
        confidences = np.array([d["confidence"] for d in detections])
        low_confidences = np.array([d["low_confidence"] for d in detections])
        depth_std_ratios = np.array([d["depth_std_ratio"] for d in detections])
        positions_cam = np.array([[d["x_cam"], d["y_cam"], d["z_cam"]] for d in detections])

        np.savez_compressed(
            save_path,
            instance_ids=instance_ids,
            confidences=confidences,
            low_confidences=low_confidences,
            depth_std_ratios=depth_std_ratios,
            positions_cam=positions_cam,
            timestamp=frame["timestamp"]
        )
    print(f"All 3D detections saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 3: Depth and mask fusion to compute 3D coordinates in camera frame.")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to process (-1 for all, default: 10)")
    args = parser.parse_args()

    max_frames = args.max_frames

    depth_maps = []
    segmentation_results = []

    # Count how many frames actually exist or try to load up to max_frames
    idx = 0
    while True:
        if max_frames >= 0 and idx >= max_frames:
            break

        depth_path = os.path.join(config.DISPARITY_OUT_DIR, f"depth_{idx:06d}.npy")
        mask_path = os.path.join(config.SEGMENTATION_OUT_DIR, f"masks_{idx:06d}.npz")

        if not os.path.exists(depth_path) or not os.path.exists(mask_path):
            if max_frames >= 0:
                print(f"Error: Step 1 or 2 output for frame {idx} not found. Run previous steps first!")
                exit(1)
            else:
                # If running on 'all' (-1), stop when files run out
                break

        depth_maps.append(np.load(depth_path))

        data = np.load(mask_path)
        masks = data["masks"]
        scores = data["scores"]
        centroids = data["centroids"]
        timestamp = float(data["timestamp"])

        flowers = []
        if masks.size > 0:
            for inst_idx in range(len(scores)):
                flowers.append({
                    "instance_id": inst_idx,
                    "confidence": scores[inst_idx],
                    "centroid_u": centroids[inst_idx, 0],
                    "centroid_v": centroids[inst_idx, 1],
                    "mask_bool": masks[inst_idx]
                })

        segmentation_results.append({
            "frame_idx": idx,
            "timestamp": timestamp,
            "flowers": flowers
        })
        idx += 1

    print(f"Fusing depth for {len(depth_maps)} frames...")
    results_3d = process_depth_fusion(depth_maps, segmentation_results)
    save_intermediate_detections_3d(results_3d, config.DEPTH_FUSION_OUT_DIR)
