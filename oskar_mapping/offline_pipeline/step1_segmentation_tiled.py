#!/usr/bin/env python3
"""Modular Step 1: Apple flower instance segmentation using Mask R-CNN.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import config
import utils

def load_segmentation_model(model_path, confidence_threshold, device="cuda"):
    """Load Detectron2 model."""
    model_path = Path(model_path)
    config_path = model_path.parent / "detectron2_config.yaml"
    
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = device
    
    print(f"Loading segmentation model onto {device}...")
    return DefaultPredictor(cfg)

def numpy_nms(boxes, scores, iou_threshold=0.5):
    """Applies Non-Maximum Suppression to bounding boxes using NumPy."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def segment_frame(left_cv, predictor, confidence_threshold=None):
    """Runs segmentation on a single rectified left image frame, supporting tiled inference."""
    if confidence_threshold is None:
        confidence_threshold = config.SEG_CONFIDENCE_THRESHOLD

    if not config.SEG_TILED_INFERENCE:
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                outputs = predictor(left_cv)
        instances = outputs["instances"]
        masks = instances.pred_masks.cpu().numpy()
        scores = instances.scores.cpu().numpy()

        frame_flowers = []
        for instance_id, (mask, score) in enumerate(zip(masks, scores)):
            if score < confidence_threshold:
                continue
            y_coords, x_coords = np.where(mask)
            if len(x_coords) == 0:
                continue
            centroid_u = float(np.mean(x_coords))
            centroid_v = float(np.mean(y_coords))
            frame_flowers.append({
                "instance_id": instance_id,
                "confidence": float(score),
                "centroid_u": centroid_u,
                "centroid_v": centroid_v,
                "mask_bool": mask
            })
        return frame_flowers

    # Tiled Inference Mode
    H, W = left_cv.shape[:2]
    tile_size = config.SEG_TILE_SIZE
    stride = tile_size - config.SEG_TILE_OVERLAP

    x_coords = []
    y_coords = []
    
    x = 0
    while x < W:
        x_coords.append(x)
        if x + tile_size >= W:
            if x != W - tile_size and W >= tile_size:
                x_coords.append(W - tile_size)
            break
        x += stride
        
    y = 0
    while y < H:
        y_coords.append(y)
        if y + tile_size >= H:
            if y != H - tile_size and H >= tile_size:
                y_coords.append(H - tile_size)
            break
        y += stride

    x_coords = sorted(list(set(x_coords)))
    y_coords = sorted(list(set(y_coords)))

    all_boxes = []
    all_scores = []
    all_tile_masks_info = []

    for y_offset in y_coords:
        for x_offset in x_coords:
            tile = left_cv[y_offset : y_offset + tile_size, x_offset : x_offset + tile_size]
            with torch.inference_mode():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
                    outputs = predictor(tile)
            instances = outputs["instances"]
            if len(instances) == 0:
                continue

            tile_boxes = instances.pred_boxes.tensor.cpu().numpy()
            tile_scores = instances.scores.cpu().numpy()
            tile_masks = instances.pred_masks.cpu().numpy()

            for box, score, mask in zip(tile_boxes, tile_scores, tile_masks):
                if score < confidence_threshold:
                    continue

                # Filter by box size to ignore tiny background clutter (e.g. background rows)
                w = box[2] - box[0]
                h = box[3] - box[1]
                if max(w, h) < config.SEG_MIN_BOX_SIZE:
                    continue

                # Calculate local centroid on 640x640 tile (very fast!)
                y_indices, x_indices = np.where(mask)
                if len(x_indices) == 0:
                    continue
                local_u = float(np.mean(x_indices))
                local_v = float(np.mean(y_indices))

                x1 = box[0] + x_offset
                y1 = box[1] + y_offset
                x2 = box[2] + x_offset
                y2 = box[3] + y_offset

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_tile_masks_info.append((mask, x_offset, y_offset, local_u, local_v))

    if len(all_boxes) == 0:
        return []

    all_boxes = np.array(all_boxes, dtype=np.float32)
    all_scores = np.array(all_scores, dtype=np.float32)

    keep = numpy_nms(all_boxes, all_scores, config.SEG_NMS_IOU_THRESHOLD)

    frame_flowers = []
    for instance_id, idx in enumerate(keep):
        mask, x_offset, y_offset, local_u, local_v = all_tile_masks_info[idx]
        score = all_scores[idx]

        full_mask = np.zeros((H, W), dtype=bool)
        full_mask[y_offset : y_offset + tile_size, x_offset : x_offset + tile_size] = mask

        centroid_u = local_u + x_offset
        centroid_v = local_v + y_offset

        frame_flowers.append({
            "instance_id": instance_id,
            "confidence": float(score),
            "centroid_u": centroid_u,
            "centroid_v": centroid_v,
            "mask_bool": full_mask
        })

    return frame_flowers

def process_segmentation(left_image_paths, predictor=None, rectification_maps=None, all_left_files=None):
    """Processes Left images, running Mask R-CNN to detect flowers.
    
    Returns a list of frames, where each frame contains a list of flower dicts.
    """
    if predictor is None:
        predictor = load_segmentation_model(
            config.SEGMENTATION_MODEL_PATH,
            config.SEG_CONFIDENCE_THRESHOLD,
            device="cuda"
        )

    if rectification_maps is None:
        rectification_maps, _ = utils.build_rectification_maps(
            config.LEFT_CALIB_FILE,
            config.RIGHT_CALIB_FILE
        )

    results = []
    total_imgs = len(left_image_paths)
    map_lx, map_ly = rectification_maps

    for idx, img_path in enumerate(left_image_paths):
        abs_idx = idx
        if all_left_files is not None:
            try:
                abs_idx = all_left_files.index(img_path)
            except ValueError:
                pass
                
        print(f"[{idx+1}/{total_imgs}] Segmenting {os.path.basename(img_path)} (Absolute Frame Index: {abs_idx})...")
        left_cv = cv2.imread(img_path)
        if left_cv is None:
            print(f"Warning: could not read {img_path}")
            results.append({"frame_idx": abs_idx, "flowers": []})
            continue

        # Rectify image
        left_cv = cv2.remap(left_cv, map_lx, map_ly, cv2.INTER_LINEAR)
        frame_flowers = segment_frame(left_cv, predictor)

        timestamp = utils.parse_frame_timestamp(img_path)

        results.append({
            "frame_idx": abs_idx,
            "image_path": img_path,
            "timestamp": timestamp,
            "flowers": frame_flowers
        })

    return results

def save_intermediate_segmentation(results, output_dir):
    """Save masks and centroids to disk for debugging/individual step execution."""
    utils.clear_dir(output_dir)
    print(f"Saving intermediate segmentation outputs and visualizations to {output_dir}...")
    rectification_maps, _ = utils.build_rectification_maps(
        config.LEFT_CALIB_FILE,
        config.RIGHT_CALIB_FILE
    )
    for frame in results:
        frame_idx = frame["frame_idx"]
        save_path = os.path.join(output_dir, f"masks_{frame_idx:06d}.npz")
        
        flowers = frame["flowers"]
        if not flowers:
            np.savez_compressed(
                save_path,
                masks=np.array([]),
                scores=np.array([]),
                centroids=np.array([]),
                image_path=frame["image_path"],
                timestamp=frame["timestamp"]
            )
            continue
        
        masks_array = np.stack([f["mask_bool"] for f in flowers], axis=0) # (N, H, W)
        scores_array = np.array([f["confidence"] for f in flowers])
        centroids_array = np.array([[f["centroid_u"], f["centroid_v"]] for f in flowers])

        if config.SEG_SAVE_NPZ:
            np.savez_compressed(
                save_path,
                masks=masks_array,
                scores=scores_array,
                centroids=centroids_array,
                image_path=frame["image_path"],
                timestamp=frame["timestamp"]
            )
        
        # Save debug visual image
        if config.SEG_SAVE_VISUALIZATIONS:
            viz_path = os.path.join(output_dir, f"visual_{frame_idx:06d}.jpg")
            utils.save_debug_segmentation(frame["image_path"], flowers, viz_path, map_l=rectification_maps)
    print(f"All intermediate masks and visualizations saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 1: Apple flower instance segmentation using Mask R-CNN.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process (overrides start/end)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (0-based, default: 0)")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (inclusive, default: None)")
    args = parser.parse_args()

    # Resolve images
    if not os.path.exists(config.LEFT_IMAGES_DIR):
        print(f"Error: left images dir {config.LEFT_IMAGES_DIR} does not exist.")
        exit(1)

    left_files = sorted([
        os.path.join(config.LEFT_IMAGES_DIR, f)
        for f in os.listdir(config.LEFT_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if args.max_frames is not None and args.max_frames >= 0:
        left_files_test = left_files[:args.max_frames]
    else:
        start_idx = args.start_frame
        end_idx = args.end_frame
        if end_idx is not None:
            left_files_test = left_files[start_idx:end_idx+1]
        else:
            left_files_test = left_files[start_idx:]

    print(f"Found {len(left_files)} images. Slicing range [{args.start_frame} to {args.end_frame if args.end_frame is not None else len(left_files)-1}]. Processing {len(left_files_test)} frames...")

    results = process_segmentation(left_files_test, all_left_files=left_files)
    save_intermediate_segmentation(results, config.SEGMENTATION_OUT_DIR)
