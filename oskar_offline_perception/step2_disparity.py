#!/usr/bin/env python3
"""Modular Step 2: Stereo disparity estimation using Fast-FoundationStereo on flower ROIs.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

# Scrub ROS 2 statistics package module shadowing
import sys
sys.path = [p for p in sys.path if "statistics" not in p]

import os
import cv2
import numpy as np
import torch
from pathlib import Path

import config
import utils

# Fast-FoundationStereo and FoundationStereo both ship a top-level `core` package, so only
# the selected repo may go on sys.path.
_VARIANT = getattr(config, "DISPARITY_MODEL_VARIANT", "fast")
_repo_dir = str(config.FS_DIR if _VARIANT == "foundationstereo" else config.FFM_DIR)
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from core.utils.utils import InputPadder


class FastFoundationStereoModel:
    """Wrapper around Fast-FoundationStereo model."""

    def __init__(self, ffm_dir, model_path, valid_iters=8, max_disp=192):
        self.valid_iters = valid_iters
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        torch.backends.cudnn.benchmark = getattr(config, "DISP_CUDNN_BENCHMARK", True)
        torch.autograd.set_grad_enabled(False)
        self.model = torch.load(
            str(model_path),
            map_location="cpu",
            weights_only=False,
        )
        self.model.args.valid_iters = valid_iters
        self.model.args.max_disp = max_disp
        self.model = self.model.to(self.device).eval()

    def forward_batch(self, left_bgr_list, right_bgr_list):
        """BGR images batch forward. Returns list of disparity maps (H, W), float32."""
        if not left_bgr_list:
            return []

        B = len(left_bgr_list)
        H, W = left_bgr_list[0].shape[:2]

        left_rgb_list = [img[..., ::-1].copy() for img in left_bgr_list]
        right_rgb_list = [img[..., ::-1].copy() for img in right_bgr_list]

        img0 = (
            torch.as_tensor(np.stack(left_rgb_list))
            .to(self.device)
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        img1 = (
            torch.as_tensor(np.stack(right_rgb_list))
            .to(self.device)
            .float()
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda",
                enabled=(self.device.type == "cuda"),
                dtype=torch.float16,
            ):
                disp = self.model.forward(
                    img0,
                    img1,
                    iters=self.valid_iters,
                    test_mode=True,
                    optimize_build_volume="pytorch1",
                )

        disp = padder.unpad(disp.float())
        disp_np = (
            disp.data.cpu()
            .numpy()
            .clip(0, None)
            .astype(np.float32)
        )
        return [disp_np[i, 0] for i in range(B)]


class FoundationStereoModel:
    """Wrapper around the full FoundationStereo model, same interface as the Fast one.

    The checkpoint stores only weights, so the network is built from the cfg.yaml next to it.
    """

    def __init__(self, repo_dir, model_path, valid_iters=None, max_disp=None):
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = getattr(config, "DISP_CUDNN_BENCHMARK", True)
        torch.autograd.set_grad_enabled(False)

        cfg = OmegaConf.load(os.path.join(os.path.dirname(str(model_path)), "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        if max_disp is not None:
            cfg["max_disp"] = max_disp
        args = OmegaConf.create(cfg)
        self.valid_iters = int(valid_iters if valid_iters is not None
                               else args.get("valid_iters", 32))
        self.hierarchical = bool(getattr(config, "FS_HIERARCHICAL", False))

        model = FoundationStereo(args)
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        self.model = model.to(self.device).eval()

    def forward_batch(self, left_bgr_list, right_bgr_list):
        """BGR images batch forward. Returns list of disparity maps (H, W), float32."""
        if not left_bgr_list:
            return []

        B = len(left_bgr_list)
        img0 = (torch.as_tensor(np.stack([i[..., ::-1].copy() for i in left_bgr_list]))
                .to(self.device).float().permute(0, 3, 1, 2).contiguous())
        img1 = (torch.as_tensor(np.stack([i[..., ::-1].copy() for i in right_bgr_list]))
                .to(self.device).float().permute(0, 3, 1, 2).contiguous())

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                if self.hierarchical:
                    disp = self.model.run_hierachical(
                        img0, img1, iters=self.valid_iters, test_mode=True, small_ratio=0.5)
                else:
                    disp = self.model.forward(
                        img0, img1, iters=self.valid_iters, test_mode=True)

        disp_np = padder.unpad(disp.float()).data.cpu().numpy().clip(0, None).astype(np.float32)
        return [disp_np[i, 0] for i in range(B)]


def load_disparity_model():
    """Builds the disparity backend named by config.DISPARITY_MODEL_VARIANT."""
    if _VARIANT == "foundationstereo":
        return FoundationStereoModel(config.FS_DIR, config.FS_MODEL_PATH,
                                     config.DISP_VALID_ITERS, config.DISP_MAX_DISP)
    return FastFoundationStereoModel(config.FFM_DIR, config.DISPARITY_MODEL_PATH,
                                     config.DISP_VALID_ITERS, config.DISP_MAX_DISP)


def disparity_frame_full(left_rect, right_rect, model, fx, baseline, scale=None):
    """Dense depth for the whole frame in one forward pass.

    Tiling produced visible seams and whole tiles collapsing to a constant, because a tile
    with little texture has nothing to constrain it. Downscaling far enough that the frame
    fits in a single pass removes that entirely; scale 3 (2.7 Mpx) is the largest this GPU
    accepts and is also faster than tiling.

    max_disp is in pixels of the processed image, so it scales with `scale`.
    """
    if scale is None:
        scale = getattr(config, "DISP_FULL_FRAME_SCALE", 3)
    height, width = left_rect.shape[:2]
    w, h = int(width / scale), int(height / scale)

    previous_max_disp = getattr(model.model.args, "max_disp", None)
    model.model.args.max_disp = max(32, int(round(config.DISP_MAX_DISP / scale / 32.0)) * 32)

    left_small = cv2.resize(left_rect, (w, h), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right_rect, (w, h), interpolation=cv2.INTER_AREA)
    disp = model.forward_batch([np.ascontiguousarray(left_small)],
                               [np.ascontiguousarray(right_small)])[0]

    depth_small = np.full(disp.shape, np.nan, dtype=np.float32)
    valid = disp > 0.5
    depth_small[valid] = (fx / scale) * baseline / disp[valid]

    if previous_max_disp is not None:
        model.model.args.max_disp = previous_max_disp
    return cv2.resize(depth_small, (width, height), interpolation=cv2.INTER_NEAREST)


def disparity_frame(left_rect, right_rect, flowers, model, fx, baseline):
    """Computes depth map for a single frame pair by running ROI batched disparity."""
    height, width = left_rect.shape[:2]
    disparity = np.zeros((height, width), dtype=np.float32)

    left_crops = []
    right_crops = []
    crop_meta = []

    for flower in flowers:
        mask_bool = flower["mask_bool"]

        y_indices, x_indices = np.where(mask_bool)
        if len(x_indices) == 0:
            continue

        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()

        h_target = config.DISP_ROI_HEIGHT
        w_target = config.DISP_ROI_WIDTH

        if h_target > height or w_target > width:
            continue

        # Vertical centering
        y_center = (ymin + ymax) // 2
        ymin_padded = y_center - h_target // 2
        ymax_padded = ymin_padded + h_target

        if ymin_padded < 0:
            ymin_padded = 0
            ymax_padded = h_target
        elif ymax_padded > height:
            ymax_padded = height
            ymin_padded = height - h_target

        # Horizontal alignment: left search expansion by max_disp
        L = xmin - config.DISP_MAX_DISP - config.DISP_PADDING_X
        R = xmax + config.DISP_PADDING_X
        
        if (R - L) <= w_target:
            x_center = (L + R) // 2
            xmin_expanded = x_center - w_target // 2
            xmax_expanded = xmin_expanded + w_target
        else:
            xmax_expanded = R
            xmin_expanded = xmax_expanded - w_target

        if xmin_expanded < 0:
            xmin_expanded = 0
            xmax_expanded = w_target
        elif xmax_expanded > width:
            xmax_expanded = width
            xmin_expanded = width - w_target

        left_tile = np.ascontiguousarray(left_rect[ymin_padded:ymax_padded, xmin_expanded:xmax_expanded])
        right_tile = np.ascontiguousarray(right_rect[ymin_padded:ymax_padded, xmin_expanded:xmax_expanded])

        left_crops.append(left_tile)
        right_crops.append(right_tile)
        crop_meta.append((ymin_padded, ymax_padded, xmin_expanded, xmax_expanded, mask_bool))

    # Deduplicate overlapping ROI tiles (same pixel region) to prevent redundant GPU forward passes
    unique_left_crops = []
    unique_right_crops = []
    unique_meta = []
    tile_map = []  # maps original crop index to unique tile index

    for l_tile, r_tile, meta in zip(left_crops, right_crops, crop_meta):
        ymin_p, ymax_p, xmin_e, xmax_e, mask_b = meta
        matched_idx = -1
        for u_idx, (u_ymin, u_ymax, u_xmin, u_xmax, _) in enumerate(unique_meta):
            if abs(ymin_p - u_ymin) <= 32 and abs(xmin_e - u_xmin) <= 32:
                matched_idx = u_idx
                break
        if matched_idx >= 0:
            tile_map.append(matched_idx)
        else:
            tile_map.append(len(unique_meta))
            unique_meta.append(meta)
            unique_left_crops.append(l_tile)
            unique_right_crops.append(r_tile)

    # Batched inference on unique tiles
    batch_size = config.DISP_BATCH_SIZE
    computed_disps = []
    for i in range(0, len(unique_left_crops), batch_size):
        left_batch = unique_left_crops[i : i + batch_size]
        right_batch = unique_right_crops[i : i + batch_size]
        tile_disps = model.forward_batch(left_batch, right_batch)
        computed_disps.extend(tile_disps)

    # Apply disparity to masks using tile_map lookup
    for orig_idx, meta in enumerate(crop_meta):
        ymin_p, ymax_p, xmin_e, xmax_e, mask_b = meta
        u_idx = tile_map[orig_idx]
        tile_disp = computed_disps[u_idx]
        crop_mask = mask_b[ymin_p:ymax_p, xmin_e:xmax_e]
        if tile_disp.shape == crop_mask.shape:
            disparity[ymin_p:ymax_p, xmin_e:xmax_e] = np.where(
                crop_mask,
                tile_disp,
                disparity[ymin_p:ymax_p, xmin_e:xmax_e]
            )

    # Depth math
    depth = np.full_like(disparity, np.nan)
    valid = disparity > 0.0
    depth[valid] = (fx * baseline) / disparity[valid]
    return depth


def process_disparity(image_pairs_paths, segmentation_results, model=None, rectification_maps=None):
    """Processes Left/Right image pairs, computing disparity maps for the flower ROIs.
    
    Returns a list of depth maps of shape (H, W).
    """
    if model is None:
        model = load_disparity_model()

    if rectification_maps is None:
        rectification_maps = utils.build_rectification_maps(
            config.LEFT_CALIB_FILE,
            config.RIGHT_CALIB_FILE
        )

    (map_lx, map_ly), (map_rx, map_ry) = rectification_maps

    # Load calibration parameters for depth calculation
    lc = utils.load_calibration_yaml(config.LEFT_CALIB_FILE)
    rc = utils.load_calibration_yaml(config.RIGHT_CALIB_FILE)
    
    P_l = np.array(lc["projectionMatrix"], dtype=np.float64)
    P_r = np.array(rc["projectionMatrix"], dtype=np.float64)
    
    fx = P_l[0, 0]
    p3 = P_r[0, 3]
    p0 = P_r[0, 0]
    baseline = (-p3 / p0) if (p0 != 0 and p3 != 0) else config.DISP_BASELINE_M

    depth_maps = []
    total_imgs = len(image_pairs_paths)

    for idx, (left_path, right_path) in enumerate(image_pairs_paths):
        print(f"[{idx+1}/{total_imgs}] Estimating disparity for {os.path.basename(left_path)}...")
        
        left_cv = cv2.imread(left_path)
        right_cv = cv2.imread(right_path)
        if left_cv is None or right_cv is None:
            print(f"Warning: could not read image pair: {left_path} / {right_path}")
            depth_maps.append(np.array([]))
            continue

        left_cv = utils.normalize_calibration_orientation(left_cv, lc)
        right_cv = utils.normalize_calibration_orientation(right_cv, rc)

        # Rectify Left/Right images
        left_cv = cv2.remap(left_cv, map_lx, map_ly, cv2.INTER_LINEAR)
        right_cv = cv2.remap(right_cv, map_rx, map_ry, cv2.INTER_LINEAR)

        # Get segmentations for this frame
        frame_seg = segmentation_results[idx]
        flowers = frame_seg["flowers"]

        depth = disparity_frame(left_cv, right_cv, flowers, model, fx, baseline)
        depth_maps.append(depth)

    return depth_maps

def save_intermediate_depths(depths, output_dir):
    """Save depth maps to disk."""
    utils.clear_dir(output_dir)
    print(f"Saving intermediate depths and visualizations to {output_dir}...")
    for idx, depth in enumerate(depths):
        save_path = os.path.join(output_dir, f"depth_{idx:06d}.npy")
        np.save(save_path, depth)
        
        # Save colorized disparity image
        viz_path = os.path.join(output_dir, f"visual_{idx:06d}.jpg")
        utils.save_debug_disparity(depth, viz_path)
    print(f"All depth maps and visualizations saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 2: Stereo disparity estimation using Fast-FoundationStereo on flower ROIs.")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to process (-1 for all, default: 10)")
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames >= 0 else None

    # Resolve images
    if not os.path.exists(config.LEFT_IMAGES_DIR) or not os.path.exists(config.RIGHT_IMAGES_DIR):
        print("Error: images directory not found.")
        exit(1)

    left_files = sorted([
        os.path.join(config.LEFT_IMAGES_DIR, f)
        for f in os.listdir(config.LEFT_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    right_files = sorted([
        os.path.join(config.RIGHT_IMAGES_DIR, f)
        for f in os.listdir(config.RIGHT_IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if max_frames is not None:
        left_files_test = left_files[:max_frames]
        right_files_test = right_files[:max_frames]
    else:
        left_files_test = left_files
        right_files_test = right_files

    print(f"Estimating disparity for {len(left_files_test)} frames...")
    image_pairs = list(zip(left_files_test, right_files_test))

    # Reconstruct segmentation results from pre-saved masks
    segmentation_results = []
    for idx in range(len(left_files_test)):
        mask_path = os.path.join(config.SEGMENTATION_OUT_DIR, f"masks_{idx:06d}.npz")
        if not os.path.exists(mask_path):
            print(f"Error: Step 1 output {mask_path} not found. Run step1_segmentation.py first!")
            exit(1)
        
        data = np.load(mask_path)
        masks = data["masks"]
        scores = data["scores"]
        
        flowers = []
        if masks.size > 0:
            for inst_idx in range(len(scores)):
                flowers.append({
                    "mask_bool": masks[inst_idx],
                    "confidence": scores[inst_idx]
                })

        segmentation_results.append({
            "frame_idx": idx,
            "flowers": flowers
        })

    depths = process_disparity(image_pairs, segmentation_results)
    save_intermediate_depths(depths, config.DISPARITY_OUT_DIR)
