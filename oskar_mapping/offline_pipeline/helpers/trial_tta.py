#!/usr/bin/env python3
"""Trial script for Change 3: Test-Time Augmentation (TTA) with darkened pass.
This script does NOT modify any existing code, config, or files.
Run standalone for comparison only.
"""
import os, cv2, numpy as np, torch
import sys
sys.path.append('/home/workstation/ros2_ws/src/oskar/oskar_mapping/offline_pipeline')
import config, utils
from step1_segmentation_tiled import load_segmentation_model, numpy_nms

frame_name = 'tree_0209.png'
img_path = os.path.join(config.LEFT_IMAGES_DIR, frame_name)

print('Running Change 3 trial: TTA darkened pass (original conf=0.60, no CLAHE)...')

predictor = load_segmentation_model(config.SEGMENTATION_MODEL_PATH, config.SEG_CONFIDENCE_THRESHOLD, device='cuda')
rectification_maps, _ = utils.build_rectification_maps(config.LEFT_CALIB_FILE, config.RIGHT_CALIB_FILE)

left_cv = cv2.imread(img_path)
map_lx, map_ly = rectification_maps
left_cv = cv2.remap(left_cv, map_lx, map_ly, cv2.INTER_LINEAR)

H, W = left_cv.shape[:2]
tile_size = config.SEG_TILE_SIZE
stride = tile_size - config.SEG_TILE_OVERLAP

x_coords = sorted(list(set(range(0, W - tile_size + 1, stride)) | {W - tile_size}))
y_coords = sorted(list(set(range(0, H - tile_size + 1, stride)) | {H - tile_size}))

all_boxes = []
all_scores = []
all_tile_masks_info = []

def run_inference_on_tiles(image, brightness_factor=1.0, tag='normal'):
    if brightness_factor != 1.0:
        img_float = image.astype(np.float32) * brightness_factor
        image = np.clip(img_float, 0, 255).astype(np.uint8)

    count_before = len(all_boxes)
    for y_offset in y_coords:
        for x_offset in x_coords:
            tile = image[y_offset : y_offset + tile_size, x_offset : x_offset + tile_size]
            with torch.inference_mode():
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                    outputs = predictor(tile)
            instances = outputs['instances']
            if len(instances) == 0:
                continue

            tile_boxes = instances.pred_boxes.tensor.cpu().numpy()
            tile_scores = instances.scores.cpu().numpy()
            tile_masks = instances.pred_masks.cpu().numpy()

            for box, score, mask in zip(tile_boxes, tile_scores, tile_masks):
                if score < config.SEG_CONFIDENCE_THRESHOLD:
                    continue
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                if max(bw, bh) < config.SEG_MIN_BOX_SIZE:
                    continue
                y_idx, x_idx = np.where(mask)
                if len(x_idx) == 0:
                    continue
                local_u = float(np.mean(x_idx))
                local_v = float(np.mean(y_idx))
                x1 = box[0] + x_offset
                y1 = box[1] + y_offset
                x2 = box[2] + x_offset
                y2 = box[3] + y_offset
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(float(score))
                all_tile_masks_info.append((mask, x_offset, y_offset, local_u, local_v))
    print(f'  -> {len(all_boxes) - count_before} proposals from [{tag}] pass', flush=True)

# Pass 1: Normal image (original baseline)
print('Pass 1: Normal brightness...', flush=True)
run_inference_on_tiles(left_cv, brightness_factor=1.0, tag='normal')

# Pass 2: TTA - darkened by -20% (multiply by 0.80)
print('Pass 2: Darkened -20% (TTA)...', flush=True)
run_inference_on_tiles(left_cv, brightness_factor=0.80, tag='darkened-20%')

print(f'Total pooled proposals before NMS: {len(all_boxes)}', flush=True)

keep = numpy_nms(
    np.array(all_boxes, dtype=np.float32),
    np.array(all_scores, dtype=np.float32),
    config.SEG_NMS_IOU_THRESHOLD
)

trial_flowers = []
for instance_id, idx in enumerate(keep):
    mask, x_offset, y_offset, local_u, local_v = all_tile_masks_info[idx]
    score = all_scores[idx]
    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[y_offset : y_offset + tile_size, x_offset : x_offset + tile_size] = mask
    trial_flowers.append({
        'instance_id': instance_id,
        'confidence': float(score),
        'centroid_u': local_u + x_offset,
        'centroid_v': local_v + y_offset,
        'mask_bool': full_mask
    })

print(f'', flush=True)
print(f'=== CHANGE 3 TTA TRIAL RESULT ===', flush=True)
print(f'Original baseline (conf=0.60, no TTA):  362', flush=True)
print(f'Trial 1 (CLAHE + conf=0.40):            385', flush=True)
print(f'Trial 3 (TTA darkened, conf=0.60):      {len(trial_flowers)}', flush=True)

out_path = '/home/workstation/.gemini/antigravity-ide/brain/fc8795bb-cef3-4fe2-b749-cc8c1221f41e/tree_39_tta_trial_visual.jpg'
utils.save_debug_segmentation(img_path, trial_flowers, out_path, map_l=rectification_maps)
print(f'Saved TTA trial visual to: {out_path}', flush=True)
