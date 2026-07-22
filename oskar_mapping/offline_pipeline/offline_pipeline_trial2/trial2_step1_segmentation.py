"""Stage 1: Segmentation Stage for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

import os
import cv2
import gzip
import pickle
from typing import Dict, List

from pipeline_types import FlowerDetection

class SegmentationStage:
    """Stage 1: Performs live GPU Mask R-CNN segmentation or retrieves from fresh_cache."""
    
    def __init__(self, offline_dir: str, trial2_dir: str, conf_thresh: float = 0.6):
        self.offline_dir = offline_dir
        self.trial2_dir = trial2_dir
        self.conf_thresh = conf_thresh
        self.fresh_cache_dir = os.path.join(trial2_dir, "fresh_cache")
        os.makedirs(self.fresh_cache_dir, exist_ok=True)
        self.predictor = None

    def run(self, left_images_dir: str, frame_filenames: List[str]) -> Dict[str, List[FlowerDetection]]:
        """Runs segmentation across requested image frame files."""
        import sys
        if self.offline_dir not in sys.path:
            sys.path.append(self.offline_dir)
        import config
        from step1_segmentation_tiled import load_segmentation_model, segment_frame

        frame_detections_map = {}
        
        for idx, frame_fname in enumerate(sorted(frame_filenames)):
            base_name = os.path.splitext(frame_fname)[0]
            cache_pkl = os.path.join(self.fresh_cache_dir, f"{base_name}.pkl.gz")
            
            if os.path.exists(cache_pkl):
                with gzip.open(cache_pkl, "rb") as cf:
                    raw_list = pickle.load(cf)
                detections = []
                for item in raw_list:
                    detections.append(FlowerDetection(
                        instance_id=item["instance_id"],
                        confidence=item["confidence"],
                        centroid_u=item["centroid_u"],
                        centroid_v=item["centroid_v"],
                        frame_name=frame_fname
                    ))
                frame_detections_map[frame_fname] = detections
            else:
                if self.predictor is None:
                    model_path = getattr(config, "SEGMENTATION_MODEL_PATH")
                    self.predictor = load_segmentation_model(model_path, self.conf_thresh, device="cuda")
                
                img_path = os.path.join(left_images_dir, frame_fname)
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    continue
                flowers = segment_frame(img_cv, self.predictor, self.conf_thresh)
                
                detections = []
                raw_list = []
                for fl in flowers:
                    det = FlowerDetection(
                        instance_id=fl["instance_id"],
                        confidence=fl["confidence"],
                        centroid_u=fl["centroid_u"],
                        centroid_v=fl["centroid_v"],
                        frame_name=frame_fname
                    )
                    detections.append(det)
                    raw_list.append({
                        "instance_id": fl["instance_id"],
                        "confidence": fl["confidence"],
                        "centroid_u": fl["centroid_u"],
                        "centroid_v": fl["centroid_v"]
                    })
                    
                with gzip.open(cache_pkl, "wb") as cf:
                    pickle.dump(raw_list, cf)
                frame_detections_map[frame_fname] = detections

        return frame_detections_map
