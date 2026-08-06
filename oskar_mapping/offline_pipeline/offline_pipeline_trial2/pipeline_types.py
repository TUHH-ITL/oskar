"""Typed Data Structures for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class FlowerDetection:
    """Represents a single flower detection throughout all pipeline stages."""
    instance_id: int
    confidence: float
    centroid_u: float
    centroid_v: float
    frame_name: str
    tree_id: Optional[int] = None
    mask_bool: Optional[np.ndarray] = None
    
    # Camera 3D coordinates (x, y, z) in meters
    cam_xyz: Optional[Tuple[float, float, float]] = None
    
    # World ENU 3D coordinates (E, N, U) in meters
    world_xyz: Optional[Tuple[float, float, float]] = None
    
    # Refined 3D coordinates after positional fusion
    refined_world_xyz: Optional[Tuple[float, float, float]] = None
    
    # Number of multi-view observations matched to this landmark
    obs_count: int = 1

@dataclass
class TreeSummary:
    """Summary of per-tree locked flower count and 3D landmarks."""
    tree_id: int
    gt_buschel: Optional[float]
    closest_frame: str
    raw_windowed_count: int
    predicted_buschel: float  # raw_windowed_count / 4.5
    column_window: Tuple[float, float]
    landmarks_3d: List[Tuple[float, float, float]] = field(default_factory=list)

@dataclass
class ThinningDecisionItem:
    """Per-tree agronomic thinning decision output."""
    tree_id: int
    gt_buschel: Optional[float]
    predicted_buschel: float
    needs_thinning: bool
    rough_removal_buschel: float
    priority_score: float
