"""Stage 4: Position Refinement Stage for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

import numpy as np
from typing import Dict, List

from pipeline_types import FlowerDetection, TreeSummary

class PositionRefinementStage:
    """Stage 4: Multi-view positional fusion across Frame A and Frame B of the SAME tree_id (r <= 1.5cm). NO count changes."""

    def __init__(self, refine_radius_m: float = 0.015):
        self.refine_radius_m = refine_radius_m  # 1.5 cm safe radius threshold

    def run(self, summaries: Dict[int, TreeSummary]) -> Dict[int, TreeSummary]:
        """Refines 3D positions using strict 1.5cm spatial threshold across SAME-TREE frames (Frame A & Frame B)."""
        for tid in sorted(list(summaries.keys())):
            summary = summaries[tid]
            win_a: List[FlowerDetection] = getattr(summary, "_windowed_a_dets", [])
            win_b: List[FlowerDetection] = getattr(summary, "_windowed_b_dets", [])

            if not win_a:
                continue

            frame_b_pts = [np.array(db.world_xyz) for db in win_b if db.world_xyz is not None]

            refined_landmarks = []
            for det in win_a:
                if det.world_xyz is None:
                    continue

                p_primary = np.array(det.world_xyz)
                matched_pts = [p_primary]

                # Match against Frame B observations of the SAME tree_id
                for p_b in frame_b_pts:
                    dist = np.linalg.norm(p_primary - p_b)
                    if dist <= self.refine_radius_m:
                        matched_pts.append(p_b)

                # Weighted average 3D position (NO point deletion, NO count alteration)
                p_refined = np.mean(np.array(matched_pts), axis=0)
                refined_pt = (float(p_refined[0]), float(p_refined[1]), float(p_refined[2]))
                det.refined_world_xyz = refined_pt
                det.obs_count = len(matched_pts)
                refined_landmarks.append(refined_pt)

            # HARD CONSTRAINT ENFORCED: len(summary.landmarks_3d) == N_i ALWAYS
            assert len(refined_landmarks) == summary.raw_windowed_count, \
                f"Tree {tid}: Landmark count changed ({len(refined_landmarks)} vs locked N_i={summary.raw_windowed_count})"
            summary.landmarks_3d = refined_landmarks

        return summaries
