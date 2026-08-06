"""Stage 5: Thinning Decision Stage for Offline Pipeline Trial 2.

All code resides strictly within offline_pipeline_trial2/.
"""

from typing import Dict, List
from pipeline_types import TreeSummary, ThinningDecisionItem

# UNVALIDATED PLACEHOLDER AGRONOMIC THRESHOLDS (AWAITING AGRONOMIC BÜSCHEL VALUES FROM MARKUS)
PLACEHOLDER_THINNING_TARGET_BUSCHEL = 20.0
PLACEHOLDER_THINNING_AGRONOMIC_MAX_BUSCHEL = 25.0

class ThinningDecisionStage:
    """Stage 5: Evaluates agronomic thinning decision on real locked count (in Büschel)."""

    def __init__(self, target_buschel: float = PLACEHOLDER_THINNING_TARGET_BUSCHEL, max_buschel: float = PLACEHOLDER_THINNING_AGRONOMIC_MAX_BUSCHEL):
        self.target_buschel = target_buschel
        self.max_buschel = max_buschel

    def run(self, summaries: Dict[int, TreeSummary]) -> List[ThinningDecisionItem]:
        """Computes per-tree thinning decisions using real total count."""
        decisions = []

        for tid in sorted(list(summaries.keys())):
            summary = summaries[tid]
            pred_count = summary.predicted_buschel

            needs_thinning = bool(pred_count > self.max_buschel)
            rough_removal = max(0.0, pred_count - self.target_buschel)
            priority = max(0.0, (pred_count - self.target_buschel) / self.target_buschel) if self.target_buschel > 0 else 0.0

            decisions.append(ThinningDecisionItem(
                tree_id=tid,
                gt_buschel=summary.gt_buschel,
                predicted_buschel=pred_count,
                needs_thinning=needs_thinning,
                rough_removal_buschel=rough_removal,
                priority_score=priority
            ))

        return decisions
