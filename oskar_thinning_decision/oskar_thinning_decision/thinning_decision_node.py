#!/usr/bin/env python3
"""Serve DecideThinning: which candidates stay on a tree. Computation only.

The rule lives in selection.py and knows nothing of ROS, so it also runs on the
offline map.
"""

import numpy as np
import rclpy
from rclpy.node import Node

from oskar_msgs.msg import ThinningDecision
from oskar_msgs.srv import DecideThinning
from oskar_thinning_decision.selection import min_spacing, select_keep


class ThinningDecisionNode(Node):
    def __init__(self):
        super().__init__("thinning_decision_node")
        self.create_service(DecideThinning, "decide_thinning", self._decide)
        self.get_logger().info("thinning decision ready")

    def _decide(self, request, response):
        candidates = request.candidates.candidates
        decision = ThinningDecision()
        decision.header = request.candidates.header
        decision.tree_id = request.candidates.tree_id
        decision.max_kept = request.max_kept

        ids = [c.id for c in candidates]
        positions = np.array([[c.pose.pose.position.x, c.pose.pose.position.y,
                               c.pose.pose.position.z] for c in candidates]).reshape(-1, 3)
        index = {candidate_id: i for i, candidate_id in enumerate(ids)}
        forced = [index[i] for i in request.keep_ids if i in index]

        keep = select_keep(positions, request.max_kept, forced)
        kept = set(keep)
        decision.keep_ids = [ids[i] for i in keep]
        decision.remove_ids = [ids[i] for i in range(len(ids)) if i not in kept]
        decision.min_spacing = min_spacing(positions, keep)
        decision.limit_met = len(keep) <= request.max_kept
        decision.message = (
            f"{len(ids)} candidates, keep {len(keep)}, remove {len(decision.remove_ids)}, "
            f"{len(forced)} kept by request, closest kept pair "
            f"{decision.min_spacing * 100:.1f} cm")
        if not decision.limit_met:
            decision.message += f"; {len(forced)} must stay, above the limit of {request.max_kept}"
        self.get_logger().info(f"tree {decision.tree_id}: {decision.message}")
        response.success = True
        response.decision = decision
        response.message = decision.message
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ThinningDecisionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
