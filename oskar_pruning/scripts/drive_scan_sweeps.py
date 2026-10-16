#!/usr/bin/env python3
"""Publish a series of scan requests, waiting for each sweep to end.

Used to drive a capture run over several trees and standoffs without a person
at the keyboard. Sweeps are "tree,standoff,z_min,z_max,columns" arguments; with
none given it runs the set the training capture used.

    python3 drive_scan_sweeps.py 0,0.95,1.0,3.2,1 1,1.05,1.0,3.2,1
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from oskar_msgs.msg import ScanRequest

# Below 0.8 m the arm cannot reach the top of the band: a 0.6 m sweep lost 157
# waypoints and stopped at 2.78 m instead of 3.40 m.
DEFAULT_SWEEPS = [
    (0, 0.8, 0.9, 3.4, 2),
    (1, 0.9, 0.9, 3.4, 2),
    (0, 1.1, 0.9, 3.4, 1),
    (1, 1.2, 0.9, 3.4, 1),
    (0, 1.0, 0.9, 3.4, 1),
    (1, 0.8, 0.9, 3.4, 1),
]
SWEEP_TIMEOUT_S = 2400


class SweepDriver(Node):
    def __init__(self, sweeps):
        super().__init__("sweep_driver")
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(ScanRequest, "/pruning/scan_request", qos)
        self.create_subscription(String, "/pruning/scan_event", self._event, qos)
        self.sweeps = list(sweeps)
        self.index = -1
        self.finished = False
        self.deadline = 0.0
        self.create_timer(2.0, self._tick)

    def _event(self, message):
        if message.data.split()[0] in ("scan_finished", "scan_failed"):
            print(f"  {message.data}", flush=True)
            self.finished = True

    def _tick(self):
        now = time.time()
        if self.index >= 0 and not self.finished and now < self.deadline:
            return
        if self.index >= 0 and not self.finished:
            print("  timed out", flush=True)
        self.index += 1
        if self.index >= len(self.sweeps):
            raise SystemExit
        tree, standoff, z_min, z_max, columns = self.sweeps[self.index]
        # The controller needs a moment after a sweep before it takes the next.
        time.sleep(5.0)
        self.publisher.publish(ScanRequest(tree_id=tree, standoff_m=standoff,
                                           z_min=z_min, z_max=z_max, columns=columns))
        self.finished = False
        self.deadline = now + SWEEP_TIMEOUT_S
        print(f"sweep {self.index + 1}/{len(self.sweeps)}: tree {tree}, "
              f"standoff {standoff}, z {z_min}-{z_max}, {columns} column(s)", flush=True)


def main():
    if len(sys.argv) > 1:
        sweeps = []
        for argument in sys.argv[1:]:
            tree, standoff, z_min, z_max, columns = argument.split(",")
            sweeps.append((int(tree), float(standoff), float(z_min),
                           float(z_max), int(columns)))
    else:
        sweeps = DEFAULT_SWEEPS

    rclpy.init()
    try:
        rclpy.spin(SweepDriver(sweeps))
    except SystemExit:
        print("all sweeps done", flush=True)


if __name__ == "__main__":
    main()
