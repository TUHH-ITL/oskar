#!/usr/bin/env python3
"""Report the simulator's real-time factor and the camera's rate in both clocks.

Wall clock says how long a scan takes to sit through. Simulation time is what
the perception and the tracker actually run on, so a low wall rate is only a
problem if the simulation rate is low with it.

  ros2 run oskar_online_perception ... is not needed; run this directly:
  python3 measure_sim_rate.py [topic] [seconds]
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image

# The interesting topics are the big ones, so the probe has to take whatever
# type the topic carries rather than assume camera_info.
TYPES = {"sensor_msgs/msg/CameraInfo": CameraInfo, "sensor_msgs/msg/Image": Image}


class RateProbe(Node):
    def __init__(self, topic, duration):
        super().__init__("measure_sim_rate")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # Discovery has not necessarily finished when the node is constructed,
        # so the type lookup comes back empty unless it is given a moment.
        message_type, found = None, []
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            found = dict(self.get_topic_names_and_types()).get(topic, [])
            message_type = next((TYPES[t] for t in found if t in TYPES), None)
            if message_type is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if message_type is None:
            raise RuntimeError(f"{topic} carries {found or 'nothing discoverable'}, "
                               "which this probe cannot read")
        self.create_subscription(message_type, topic, self._on_camera, qos)
        self.create_subscription(Clock, "/clock", self._on_clock, qos)
        self.stamps, self.wall = [], []
        self.clock_first = self.clock_last = None
        self.wall_start = time.perf_counter()
        self.duration = duration
        self.topic = topic
        self.create_timer(0.2, self._check)
        self.get_logger().info(f"measuring {topic} for {duration:.0f} s of wall clock")

    def _on_camera(self, msg):
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.wall.append(time.perf_counter())

    def _on_clock(self, msg):
        t = msg.clock.sec + msg.clock.nanosec * 1e-9
        if self.clock_first is None:
            self.clock_first = t
        self.clock_last = t

    def _check(self):
        elapsed = time.perf_counter() - self.wall_start
        if elapsed < self.duration:
            return
        self._report(elapsed)
        rclpy.shutdown()

    def _report(self, elapsed):
        if len(self.stamps) < 3:
            self.get_logger().error(f"only {len(self.stamps)} messages on {self.topic}")
            return
        sim_span = self.stamps[-1] - self.stamps[0]
        wall_span = self.wall[-1] - self.wall[0]
        frames = len(self.stamps) - 1
        rtf = sim_span / wall_span if wall_span > 0 else 0.0
        clock_rtf = ((self.clock_last - self.clock_first) / elapsed
                     if self.clock_first is not None else float("nan"))

        print(f"\n{self.topic}")
        print(f"  {frames} frames over {wall_span:.1f} s of wall clock, "
              f"{sim_span:.1f} s of simulation")
        print(f"  wall clock rate      : {frames / wall_span:6.2f} Hz")
        print(f"  simulation time rate : {frames / sim_span:6.2f} Hz   <- what the "
              f"perception and tracker run on")
        print(f"  real-time factor     : {rtf:6.2f}  (from /clock: {clock_rtf:.2f})")
        for speed in (0.05, 0.10, 0.20):
            step = speed / (frames / sim_span)
            verdict = "ok" if step < 0.025 else "TOO FAST for a 25 mm match radius"
            print(f"  at scan_speed {speed:.2f} m/s the camera moves "
                  f"{step * 1000:5.1f} mm per frame  {verdict}")


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/wrist_cam/left/camera_info"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    rclpy.init()
    node = RateProbe(topic, duration)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
