#!/usr/bin/env python3
import os
import math
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

DATA_DIR = "/home/workstation/ros2_ws/src/oskar/oskar_mapping/bagfile_data"
BAG_PATH = os.path.join(DATA_DIR, "2024-04-15_10-59-41_A27_Bluete-no_lidar.bag")

def quaternion_to_euler(x, y, z, w):
    """Convert a quaternion to roll, pitch, yaw in degrees."""
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    return math.degrees(roll), math.degrees(pitch)

def main():
    if not os.path.exists(BAG_PATH):
        print(f"Error: Bag file not found at {BAG_PATH}")
        return

    print("Reading /imu/data connections from bag...")
    ts = get_typestore(Stores.ROS1_NOETIC)
    
    timestamps = []
    rolls = []
    pitches = []
    
    with Reader(BAG_PATH) as r:
        for c in r.connections:
            try:
                ts.register(get_types_from_msg(c.msgdef, c.msgtype))
            except Exception:
                pass
                
        count = 0
        for c, t, raw in r.messages():
            if c.topic == "/imu/data":
                m = ts.deserialize_ros1(raw, c.msgtype)
                o = m.orientation
                
                roll, pitch = quaternion_to_euler(o.x, o.y, o.z, o.w)
                
                timestamps.append(t / 1e9)
                rolls.append(roll)
                pitches.append(pitch)
                
                count += 1
                if count % 10000 == 0:
                    print(f"  Processed {count} IMU messages...")

    if not rolls:
        print("Error: No IMU messages found.")
        return

    rolls = np.array(rolls)
    pitches = np.array(pitches)
    timestamps = np.array(timestamps)

    print("\n=== IMU Roll/Pitch Diagnostic Statistics ===")
    print(f"Total IMU Samples: {len(rolls)}")
    print(f"Roll (deg):  Mean={rolls.mean():.3f}, Std={rolls.std():.3f}, Min={rolls.min():.3f}, Max={rolls.max():.3f}")
    print(f"Pitch (deg): Mean={pitches.mean():.3f}, Std={pitches.std():.3f}, Min={pitches.min():.3f}, Max={pitches.max():.3f}")

    # Check for large values
    large_rolls = np.sum(np.abs(rolls) > 15.0)
    large_pitches = np.sum(np.abs(pitches) > 15.0)
    print(f"Rolls > 15 deg:  {large_rolls} samples ({large_rolls/len(rolls)*100:.2f}%)")
    print(f"Pitches > 15 deg: {large_pitches} samples ({large_pitches/len(pitches)*100:.2f}%)")

if __name__ == "__main__":
    main()
