#!/usr/bin/env python3
"""Modular Step 4: Transforming 3D flower detections to ENU world coordinates.

Can be run individually to save outputs to config.OUTPUT_DIR, or imported in run_pipeline.py.
"""

import os
import numpy as np

import config
import utils

def interpolate_pose(timestamp, t_array, E_array, N_array, U_array, cos_array, sin_array):
    """Interpolates the robot pose at a given timestamp."""
    e = float(np.interp(timestamp, t_array, E_array))
    n = float(np.interp(timestamp, t_array, N_array))
    u = float(np.interp(timestamp, t_array, U_array))
    c = float(np.interp(timestamp, t_array, cos_array))
    s = float(np.interp(timestamp, t_array, sin_array))
    yaw = np.arctan2(s, c)
    return {"e": e, "n": n, "u": u, "yaw": yaw}

def get_base_to_camera_transform(side="right", xyz=[0.1, 0.0, 1.0]):
    """Returns static rotation matrix and translation vector from base_link to camera optical frame."""
    if side == "left":
        # optical_x=+x, optical_y=-z, optical_z=+y
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0]])
    else:  # right
        # optical_x=-x, optical_y=-z, optical_z=-y
        R = np.array([[-1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [0.0, -1.0, 0.0]])
    t = np.array(xyz)
    return R, t

def process_backprojection(detections_3d_results, trajectory_file=None):
    """Transforms 3D camera-frame detections to ENU world-frame coordinates.
    
    Returns a list of accumulated world coordinates [E, N, U, confidence, instance_id].
    """
    if trajectory_file is None:
        trajectory_file = config.POSE_TRAJECTORY_FILE

    # Load trajectory
    traj_data = np.load(trajectory_file)
    t_arr = traj_data["t"]
    E_arr = traj_data["E"]
    N_arr = traj_data["N"]
    U_arr = traj_data["U"]
    yaw_arr = traj_data["yaw"]
    cos_arr = np.cos(yaw_arr)
    sin_arr = np.sin(yaw_arr)

    # Static camera TF parameters (hardcoded defaults matching mapping_params_sim/real)
    camera_side = getattr(config, "CAMERA_SIDE", "right")
    camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
    R_base_to_cam, t_base_to_cam = get_base_to_camera_transform(camera_side, camera_xyz)

    world_observations = []
    total_frames = len(detections_3d_results)

    for idx, frame in enumerate(detections_3d_results):
        timestamp = frame["timestamp"]
        detections = frame["detections"]
        print(f"[{idx+1}/{total_frames}] Backprojecting {len(detections)} flowers for timestamp {timestamp:.3f}...")

        if not detections:
            continue

        # Interpolate robot pose
        pose = interpolate_pose(timestamp, t_arr, E_arr, N_arr, U_arr, cos_arr, sin_arr)
        e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]

        # Map to base_link transform matrix
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        R_map_to_base = np.array([[cos_y, -sin_y, 0.0],
                                   [sin_y, cos_y, 0.0],
                                   [0.0, 0.0, 1.0]])
        t_map_to_base = np.array([e, n, u])

        for det in detections:
            p_cam = np.array([det["x_cam"], det["y_cam"], det["z_cam"]])
            
            # Base link transform
            p_base = R_base_to_cam @ p_cam + t_base_to_cam
            
            # Map frame transform
            p_map = R_map_to_base @ p_base + t_map_to_base

            world_observations.append([
                p_map[0],          # East
                p_map[1],          # North
                p_map[2],          # Up
                det["confidence"],
                det["instance_id"],
                float(frame.get("frame_idx", idx))  # Frame index (0-based)
            ])

    return np.array(world_observations, dtype=np.float32)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 4: Transforming 3D flower detections to ENU world coordinates.")
    parser.add_argument("--max-frames", type=int, default=10, help="Maximum number of frames to process (-1 for all, default: 10)")
    args = parser.parse_args()

    max_frames = args.max_frames

    detections_3d_results = []
    
    idx = 0
    while True:
        if max_frames >= 0 and idx >= max_frames:
            break

        det_path = os.path.join(config.DEPTH_FUSION_OUT_DIR, f"detections_3d_{idx:06d}.npz")
        if not os.path.exists(det_path):
            if max_frames >= 0:
                print(f"Error: Step 3 output {det_path} not found. Run previous steps first!")
                exit(1)
            else:
                break

        data = np.load(det_path)
        insts = data["instance_ids"]
        confs = data["confidences"]
        low_confs = data["low_confidences"]
        positions = data["positions_cam"]
        timestamp = float(data["timestamp"])

        detections = []
        if insts.size > 0:
            for inst_idx in range(len(insts)):
                detections.append({
                    "instance_id": int(insts[inst_idx]),
                    "confidence": float(confs[inst_idx]),
                    "low_confidence": bool(low_confs[inst_idx]),
                    "x_cam": float(positions[inst_idx, 0]),
                    "y_cam": float(positions[inst_idx, 1]),
                    "z_cam": float(positions[inst_idx, 2])
                })

        detections_3d_results.append({
            "frame_idx": idx,
            "timestamp": timestamp,
            "detections": detections
        })
        idx += 1

    print(f"Backprojecting {len(detections_3d_results)} frames...")
    world_obs = process_backprojection(detections_3d_results)
    utils.clear_dir(config.BACKPROJ_OUT_DIR)
    save_path = os.path.join(config.BACKPROJ_OUT_DIR, "world_obs.npy")
    np.save(save_path, world_obs)
    print(f"Saved {len(world_obs)} world observations to {save_path}")
