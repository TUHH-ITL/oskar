#!/usr/bin/env python3
"""Step 2a: Extract the robot pose trajectory from the Fraunhofer bag.

Produces a clean table  time -> (E, N, U, yaw)  saved as an .npz, where:
  - position (E,N,U) comes from RTK GNSS (/ublox/fix) projected to a local ENU
    frame whose origin is the TREE-MAP CENTROID (so `map` is centred on the orchard);
  - yaw (heading) comes from GNSS course-over-ground (direction of travel), smoothed,
    NOT from the IMU (whose yaw drifts at the U-turn — see docs/BAG_PIPELINE_BRINGUP.md §9a).

Also stores the 238 tree points in the same ENU frame for validation.

Run (plain python, no ROS needed):
    ~/oskar_venv/bin/python extract_pose_trajectory.py
"""

import argparse
import glob
import json
import math
import os

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

DATA = os.path.expanduser("~/ros2_ws/src/oskar/oskar_mapping/bagfile_data")


def enu_projector(lat0_deg, lon0_deg):
    """Return f(lon,lat)->(E,N) metres, simple equirectangular (fine over <1 km)."""
    R = 6378137.0
    lat0 = math.radians(lat0_deg)

    def f(lon, lat):
        e = math.radians(lon - lon0_deg) * math.cos(lat0) * R
        n = math.radians(lat - lat0_deg) * R
        return e, n

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=f"{DATA}/2024-04-15_10-59-41_A27_Bluete-no_lidar.bag")
    ap.add_argument("--geojson", default=None, help="defaults to the *.geojson in bagfile_data")
    ap.add_argument("--out", default=f"{DATA}/pose_trajectory.npz")
    ap.add_argument("--move-thresh", type=float, default=0.10,
                    help="metres of displacement over the heading window to count as 'moving'")
    ap.add_argument("--heading-window", type=int, default=5,
                    help="+/- N fixes used to compute course-over-ground (10 Hz GNSS)")
    ap.add_argument("--smooth", type=int, default=9, help="circular moving-average window for yaw")
    args = ap.parse_args()

    geojson = args.geojson or glob.glob(f"{DATA}/*.geojson")[0]

    # --- tree centroid -> ENU origin -------------------------------------
    gj = json.load(open(geojson))
    tree_ll = [f["geometry"]["coordinates"][:2]
               for f in gj["features"]
               if f["geometry"]["type"] == "Point"]
    tree_ll = np.array(tree_ll)  # (N,2) lon,lat
    lon0, lat0 = float(tree_ll[:, 0].mean()), float(tree_ll[:, 1].mean())
    proj = enu_projector(lat0, lon0)
    print(f"ENU origin (tree centroid): lat={lat0:.7f} lon={lon0:.7f}  ({len(tree_ll)} trees)")

    # --- read GNSS track from bag ----------------------------------------
    ts = get_typestore(Stores.ROS1_NOETIC)
    gt, lats, lons, alts = [], [], [], []
    with Reader(args.bag) as r:
        for c in r.connections:
            try:
                ts.register(get_types_from_msg(c.msgdef, c.msgtype))
            except Exception:
                pass
        for c, t, raw in r.messages():
            if c.topic == "/ublox/fix":
                m = ts.deserialize_ros1(raw, c.msgtype)
                s = m.header.stamp
                gt.append(s.sec + s.nanosec / 1e9)
                lats.append(m.latitude)
                lons.append(m.longitude)
                alts.append(m.altitude)
    gt = np.array(gt)
    order = np.argsort(gt)
    gt, lats, lons, alts = gt[order], np.array(lats)[order], np.array(lons)[order], np.array(alts)[order]

    E = np.empty(len(gt))
    N = np.empty(len(gt))
    for i, (lo, la) in enumerate(zip(lons, lats)):
        E[i], N[i] = proj(lo, la)
    U = alts - float(alts.mean())

    # --- heading from course-over-ground (windowed), then fill + smooth ---
    w = args.heading_window
    yaw = np.full(len(gt), np.nan)
    for i in range(len(gt)):
        a, b = max(0, i - w), min(len(gt) - 1, i + w)
        dx, dy = E[b] - E[a], N[b] - N[a]
        if math.hypot(dx, dy) > args.move_thresh:
            yaw[i] = math.atan2(dy, dx)  # ENU yaw: CCW from East
    valid = ~np.isnan(yaw)
    print(f"heading valid (moving) at {valid.sum()}/{len(yaw)} fixes; filling the rest")
    idxs = np.where(valid)[0]
    for i in np.where(~valid)[0]:                      # hold nearest valid heading
        yaw[i] = yaw[idxs[np.argmin(np.abs(idxs - i))]]
    k = args.smooth                                    # circular moving average
    cs = np.convolve(np.cos(yaw), np.ones(k) / k, mode="same")
    ss = np.convolve(np.sin(yaw), np.ones(k) / k, mode="same")
    yaw = np.arctan2(ss, cs)

    # --- tree points in ENU (for validation/tree_assignment later) -------
    tE = np.array([proj(lo, la)[0] for lo, la in tree_ll])
    tN = np.array([proj(lo, la)[1] for lo, la in tree_ll])

    np.savez(args.out, t=gt, E=E, N=N, U=U, yaw=yaw,
             origin_lat=lat0, origin_lon=lon0,
             tree_E=tE, tree_N=tN)
    print(f"\nSaved {args.out}")
    print(f"  {len(gt)} poses, t=[{gt[0]:.1f},{gt[-1]:.1f}] ({gt[-1]-gt[0]:.0f}s)")
    print(f"  E[{E.min():.1f},{E.max():.1f}] N[{N.min():.1f},{N.max():.1f}] U[{U.min():.2f},{U.max():.2f}]")
    print(f"  yaw clusters (deg): "
          f"{np.round(np.degrees(np.percentile(yaw,[10,50,90])),1).tolist()}")


if __name__ == "__main__":
    main()
