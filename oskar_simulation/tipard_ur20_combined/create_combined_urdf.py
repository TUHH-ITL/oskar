#!/usr/bin/env python3
"""
Create a combined Tipard + UR20 Robotiq URDF.

Usage:
  python3 create_combined_urdf.py \
    --base /home/workstation/ros2_ws/src/oskar/oskar_simulation/urdf/tipard_new/tipard_robot.urdf \
    --arm /home/workstation/ros2_ws/src/oskar/oskar_simulation/ur20_robotiq_2f85_combined/urdf/ur20_robotiq_2f85_combined.urdf \
    --output /home/workstation/ros2_ws/src/oskar/oskar_simulation/urdf/tipard_ur20_combined/tipard_ur20_combined.urdf \
    --xyz 0.25 0.0 0.75 \
    --rpy 0 0 0
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_robot(path: Path) -> ET.Element:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise SystemExit(f"XML parse error in {path}: {exc}")

    root = tree.getroot()
    if root.tag != "robot":
        raise SystemExit(f"Expected <robot> as root tag in {path}, got <{root.tag}>")
    return root


def collect_names(robot: ET.Element, tag: str) -> list[str]:
    return [elem.attrib["name"] for elem in robot.findall(tag) if "name" in elem.attrib]


def find_duplicates(items: list[str]) -> list[str]:
    counts = Counter(items)
    return sorted([name for name, count in counts.items() if count > 1])


def indent(elem: ET.Element, level: int = 0) -> None:
    # Pretty-print XML for Python versions without ET.indent compatibility concerns
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine Tipard mobile base URDF with UR20 arm URDF.")
    parser.add_argument("--base", required=True, type=Path, help="Path to Tipard mobile robot URDF")
    parser.add_argument("--arm", required=True, type=Path, help="Path to UR20 + gripper URDF")
    parser.add_argument("--output", required=True, type=Path, help="Output combined URDF path")
    parser.add_argument("--parent-link", default="base_link", help="Tipard link where arm is mounted")
    parser.add_argument("--child-link", default="ur20_base_link", help="UR20 root/base link")
    parser.add_argument("--joint-name", default="tipard_to_ur20_mount_joint", help="Name of fixed mounting joint")
    parser.add_argument("--xyz", nargs=3, default=["0.25", "0.0", "0.75"], help="Mount xyz offset")
    parser.add_argument("--rpy", nargs=3, default=["0", "0", "0"], help="Mount rpy rotation")
    parser.add_argument("--force", action="store_true", help="Write output even if duplicate link/joint names are detected")
    args = parser.parse_args()

    if not args.base.exists():
        raise SystemExit(f"Base URDF not found: {args.base}")
    if not args.arm.exists():
        raise SystemExit(f"Arm URDF not found: {args.arm}")

    base_robot = parse_robot(args.base)
    arm_robot = parse_robot(args.arm)

    base_links = collect_names(base_robot, "link")
    arm_links = collect_names(arm_robot, "link")
    base_joints = collect_names(base_robot, "joint")
    arm_joints = collect_names(arm_robot, "joint")

    if args.parent_link not in base_links:
        raise SystemExit(f"Parent link '{args.parent_link}' not found in base URDF.")
    if args.child_link not in arm_links:
        raise SystemExit(f"Child link '{args.child_link}' not found in arm URDF.")

    duplicate_links = sorted(set(base_links).intersection(arm_links))
    duplicate_joints = sorted(set(base_joints).intersection(arm_joints))

    if args.joint_name in base_joints or args.joint_name in arm_joints:
        duplicate_joints.append(args.joint_name)

    if (duplicate_links or duplicate_joints) and not args.force:
        print("Duplicate names detected. Fix these first, or rerun with --force if you know it is safe.\n")
        if duplicate_links:
            print("Duplicate links:")
            for name in duplicate_links:
                print(f"  - {name}")
        if duplicate_joints:
            print("Duplicate joints:")
            for name in sorted(set(duplicate_joints)):
                print(f"  - {name}")
        raise SystemExit(1)

    combined = ET.Element("robot", {"name": "tipard_ur20_combined"})

    # Copy children from both robots, excluding nested robot wrapper only.
    for child in list(base_robot):
        combined.append(copy.deepcopy(child))

    for child in list(arm_robot):
        combined.append(copy.deepcopy(child))

    # Add fixed mounting joint.
    mount_joint = ET.SubElement(combined, "joint", {"name": args.joint_name, "type": "fixed"})
    ET.SubElement(mount_joint, "parent", {"link": args.parent_link})
    ET.SubElement(mount_joint, "child", {"link": args.child_link})
    ET.SubElement(
        mount_joint,
        "origin",
        {
            "xyz": " ".join(args.xyz),
            "rpy": " ".join(args.rpy),
        },
    )

    # Final duplicate check inside combined file.
    all_links = collect_names(combined, "link")
    all_joints = collect_names(combined, "joint")
    final_duplicate_links = find_duplicates(all_links)
    final_duplicate_joints = find_duplicates(all_joints)

    if (final_duplicate_links or final_duplicate_joints) and not args.force:
        print("Final combined URDF still has duplicate names. Output was not written.\n")
        if final_duplicate_links:
            print("Duplicate links:", final_duplicate_links)
        if final_duplicate_joints:
            print("Duplicate joints:", final_duplicate_joints)
        raise SystemExit(1)

    indent(combined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(combined).write(args.output, encoding="utf-8", xml_declaration=True)

    print("Combined URDF created successfully:")
    print(args.output)
    print("\nMount joint:")
    print(f"  parent: {args.parent_link}")
    print(f"  child : {args.child_link}")
    print(f"  xyz   : {' '.join(args.xyz)}")
    print(f"  rpy   : {' '.join(args.rpy)}")
    print("\nNext check:")
    print(f"  check_urdf {args.output}")


if __name__ == "__main__":
    main()
