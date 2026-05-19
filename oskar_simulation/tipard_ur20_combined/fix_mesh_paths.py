#!/usr/bin/env python3

from pathlib import Path
import re

urdf_path = Path("tipard_ur20_combined.urdf")

text = urdf_path.read_text()

# UR20 + Robotiq
text = text.replace(
    "package://ur20_robotiq_2f85_combined/meshes/",
    "meshes/"
)

# Tipard package path - adjust package name if different
text = re.sub(
    r'package://[^/]+/meshes/',
    'meshes/tipard/',
    text
)

urdf_path.write_text(text)

print("Updated mesh paths in:", urdf_path)
