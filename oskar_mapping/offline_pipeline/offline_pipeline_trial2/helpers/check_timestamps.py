#!/usr/bin/env python3
"""Check exact timestamps for code files, model checkpoint, and cache files."""

import os
import sys
import datetime

WORKSPACE_DIR = "/home/workstation/ros2_ws/src/oskar"
OFFLINE_DIR = os.path.join(WORKSPACE_DIR, "oskar_mapping/offline_pipeline")
sys.path.append(OFFLINE_DIR)

import config

def get_formatted_mtime(path):
    if not os.path.exists(path):
        return f"FILE NOT FOUND: {path}"
    mtime = os.path.getmtime(path)
    dt = datetime.datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

step1_path = os.path.join(OFFLINE_DIR, "step1_segmentation_tiled.py")
config_path = os.path.join(OFFLINE_DIR, "config.py")
model_path = os.path.expanduser(config.SEGMENTATION_MODEL_PATH)
cache_dir = os.path.join(OFFLINE_DIR, "data", "cache")

print("=== ACTUAL FILE MODIFICATION TIMESTAMPS ===")
print(f"step1_segmentation_tiled.py : {get_formatted_mtime(step1_path)}")
print(f"config.py                    : {get_formatted_mtime(config_path)}")
print(f"model_final.pth              : {get_formatted_mtime(model_path)}")

if os.path.exists(cache_dir):
    cache_files = sorted([os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.pkl.gz')])
    if cache_files:
        first_cache = cache_files[0]
        last_cache = cache_files[-1]
        print(f"Cache dir ({len(cache_files)} files)   : First file {os.path.basename(first_cache)} mtime = {get_formatted_mtime(first_cache)}")
        print(f"                               Last file {os.path.basename(last_cache)} mtime = {get_formatted_mtime(last_cache)}")
else:
    print(f"Cache dir NOT FOUND at {cache_dir}")
