#!/usr/bin/env python3
"""Draw a contact sheet of a captured split, masks painted on the frames.

Reading the annotation counts tells you a set exists; only looking at it tells
you the masks sit on flowers.

    python3 inspect_dataset.py <batch_root> --split train --out sheet.png
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, default=Path("dataset_sheet.png"))
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    annotations = args.batch_root / "annotations" / f"{args.split}.json"
    data = json.loads(annotations.read_text())
    by_image = {}
    for annotation in data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    random.seed(args.seed)
    chosen = random.sample(data["images"], min(args.rows * args.columns, len(data["images"])))

    tiles = []
    for image in chosen:
        frame = cv2.imread(str(args.batch_root / args.split / image["file_name"]))
        overlay = frame.copy()
        for annotation in by_image.get(image["id"], []):
            overlay[mask_utils.decode(annotation["segmentation"]).astype(bool)] = (0, 0, 255)
        frame = cv2.addWeighted(frame, 0.45, overlay, 0.55, 0)
        cv2.putText(frame, f"{image['file_name']}  {len(by_image.get(image['id'], []))} flowers",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        height = int(args.width * frame.shape[0] / frame.shape[1])
        tiles.append(cv2.resize(frame, (args.width, height)))

    while len(tiles) < args.rows * args.columns:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r * args.columns:(r + 1) * args.columns])
                       for r in range(args.rows)])
    cv2.imwrite(str(args.out), sheet)

    areas = np.array([a["area"] for a in data["annotations"]])
    print(f"{len(data['images'])} frames, {len(areas)} instances, "
          f"{len(areas) / max(1, len(data['images'])):.1f} per frame")
    print(f"mask area px: p5 {np.percentile(areas, 5):.0f}  median "
          f"{np.median(areas):.0f}  p95 {np.percentile(areas, 95):.0f}  "
          f"max {areas.max():.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
