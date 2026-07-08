# Design Tradeoffs Analysis

This document provides a detailed comparison of execution modes (ROS 2 Online vs. Pure Python Offline) and instance segmentation models (Mask R-CNN vs. YOLOv8/v11) to help align design decisions with the project supervisor.

---

## 1. Execution Modes: ROS 2 Online vs. Offline Python

| Mode | Core Approach | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **ROS 2 Online** | Nodes running asynchronously, communicating over ROS topics (DDS). | - Live visualization in RViz during runs.<br>- Seamless integration with UR20 control and physical robot actuators. | - **Large Serialization Overhead**: Passing 9 MB images and 64 MB masks creates CPU bottle-necks.<br>- **Frame Drops**: Tight scheduling causes frames to be dropped if GPU stalls.<br>- **Harder Debugging**: Multi-process async logic is harder to trace. |
| **Offline Python** | Single script running sequentially, reading images from disk and passing variables directly in memory. | - **Zero Serialization Latency**: No topic copies; data stays in CPU/GPU memory.<br>- **No Frame Drops**: 100% data completeness for post-run analysis.<br>- **Step-by-Step Debugging**: Developers can run and profile individual scripts (e.g. disparity only) on pre-saved data.<br>- **Simple Environment**: No ROS sourcing or build overlay needed. | - No real-time visualization on the physical robot in the field.<br>- Outputs must be written to files (.ply, .geojson) and viewed post-run. |

---

## 2. Segmentation Models: Mask R-CNN vs. YOLO (YOLOv8-Seg / YOLOv11-Seg)

| Model | Core Architecture | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Mask R-CNN** (Current) | Two-stage detector: First proposes regions (RPN), then computes classification, bounding boxes, and pixel-level masks. | - **High Boundary Accuracy**: Exceptional at capturing complex overlapping boundaries (e.g. dense corymb flower structures).<br>- Already trained, tested, and validated on the apple orchard dataset. | - **Extremely Slow**: Takes **~1200 ms** per frame on standard GPU hardware (max throughput ~0.8 Hz).<br>- Heavy memory footprint, limiting parallel batching options. |
| **YOLO-Seg** (Alternative) | Single-stage detector: Predicts bounding boxes and mask coefficients in a single forward pass. | - **Extremely Fast**: Takes **~12–20 ms** with TensorRT optimization (50x-100x faster than Mask R-CNN).<br>- Lightweight model size, enabling easy GPU batching.<br>- Very active open-source support. | - May exhibit slightly lower boundary precision or struggle with heavily overlapping objects in dense clusters.<br>- **Requires Re-training**: Requires retraining the model weights on the apple flower dataset and supervisor approval. |

---

## 3. Tiled vs. Non-Tiled Inference for Segmentation

| Mode | Core Approach | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Standard (Non-Tiled)** | Processes the full image in a single pass at its original size. | - **Extremely Fast**: Takes **~0.15 seconds** per image (GPU).<br>- **Low Disk Usage**: Negligible `.npz` saving time (under 0.01s). | - **Scale Mismatch Blocker**: If the model was trained on 640x640 crops, running on the full high-res image (5328x4608) causes severe scale mismatch, leading to missed flowers (very low recall). |
| **Tiled Inference** | Slices the high-res image into overlapping 640x640 crops, runs inference, filters small boxes, and merges duplicates using NMS. | - **High Detection Recall**: Matches the training scale of the model, allowing detailed detection of individual blossoms on the branches. | - **Extremely Heavy**: Takes **~8.8 seconds** per image on GPU (almost 60x slower).<br>- **Compression Bottleneck**: Writing hundreds of 20MP boolean masks per frame to compressed `.npz` files takes **2 to 3 minutes** per frame (zlib compression bottleneck). |
