# How `bio_sanity_node` Works

File: [`oskar_mapping/oskar_mapping/bio_sanity_node.py`](file:///home/oskarstudent/Documents/oskar_projeckt/oskar_student/oskar_mapping/bio_sanity_node.py)

This is the **botanical inference and filter node** in the Phase 1 mapping pipeline — the biological corrector that uses apple flower structural priors (such as typical flower counts per cluster) to clean up noise and infer missing or occluded blossoms.

---

## Its job in one sentence

It takes fused landmarks, merges redundant double-detections, groups the cleaned landmarks into clusters (corymbs), and uses biological priors to infer missing flowers that were likely hidden or occluded.

---

## What it needs (Inputs & Requirements)

1. **Incoming Message:**
   * Topic: `/flowers/landmarks` (`oskar_msgs/msg/FlowerLandmarks`, from `multiview_fusion_node`)
2. **Biological Parameters:**
   * `expected_flowers_per_corymb` (parameter): The expected number of flowers in a single cluster/corymb (default: `5`).
   * `intra_flower_spacing_m` (parameter): The average spacing between flowers in the same cluster (default: `0.03 m`).
   * `merge_distance_m` (parameter): The maximum distance between two observations to be considered a duplicate detection of the same flower (default: `0.02 m`).
   * `corymb_radius_m` (parameter): The maximum radius of a single corymb cluster (default: `0.06 m`).

---

## Where output gets published (Outputs)

1. **`/flowers/landmarks_bio`** (`oskar_msgs/msg/FlowerLandmarks`)
   * **The biological landmark output.** Contains the array of both verified and inferred landmarks.
   * Inferred landmarks are flagged with `is_inferred = True` and assigned a lower confidence score (e.g. `0.3`).

---

## Core botanical priors & logic

### 1. Merging Double-Detections
Due to minor tracking noise or overlapping masks, the same flower might be registered twice. If two landmarks are closer than `merge_distance_m`, the node merges them, keeping only the one with the highest confidence value.

### 2. Corymb Clustering
Apple flowers grow in clusters called **corymbs**. The node uses DBSCAN with `eps = corymb_radius_m` to group nearby flowers into candidate corymbs.

### 3. Corymb Completion & Inference
Once the flowers are grouped, the node checks the count against `expected_flowers_per_corymb`:
* **Singleton Detections:** If only one flower is detected, the node assumes the rest of the cluster is occluded. It promotes the singleton to a full corymb by inferring `expected_flowers_per_corymb - 1` missing flowers.
* **Sparse Clusters:** If the cluster has between 2 and `expected_flowers-1` flowers, it infers the missing ones to round it up to `expected_flowers_per_corymb`.
* **Full Clusters:** If the count is equal or higher, it does not infer anything.

### 4. Circular Offset Placement
Inferred flowers are placed in a circle around the group's centroid with a radius of half the `intra_flower_spacing_m` to simulate natural orchard cluster distribution:
$$\text{angle} = \frac{2\pi \times i}{\text{to\_infer}}$$
$$\Delta x = \frac{\text{intra\_spacing}}{2} \times \cos(\text{angle}), \quad \Delta y = \frac{\text{intra\_spacing}}{2} \times \sin(\text{angle})$$

---

## The code, top to bottom

### 1. BioSanityNode Initialization — lines 20–47
* Declares parameters for the expected count, intra-flower spacing, merge distance, and corymb grouping radius.
* Sets up a subscription on `/flowers/landmarks` and a publisher on `/flowers/landmarks_bio`.

### 2. The Landmarks Callback — lines 48–76
Processes each incoming set of landmarks step-by-step:
* **Line 54:** Merges duplicate detections.
* **Line 57:** Clusters the merged landmarks into corymb groups.
* **Lines 60–63:** Iterates over each group, calling `infer_corymb` to inject missing flowers, and extends them into the final output list.
* **Line 72:** Publishes the results on `/flowers/landmarks_bio`.

### 3. Merging Double Detections — lines 77–103
* **Lines 82–85:** Extracts 3D coordinates and fits DBSCAN with `eps = merge_distance` and `min_samples = 1` so every point is assigned to a cluster.
* **Lines 95–101:** Loops over each cluster. If a cluster contains more than one detection, it sorts them by confidence and retains only the highest-confidence landmark, pruning duplicates.

### 4. Grouping into Corymbs — lines 105–122
* Uses DBSCAN with `eps = corymb_radius` and `min_samples = 1` to find larger clusters of flowers that represent distinct corymbs.

### 5. Corymb Completion (Inference) — lines 124–176
* **Lines 133–134:** Adds existing confirmed landmarks directly to the output.
* **Lines 137–150:** Computes the number of flowers to infer (`to_infer`) based on the cluster size.
* **Lines 152–153:** Computes the arithmetic centroid of the cluster.
* **Lines 156–174:** Loops through `to_infer`, calculates circular angular offsets, instantiates new `FlowerLandmark` messages with unique IDs, sets `is_inferred = True`, sets `mean_confidence = 0.3`, and appends them to the output list.

---

## What we did (Real-World Bag Fixes)

* **No parameter changes** — runs as-is on `/flowers/landmarks`.
* **It works WITHOUT calibration** (this corrects an earlier note that it "waits" for it):
  the node is empty-in → empty-out, so it only *looked* empty while multi-view fusion was
  starved. Once fusion was densified (the `eps` fix in the multiview doc), bio_sanity fired.
* **Result:** given 3 confirmed landmarks in, it inflated **each** into a 5-flower corymb —
  1 real (`is_inferred: false`) + 4 inferred (`is_inferred: true`, `observation_count: 0`,
  `mean_confidence: 0.3`, arranged in a ~1.5 cm ring) → 15 landmarks out on
  `/flowers/landmarks_bio`. Logic verified.
* **Caveat / assumption:** `corymb_radius_m` / `intra_flower_spacing_m` are in metres, so the
  *geometry/scale* of the inferred ring (and corymb grouping) is only correct once depth is
  metric (calibration). The inference logic itself is proven now.
  (Full assumptions: `BAG_PIPELINE_BRINGUP.md` §7 / §9c.)
