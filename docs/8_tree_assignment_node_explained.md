# How `tree_assignment_node` Works

File: [`oskar_mapping/oskar_mapping/tree_assignment_node.py`](file:///home/oskarstudent/Documents/oskar_projeckt/oskar_student/oskar_mapping/tree_assignment_node.py)

This is the **spatial partitioning and mapping node** in the Phase 1 mapping pipeline — the spatial compiler that maps individual 3D flower landmarks in world coordinates to specific physical tree locations.

---

## Its job in one sentence

It loads the locations of trees from a map configuration file, identifies the closest tree to each flower landmark within a maximum distance threshold, and aggregates confirmed/inferred flower counts per tree.

---

## What it needs (Inputs & Requirements)

1. **Incoming Message:**
   * Topic: `/flowers/landmarks_bio` (`oskar_msgs/msg/FlowerLandmarks`, from `bio_sanity_node`)
2. **Tree Map Coordinates (Configuration):**
   * Loaded from a YAML map file (configured by the `tree_map_file` parameter, defaulting to `config/tree_map.yaml`).
3. **Assignment Thresholds:**
   * `max_radius_m` (parameter): The maximum allowed Euclidean distance in the horizontal plane ($x, y$) to assign a flower landmark to a tree (default: `0.8 m`). Detections further than this limit are ignored.

---

## Where output gets published (Outputs)

1. **`/orchard/per_tree_flowers`** (`oskar_msgs/msg/PerTreeFlowers`)
   * **The aggregated per-tree output.**
   * Each message carries:
     * `tree_ids`: Sorted array of tree identification numbers.
     * `confirmed_counts`: Counts of verified flower landmarks mapped to each tree.
     * `inferred_counts`: Counts of inferred/occluded flower landmarks mapped to each tree.
     * `per_tree_landmarks`: An array of `FlowerLandmarks` messages, each containing the list of all individual landmarks assigned to that tree.

---

## Core spatial partitioning details

### 1. Nearest Tree Lookup
For every incoming 3D landmark, the node computes the horizontal Euclidean distance $d$ to all preloaded trees in the map:
$$d = \sqrt{(x_{landmark} - x_{tree})^2 + (y_{landmark} - y_{tree})^2}$$
The landmark is assigned to the tree $T$ that minimizes $d$, provided $d < \text{max\_radius}$. If no tree is within this radius, the landmark is not assigned.

### 2. Metric Verification Dependancy
Because tree coordinates in the orchard map are metrically accurate (derived from RTK GNSS), this spatial matching will fail to assign landmarks correctly if the inputs to this node have coordinate offset errors. The mapping must first be metrically calibrated.

---

## The code, top to bottom

### 1. TreeAssignmentNode Initialization — lines 20–46
* Declares parameters for the tree map file path and the maximum assignment radius.
* Sets up a subscriber to `/flowers/landmarks_bio` and a publisher on `/orchard/per_tree_flowers`.
* Initializes an empty dictionary `self.trees` and triggers `load_tree_map`.

### 2. Loading Tree Coordinates — lines 47–79
* Resolves the target YAML map path either relative to the current working directory or using `ament_index_python` to find it within the share directory of the `oskar_mapping` package.
* Reads the file and populates the `self.trees` dictionary mapping integer `tree_id` keys to `(x, y)` float tuples.

### 3. The Landmarks Callback — lines 80–136
Runs for every incoming message array:
* **Lines 87–89:** Initializes an empty stats structure for every loaded tree.
* **Lines 91–110:** Loops over all incoming landmarks:
  * Performs the nearest-neighbor search, tracking the closest tree ID within the `max_radius` threshold.
  * If found, sets `landmark.tree_id = closest_tree_id`, increments the corresponding count based on the landmark's `is_inferred` flag, and appends the landmark to that tree's array.
* **Lines 113–129:** Constructs the outgoing `PerTreeFlowers` message, populating sorted list entries of ID values, counts, and nested lists of individual landmarks.
* **Line 132:** Publishes the populated message on `/orchard/per_tree_flowers`.

---

## What we did (Real-World Bag Fixes)

* **Generated a real tree map.** The sim `tree_map_sim.yaml` has only 5 fake trees, so we built
  **`config/tree_map_real.yaml`** with all **238** real orchard trees (x = ENU East, y = ENU
  North — the *same* frame as `/flowers/world_obs`) from the geojson via `pose_trajectory.npz`.
  Point the node at it: `-p tree_map_file:=.../config/tree_map_real.yaml`.
* **Widened `max_radius_m` 0.8 → 8.0 for a plumbing test (and why).** Because the world flowers
  are ~5–8 m off the trees (calibration gap), nothing assigns at the proper 0.8 m radius. We use
  `-p max_radius_m:=8.0` only to confirm the assignment *logic* fires — flowers bind to the
  nearest tree (the *wrong* tree, but it proves the node runs).
* **Result:** loads 238 trees and publishes `/orchard/per_tree_flowers` (full 238-tree skeleton).
  At radius 8.0, a caught populated frame showed **one tree with `confirmed_count: 6`** — the
  assignment logic fires. Most frames are all-zero because the upstream is sparse (empty input →
  empty assignment). Use `ros2 topic echo … --field confirmed_counts | grep -m1 '[1-9]'` to
  catch a non-empty frame (`--once` mostly catches empty ones).
* **Assumption:** correct, meaningful assignment requires **metric calibration** so flowers sit
  within 0.8 m of their true tree. `max_radius_m` is an algorithm knob, not calibration.
  (Full assumptions: `BAG_PIPELINE_BRINGUP.md` §7 / §9c.)
