# How `thinning_decision_node` Works

File: [`oskar_mapping/oskar_mapping/thinning_decision_node.py`](../oskar_mapping/oskar_mapping/thinning_decision_node.py)

This is the **final decision node** of the Phase 1 mapping pipeline — the agronomic brain that
turns per-tree flower counts into a coarse "which trees need thinning, and roughly how much"
plan. It's the last box of the map phase and the hand-off point to Phase 2 (prune).

---

## Its job in one sentence

For each tree, it adds up the confirmed + inferred flowers, compares that total against an
agronomic target, and decides whether the tree needs thinning, how many flowers to remove
(roughly), and how urgent it is (a priority score).

---

## What it needs (Inputs & Requirements)

1. **Incoming Message:**
   * Topic: `/orchard/per_tree_flowers` (`oskar_msgs/msg/PerTreeFlowers`, from `tree_assignment_node`)
2. **Agronomic Parameters:**
   * `target_flowers_per_tree` (parameter): The ideal flower count to aim for (default: `20`).
   * `agronomic_min` (parameter): Lower bound of the healthy range (default: `15`).
   * `agronomic_max` (parameter): Upper bound — above this, the tree is overloaded and needs
     thinning (default: `25`).

---

## Where output gets published (Outputs)

1. **`/orchard/thinning_plan`** (`oskar_msgs/msg/ThinningPlan`)
   * **The Phase 1 result.** Parallel arrays, one entry per tree:
     * `tree_ids`: tree identifiers.
     * `needs_thinning`: boolean — `True` if the tree is overloaded.
     * `rough_removal_count`: how many flowers to remove to reach the target (coarse).
     * `priority_score`: how far over target the tree is (higher = more urgent).

---

## Core decision logic

For each tree, with `total = confirmed_count + inferred_count`:

* **Needs thinning?** `needs_thinning = total > agronomic_max`
  (the tree has more flowers than the healthy upper bound).
* **Rough removal count:** `rough_removal_count = max(0, total − target)`
  (bring it down to the target; never negative).
* **Priority score:** `priority_score = max(0, (total − target) / target)`
  (a normalized "how overloaded" measure; 0 if at/under target).

This is intentionally **coarse** — it's a screening pass to flag trees and estimate workload.
The fine, per-flower "which flowers to actually remove" decision happens in Phase 2.

---

## The code, top to bottom

### 1. ThinningDecisionNode Initialization — lines 17–40
* Declares the three agronomic parameters (target, min, max).
* Subscribes to `/orchard/per_tree_flowers` and sets up the `/orchard/thinning_plan` publisher
  with a `BEST_EFFORT` QoS profile.

### 2. The Flowers Callback — lines 42–74
Runs for every incoming per-tree message:
* **Lines 45–48:** Creates the output `ThinningPlan`, copying the header.
* **Lines 51–64:** Loops over every tree in the message:
  * Sums `confirmed_counts[i] + inferred_counts[i]` into `total`.
  * Computes `needs_thinning`, `rough_removal`, and `priority` with the formulas above.
  * Appends the per-tree results to the output arrays.
* **Line 71:** Publishes the assembled `ThinningPlan` on `/orchard/thinning_plan`.

---

## What we did (Real-World Bag Fixes)

* **No changes needed** — runs on `/orchard/per_tree_flowers` as-is with default parameters.
* **Always produces a full plan.** Because `tree_assignment` always emits all 238 trees (even
  with zero counts), this node always outputs a 238-entry `ThinningPlan` — so `--once` works
  here (unlike the sparse mid-pipeline topics).
* **Result (verified 2026-06-11):** ran it on the live pipeline — it published a full **238-tree**
  `ThinningPlan` with **every `needs_thinning: false`** (and `rough_removal: 0`). That's the node
  working correctly: our per-tree totals are **low** (sparse fusion + non-metric scale), all below
  the `agronomic_max=25` threshold, so nothing is flagged. It does NOT mean "nothing was decided" —
  a complete plan was produced.
* **Assumption:** real thinning flags (`needs_thinning: true`) require realistic per-tree counts,
  which depend on **metric calibration** upstream (so flowers land on the right trees and counts
  are accurate). (Full assumptions: `BAG_PIPELINE_BRINGUP.md` §7 / §9c.)
