import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)

# Dataset Base Path Resolution
_ds_paths = [
    os.path.expanduser("~/ros2_ws/src/oskar/dataset/2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1"),
    os.path.expanduser("~/ros2_ws/src/oskar/datasets/Blossom2024/2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1")
]
DATASET_BASE = next((p for p in _ds_paths if os.path.exists(p)), _ds_paths[0])

# Dataset Image Directories
LEFT_IMAGES_DIR = os.path.join(DATASET_BASE, "Basler_Left_Elstar")
RIGHT_IMAGES_DIR = os.path.join(DATASET_BASE, "Basler_Right_Elstar")

# Calibration & Pose files
LEFT_CALIB_FILE = os.path.join(DATASET_BASE, "SAMSON3_SAMSON4_stereo.yaml")
RIGHT_CALIB_FILE = os.path.join(DATASET_BASE, "SAMSON4_SAMSON3_stereo.yaml")
# Generated explicitly from the Sensorbox1 source bag in DATASET_BASE.  Keep
# derived trajectory data out of bagfile_data so no stage can silently use its copied bag.
POSE_TRAJECTORY_FILE = os.path.join(BASE_DIR, "data", "pose_trajectory_sensorbox1_gnss_cog.npz")
TREE_MAP_FILE = os.path.join(REPO_DIR, "oskar_orchard", "config", "tree_map_real.yaml")

# Segmentation model: a model file of oskar_detection, by name or path. The same
# file the online perception loads; a new object class is a new model file.
SEGMENTATION_MODEL = "flowers_maskdino"
DETECTION_DIR = os.path.join(REPO_DIR, "oskar_detection")
FFM_DIR = os.path.expanduser("~/oskar/Fast-FoundationStereo")
DISPARITY_MODEL_PATH = os.path.expanduser(
    "~/oskar/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth"
)
# Disparity backend: "fast" is the distilled real-time model, "foundationstereo" the full
# offline one. Both repos ship a package called `core`, so step2 puts only the selected
# repo on sys.path.
DISPARITY_MODEL_VARIANT = "fast"
FS_DIR = os.path.expanduser("~/oskar/FoundationStereo")
# cfg.yaml carries the architecture and must sit beside the checkpoint.
FS_MODEL_PATH = os.path.expanduser(
    "~/oskar/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
)
# Tiled inference the repo recommends above ~1K; lets the frame run at a lower DISP_FULL_FRAME_SCALE.
FS_HIERARCHICAL = False

# Output directory for debug states
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step-specific debug folders
SEGMENTATION_OUT_DIR = os.path.join(OUTPUT_DIR, "step1_segmentation")
DISPARITY_OUT_DIR = os.path.join(OUTPUT_DIR, "step2_disparity")
DEPTH_FUSION_OUT_DIR = os.path.join(OUTPUT_DIR, "step3_depth_fusion")
BACKPROJ_OUT_DIR = os.path.join(OUTPUT_DIR, "step4_backprojection")
MULTIVIEW_FUSION_OUT_DIR = os.path.join(OUTPUT_DIR, "step5_multiview_fusion")
BIO_SANITY_OUT_DIR = os.path.join(OUTPUT_DIR, "step6_bio_sanity")
TREE_ASSIGN_OUT_DIR = os.path.join(OUTPUT_DIR, "step7_tree_assignment")
THINNING_OUT_DIR = os.path.join(OUTPUT_DIR, "step8_thinning_decision")

# Camera mounting static transform configuration
CAMERA_SIDE = "left"
# Camera position relative to the GNSS antenna, in base_link (x forward, y left, z up).
# From D. Berschauer's sensorbox drawing (valid from 2025-11-11), Basler camera 2 minus GPS
# antenna 2 = (-263, -100, -390) mm in the box frame, where x is lateral (the antennas sit
# symmetrically at x = +/-281) and z is up. That gives fore-aft 0.100 m, lateral 0.263 m,
# vertical -0.390 m: the camera sits BELOW the antenna, not 1.0 m above it as assumed before.
#
# The old [0.1, 0.0, 1.0] had the fore-aft term right by luck and the lateral term missing
# entirely. Adding the lateral term removes most of the along-row bias against 65 hand-labelled
# trunks: -0.083 m -> +0.022 m, scatter 0.065 -> 0.058 m. Signs confirmed independently by
# canopy geometry: front and back passes image opposite faces of the row, and only a positive
# lateral offset separates the two clouds by ~0.81 m, matching the 0.8-1.0 m canopy diameter.
#
# CAVEAT: the drawing is dated valid-from 2025-11-11 while this dataset was recorded
# 2024-04-15, so the box may have been rebuilt in between. It fits this data well, but confirm
# with the colleague that the geometry was the same in April 2024.
CAMERA_XYZ = [0.100, 0.263, -0.390]

# Hyperparameters
SEG_CONFIDENCE_THRESHOLD = 0.6
SEG_TILED_INFERENCE = True
SEG_TILE_SIZE = 640
SEG_TILE_OVERLAP = 128
SEG_NMS_IOU_THRESHOLD = 0.5
SEG_MIN_BOX_SIZE = 50
SEG_SAVE_VISUALIZATIONS = True
SEG_SAVE_NPZ = True
DISP_MAX_DISP = 384
DISP_VALID_ITERS = 8
DISP_BASELINE_M = 0.13
DISP_PADDING_X = 16
DISP_PADDING_Y = 16
DISP_ROI_HEIGHT = 256
DISP_ROI_WIDTH = 512
DISP_BATCH_SIZE = 16
DISP_CUDNN_BENCHMARK = False


# Compute disparity once for the whole frame instead of per-flower ROI tiles. The ROI path
# leaves each flower only ~112 px of its 512 px tile after the 400 px disparity search
# expansion, and shares one tile between masks whose origins fall within 32 px. At the ~430
# detections/frame MaskDINO produces, that returned a near-constant 0.9 m for 92% of
# detections. Set False to restore the old behaviour.
DISP_FULL_FRAME = True
# Whole frame in one pass; tiling left seams and constant-valued tiles. Scale 3 (2.7 Mpx,
# 0.56 s/frame) is the largest single pass the "fast" backend accepts before cuDNN refuses;
# "foundationstereo" runs scale 2 (6.1 Mpx, 4.6 s, 14.3 GiB).
DISP_FULL_FRAME_SCALE = 3

DEPTH_FUSION_EROSION_RADIUS = 3
DEPTH_FUSION_STD_THRESHOLD = 0.15
DEPTH_FUSION_MIN_VALID_PIXELS = 5
# Near-row working range. The segmentation also detects blossom on the next row over, which
# is real but must not be credited to the tree being driven past. Set either to None to disable.
# Blossom sits on the canopy's NEAR FACE, not at the trunk. Hand-labelled trunks put the
# camera 1.21 m from the trunk line, and the canopy is 0.8-1.0 m across, so blossom facing
# the camera is around 0.7-0.8 m away and the far side around 1.7 m. An earlier 1.0-2.5 m
# gate was set from the trunk range and discarded 74% of detections - most of the near face.
DEPTH_FUSION_MIN_DEPTH_M = 0.5
DEPTH_FUSION_MAX_DEPTH_M = 2.0
DEPTH_FUSION_DROP_LOW_CONFIDENCE = False

# Step 5 duplicate removal. "tracking" follows each flower across the frames that see it
# and merges those views; "dbscan" is the original density clustering. DBSCAN links any two
# points within eps transitively, so at MaskDINO's detection density neighbouring blossoms
# chain into runaway clusters (one absorbed 13,421 of 82,917 observations). A track can gain
# at most one observation per frame, so chaining cannot occur.
FUSION_METHOD = "tracking"
# How far a flower may move between consecutive frames before it is considered a different
# flower. The camera advances ~0.2 m per frame and the pose is good to a few cm, so the same
# blossom should barely move; this is dominated by per-observation depth noise.
TRACK_MATCH_RADIUS_M = 0.06
TRACK_MAX_GAP_FRAMES = 2      # tolerate brief occlusion without splitting the track
TRACK_MIN_OBSERVATIONS = 3    # a blossom seen once or twice is more likely a false positive

# Voxel size ~ observation scatter (0.05 m) and corymb diameter (0.06 m): one voxel is
# roughly one flower cluster.
VOXEL_SIZE_M = 0.05
VOXEL_MIN_OBSERVATIONS = 3

MULTIVIEW_FUSION_EPS = 0.08
MULTIVIEW_FUSION_MIN_SAMPLES = 3

BIO_SANITY_EXPECTED_FLOWERS_PER_CORYMB = 5
BIO_SANITY_INTRA_FLOWER_SPACING_M = 0.03
BIO_SANITY_MERGE_DISTANCE_M = 0.02
BIO_SANITY_CORYMB_RADIUS_M = 0.06

TREE_ASSIGNMENT_MAX_RADIUS_M = 0.7

THINNING_TARGET_FLOWERS_PER_TREE = 20
THINNING_AGRONOMIC_MIN = 15
THINNING_AGRONOMIC_MAX = 25
