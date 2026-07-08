import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_DIR = os.path.dirname(BASE_DIR)

# Dataset Image Directories
LEFT_IMAGES_DIR = os.path.expanduser(
    "~/ros2_ws/src/oskar/datasets/Blossom2024/2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1/Basler_Left_Elstar"
)
RIGHT_IMAGES_DIR = os.path.expanduser(
    "~/ros2_ws/src/oskar/datasets/Blossom2024/2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1/Basler_Right_Elstar"
)

# Calibration & Pose files
LEFT_CALIB_FILE = os.path.expanduser(
    "~/ros2_ws/src/oskar/datasets/Blossom2024/2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1/SAMSON3_SAMSON4_stereo.yaml"
)
RIGHT_CALIB_FILE = os.path.expanduser(
    "~/ros2_ws/src/oskar/datasets/Blossom2024/2024-04-15_10-59-41_Bluete_Elstar_Flaeche_A27_Esteburg_Sensorbox1/SAMSON4_SAMSON3_stereo.yaml"
)
POSE_TRAJECTORY_FILE = os.path.join(MAPPING_DIR, "bagfile_data", "pose_trajectory.npz")
TREE_MAP_FILE = os.path.join(MAPPING_DIR, "config", "tree_map_real.yaml")

# Model checkpoints
SEGMENTATION_MODEL_PATH = os.path.expanduser(
    "~/oskar/synthetic_apple_flowers/results/exp3_real_real/train/model_final.pth"
)
FFM_DIR = os.path.expanduser("~/oskar/Fast-FoundationStereo")
DISPARITY_MODEL_PATH = os.path.expanduser(
    "~/oskar/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth"
)

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
CAMERA_XYZ = [0.1, 0.0, 1.0]

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


DEPTH_FUSION_EROSION_RADIUS = 3
DEPTH_FUSION_STD_THRESHOLD = 0.15
DEPTH_FUSION_MIN_VALID_PIXELS = 5

MULTIVIEW_FUSION_EPS = 0.08
MULTIVIEW_FUSION_MIN_SAMPLES = 3

BIO_SANITY_EXPECTED_FLOWERS_PER_CORYMB = 5
BIO_SANITY_INTRA_FLOWER_SPACING_M = 0.03
BIO_SANITY_MERGE_DISTANCE_M = 0.02
BIO_SANITY_CORYMB_RADIUS_M = 0.06

TREE_ASSIGNMENT_MAX_RADIUS_M = 0.8

THINNING_TARGET_FLOWERS_PER_TREE = 20
THINNING_AGRONOMIC_MIN = 15
THINNING_AGRONOMIC_MAX = 25
