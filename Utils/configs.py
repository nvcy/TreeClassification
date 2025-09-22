# Define BASE DIRECTORY
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"dataset"
TRAIN_DIR = DATA_DIR/"train"
TEST_DIR = DATA_DIR/"test"

# Define where models will be saved

MODEL_DIR = BASE_DIR/"outputs"/"models"

# Define Features EXtracted directory
    # For LBP 2D EXtractor
FEATURES_DIR = BASE_DIR/"outputs"/"features"
TRAIN_FEATURES_DIR = FEATURES_DIR/"train"
TEST_FEATURES_DIR = FEATURES_DIR/"test"
    # For FPFX 3D EXtractor
FPFH_FEATURES_DIR = BASE_DIR/"outputs"/"features3d"
TRAIN_FPFH_FEATURES_DIR = FPFH_FEATURES_DIR/"train"
TEST_FPFH_FEATURES_DIR = FPFH_FEATURES_DIR/"test"

# Define where all the results will be saved

RESULTS_DIR = BASE_DIR/"outputs"/"results"


# Define Multiview Images directory

MULTIVIEW_DIR = BASE_DIR/"outputs"/"MultiViewData"

# Define Multiview Train & test Data directory
MULTIVIEW_TRAIN_DIR = MULTIVIEW_DIR/"train"
MULTIVIEW_TEST_DIR = MULTIVIEW_DIR/"test"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)