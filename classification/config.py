import torch

# Path config
MODEL_PATH = "outputs/dapt_model"
DATA_PATH = "data/processed/merged_two.csv"
OUTPUT_DIR = "outputs/classification/roberta_textcnn"

# Basic config
MAX_LENGTH = 256
RANDOM_SEED = 3407

BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 20

ROBERTA_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01

DROPOUT = 0.3
CONV_KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 128
USE_BATCH_NORM = True

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_NAMES = [
    "Normal",
    "SQLi",
    "XSS",
    "SSI",
    "XPath",
    "LDAPi",
    "PathTraversal",
    "OSCommandInjection",
]