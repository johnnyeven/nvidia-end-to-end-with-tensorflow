MODEL_PATH = "./trained/models/"
LOG_PATH = "./trained/logs/"

# Dataset config
DATASET_PATH = "./data/"
DATASET_META_FILE = "driving_data_meta.csv"
INPUT_IMAGE_CROP = [60, -25, 0, 319]  # Keeping regions in the input image in order [start_x, end_x, start_y, end_y]

# Training config
MAX_STEPS = 1000
KEEP_PROB = 0.5  # keep probability used by dropout layer
REGULARIZER_WEIGHT = 0.0  # coefficient used by L2 regularizer
BATCH_SIZE = 128  # training batch size

LOG_INTERVAL = 5
SAVE_INTERVAL = 500
