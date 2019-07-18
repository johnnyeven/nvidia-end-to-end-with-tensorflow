BASE_PATH = "."
MODEL_PATH = BASE_PATH + "/trained/models/"
MODEL_NAME = "nvidia-end-to-end"
LOG_PATH = BASE_PATH + "/trained/logs/"

# Dataset config
DATASET_PATH = "./data/"
DATASET_META_FILE = "driving_data_meta.csv"
ANGLE_DELTA_CORRECTION_LEFT = 0.2  # steering angle correction for the left camera image
ANGLE_DELTA_CORRECTION_RIGHT = -0.2
INPUT_IMAGE_CROP = [60, -25]  # the input image in order [start_x, end_x], to cut the sky in the image
SPLIT_SIZE = 0.2

# Training config
MAX_STEPS = 30000
KEEP_PROB = 0.5  # keep probability used by dropout layer
REGULARIZER_WEIGHT = 0.0  # coefficient used by L2 regularizer
BATCH_SIZE = 128  # training batch size

LOG_INTERVAL = 5
SAVE_INTERVAL = 500
