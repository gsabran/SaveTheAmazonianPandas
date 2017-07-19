LABELS = ['clear', 'cloudy', 'haze', 'partly_cloudy', 'agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'conventional_mine', 'cultivation', 'habitation', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']
WEATHER_IDX = [0, 1, 2, 3]

TRAIN_DATA_DIR = "./rawInput/train-jpg"
TEST_DATA_DIR = "./rawInput/test-jpg"
DATA_DIR = "./rawInput/train-jpg-augmented"
ORIGINAL_LABEL_FILE = "./rawInput/train.csv"
LABEL_FILE = "./rawInput/train-augmented.csv"

IMG_ROWS = 256
IMG_COLS = 256
CHANNELS = 3
NUM_TAGS = 13
NUM_WEATHER = 4

IMG_SCALE = 0.8