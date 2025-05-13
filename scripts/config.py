import os

# General settings
SEED = 42
PWD = os.getcwd()

# Column names
TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "cyberbullying_type"
BINARY_LABEL_COLUMN = "is_cyberbullying"

# File paths
RAW_DATA_PATH = f"{PWD}/cyberbullying/data/raw_data/cyberbullying_tweets.csv"
PROCESSED_DATA_PATH = f"{PWD}/cyberbullying/data/processed_data"
INTERIM_DATA_PATH = f"{PWD}/cyberbullying/data/interim/"

# Interim dataset files (after preprocessing and splitting)
TRAIN_SET_PATH = INTERIM_DATA_PATH + "train.csv"
VAL_SET_PATH = INTERIM_DATA_PATH + "val.csv"
TEST_SET_PATH = INTERIM_DATA_PATH + "test.csv"

# Output folders
OUTPUT_DIR = f"{PWD}/outputs/"
MODEL_DIR = f"{PWD}/models/"

# Train/test/val split
TEST_SIZE = 0.10
VALIDATION_SIZE = 0.19 / 0.90  # 19% of the dev set (90%)

# Preprocessing options
REMOVE_STOPWORDS = True
APPLY_LEMMATIZATION = True
APPLY_STEMMING = True

# Language detection settings
LANGUAGE = "en"

# Augmentation settings (to check and implement)
AUGMENT_MINORITY_CLASSES = True
AUGMENTATION_TIMES = 2

# Save options
SAVE_SPLITS = True