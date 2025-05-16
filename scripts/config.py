# General settings
SEED = 42

# Column names
TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "cyberbullying_type"
BINARY_LABEL_COLUMN = "is_cyberbullying"
MULTI_LABEL_COLUMN = "cyberbullying_label"

# File paths
RAW_DATA_PATH = "/Users/manuelemessere/Documents/Università /a) corsi/Human Language Technologies/HLT24_25/hlt_projct/cyberbullying/data/raw_data/cyberbullying_tweets.csv"
PROCESSED_DATA_PATH = "/Users/manuelemessere/Documents/Università /a) corsi/Human Language Technologies/HLT24_25/hlt_projct/cyberbullying/data/processed_data"
INTERIM_DATA_PATH = "/Users/manuelemessere/Documents/Università /a) corsi/Human Language Technologies/HLT24_25/hlt_projct/cyberbullying/data/interim/"

# Models path
W2V_PATH = "/Users/manuelemessere/Documents/Università /a) corsi/Human Language Technologies/HLT24_25/hlt_projct/cyberbullying/models/word_embedding/GoogleNews-vectors-negative300.bin"

# Interim dataset files (after preprocessing and splitting)
TRAIN_SET_PATH = INTERIM_DATA_PATH + "train.csv"
VAL_SET_PATH = INTERIM_DATA_PATH + "val.csv"
TEST_SET_PATH = INTERIM_DATA_PATH + "test.csv"

# Output folders
OUTPUT_DIR = "outputs/"
MODEL_DIR = "models/"

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

SIMILARITY_THRESHOLD = 0.85

# Save options
SAVE_SPLITS = True