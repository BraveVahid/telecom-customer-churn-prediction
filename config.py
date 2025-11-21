from dotenv import load_dotenv
from os import getenv

load_dotenv()
DATASET_PATH = getenv("DATASET_PATH")
X_TRAIN_PATH = getenv("X_TRAIN_PATH")
X_TEST_PATH = getenv("X_TEST_PATH")
Y_TRAIN_PATH = getenv("Y_TRAIN_PATH")
Y_TEST_PATH = getenv("Y_TEST_PATH")
