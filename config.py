import os
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
X_TRAIN_PATH = os.getenv("X_TRAIN_PATH")
X_TEST_PATH = os.getenv("X_TEST_PATH")
Y_TRAIN_PATH = os.getenv("Y_TRAIN_PATH")
Y_TEST_PATH = os.getenv("Y_TEST_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
