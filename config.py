from dotenv import load_dotenv
from os import getenv

load_dotenv()

TRAIN_PATH = getenv("TRAIN_PATH")
TEST_PATH = getenv("TEST_PATH")
TRAIN_PROCESSED_PATH = ""
TEST_PROCESSED_PATH = ""
