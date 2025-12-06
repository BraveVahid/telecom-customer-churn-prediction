import os
import sys
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from config import MODEL_PATH


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    
    folder = os.path.dirname(MODEL_PATH)
    os.makedirs(folder, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Size: {os.path.getsize(MODEL_PATH) / 1e6:.2f} MB")
