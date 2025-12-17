import warnings
warnings.filterwarnings('ignore')

import time
from src.preprocessing import run_preprocessing
from src.model import get_triboost_model
from src.train import train_model


def print_header():
    print("\n==============================")
    print(" XAI-CHURN TRIBOOST (Reproducible Implementation)")
    print("==============================\n")
    print("A data-driven approach with explainable artificial intelligence")
    print("for customer churn prediction in the telecommunications industry\n")
    print("Authors: Daniyal Asif · Muhammad Shoaib Ali · Aiman Mukheimer")
    print("Publication: Results in Engineering 26 (2025) 104629 — Elsevier")
    print("DOI: 10.1016/j.rineng.2025.104629\n")

def print_preprocessing_pipeline():
    print("\n==============================")
    print(" Preprocessing Pipeline")
    print("==============================\n")


def print_model_architecture():
    print("\n==============================")
    print(" TriBoost Ensemble Architecture")
    print("==============================\n")


def execute_preprocessing():
    print("\nExecuting Preprocessing Pipeline...\n")
    print("Starting data loading, cleaning, and transformation...\n")
    x_train, x_test, y_train, y_test = run_preprocessing()
    print("Preprocessing complete!\n")
    time.sleep(0.5)
    return x_train, x_test, y_train, y_test


def display_data_summary(x_train, x_test, y_train, y_test):
    print("\n==============================")
    print(" Data Summary")
    print("==============================\n")
    print(f"Training Set → Samples: {x_train.shape[0]:,},  Features: {x_train.shape[1]},  Churn Rate: {y_train.mean():.2%}")
    print(f"Test Set     → Samples: {x_test.shape[0]:,},  Features: {x_test.shape[1]},  Churn Rate: {y_test.mean():.2%}\n")


def train_triboost(x_train, y_train):
    print("\nBuilding and Training TriBoost Ensemble...\n")
    model = get_triboost_model()
    print("Training XGBoost + CatBoost + LightGBM ensemble...\n")
    train_model(model, x_train, y_train)
    print("Training complete!\n")
    time.sleep(0.5)
    return model


def print_next_steps():
    print("\n==============================")
    print(" Next Steps")
    print("==============================\n")
    print("1. XAI Analysis → notebooks/xai_analysis.ipynb")
    print("   Includes SHAP values, feature importance, decision plots\n")
    print("2. Model Evaluation → notebooks/model_evaluation.ipynb")
    print("   Includes metrics, confusion matrix, ROC curves\n")


def print_footer():
    print("\n==============================")
    print(" TRAINING COMPLETED SUCCESSFULLY")
    print("==============================\n")
    print("Model saved and ready for evaluation and deployment\n")


def main():
    print_header()
    print_preprocessing_pipeline()
    x_train, x_test, y_train, y_test = execute_preprocessing()
    display_data_summary(x_train, x_test, y_train, y_test)
    print_model_architecture()
    train_triboost(x_train, y_train)
    print_next_steps()
    print_footer()

if __name__ == "__main__":
    main()
