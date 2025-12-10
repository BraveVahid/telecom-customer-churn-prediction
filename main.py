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


def print_critical_fix():
    print("\n==============================")
    print(" Data Leakage Correction")
    print("==============================\n")
    print("Issue Identified: The original implementation had a train-test split data leakage problem.\n")
    print("Problem: Imputation, feature engineering, and scaling were applied BEFORE splitting,")
    print("         causing test information leakage.\n")
    print("Solution: Split first, then apply transformations separately on train and test.\n")


def print_preprocessing_pipeline():
    print("\n==============================")
    print(" Preprocessing Pipeline")
    print("==============================\n")
    steps = [
        "1. Data Cleaning — Removing irrelevant features",
        "2. Encoding — Ordinal encoding for TENURE",
        "3. Train-Test Split — 60-40 split (fixed position)",
        "4. Imputation — IterativeImputer (BayesianRidge)",
        "5. Feature Engineering — OFF_NET = ORANGE + TIGO",
        "6. Feature Selection — 9 Boruta-confirmed features",
        "7. Scaling — Sequential transformation",
        "8. Balancing — SMOTE oversampling",
    ]
    for s in steps:
        print(s)
    print()


def print_model_architecture():
    print("\n==============================")
    print(" TriBoost Ensemble Architecture")
    print("==============================\n")
    print("Base Learners:")
    print(" • XGBoost (weight=2): 300 est, depth=6, lr=0.1")
    print(" • CatBoost (weight=1): 300 iter, depth=3, lr=0.1")
    print(" • LightGBM (weight=3): 200 est, depth=6, lr=0.1")
    print("\nEnsemble: Soft voting with weights")
    print("Hyperparameters: RandomizedSearchCV (Table 7)\n")


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

    print_critical_fix()

    print_preprocessing_pipeline()

    x_train, x_test, y_train, y_test = execute_preprocessing()
    display_data_summary(x_train, x_test, y_train, y_test)

    print_model_architecture()

    train_triboost(x_train, y_train)

    print_next_steps()
    print_footer()

    print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
