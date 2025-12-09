import warnings
warnings.filterwarnings('ignore')

import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from src.preprocessing import run_preprocessing
from src.model import get_triboost_model
from src.train import train_model

console = Console()


def print_header():
    header_content = Text(style="dim")
    header_content.append("A data-driven approach with explainable artificial intelligence\n")
    header_content.append("for customer churn prediction in the telecommunications industry\n\n")
    header_content.append("Authors: Daniyal Asif · Muhammad Shoaib Ali · Aiman Mukheimer\n")
    header_content.append("Publication: Results in Engineering 26 (2025) 104629 — Elsevier\n")
    header_content.append("DOI: 10.1016/j.rineng.2025.104629")
    
    console.print(Panel(
        header_content,
        title="XAI-CHURN TRIBOOST (Fully Reproducible Implementation)",
        title_align="center",
        box=box.ROUNDED,
        border_style="dim",
        padding=(1, 2)
    ))


def print_critical_fix():
    fix_text = Text(style="dim")
    fix_text.append("Issue Identified: The original paper's implementation contained a ")
    fix_text.append("train-test split data leakage problem.\n\n")
    fix_text.append("Problem: Imputation, feature engineering, and scaling were applied ")
    fix_text.append("before splitting the data, causing test information leakage.\n\n")
    fix_text.append("Solution: Pipeline reordered — split first, then transform train/test separately.")
    
    console.print(Panel(
        fix_text,
        title="Data Leakage Correction",
        title_align="center",
        box=box.ROUNDED,
        border_style="dim",
        padding=(1, 2)
    ))


def print_preprocessing_pipeline():
    table = Table(box=box.SIMPLE, show_header=True, header_style="dim", border_style="dim")
    table.add_column("Step", style="dim", width=25)
    table.add_column("Operation", style="dim", width=60)
    
    steps = [
        ("1. Data Cleaning", "Removing irrelevant features (user_id, REGION etc.)"),
        ("2. Encoding", "Ordinal encoding for TENURE"),
        ("3. Train-Test Split", "60-40 split — FIXED: Split before any transformation"),
        ("4. Imputation", "IterativeImputer with BayesianRidge"),
        ("5. Feature Engineering", "Create OFF_NET = ORANGE + TIGO"),
        ("6. Feature Selection", "9 Boruta-confirmed features"),
        ("7. Scaling", "Sequential transformation"),
        ("8. Balancing", "SMOTE oversampling"),
    ]
    
    for step, operation in steps:
        table.add_row(step, operation, style="dim") 
    
    console.print(table)


def print_model_architecture():
    arch_text = Text(style="dim")
    arch_text.append("Architecture: Weighted Soft Voting Ensemble\n\n")
    arch_text.append("Base Learners:\n")
    arch_text.append(" • XGBoost (weight=2): 300 est, depth=6, lr=0.1\n")
    arch_text.append(" • CatBoost (weight=1): 300 iter, depth=3, lr=0.1\n")
    arch_text.append(" • LightGBM (weight=3): 200 est, depth=6, lr=0.1\n\n")
    arch_text.append("Ensemble: Soft voting with weights\n")
    arch_text.append("Hyperparameters: RandomizedSearchCV (Table 7)")
    
    console.print(Panel(
        arch_text,
        title="Section 3.7: XAI-Churn TriBoost Ensemble",
        title_align="center",
        box=box.ROUNDED,
        border_style="dim",
        padding=(1, 2)
    ))


def execute_preprocessing():
    console.print("\nExecuting Preprocessing Pipeline...\n", style="dim")
    console.print("Starting data loading, cleaning, and transformation...", style="dim")
    x_train, x_test, y_train, y_test = run_preprocessing()
    console.print("Preprocessing complete!", style="dim")
    time.sleep(0.5)
    return x_train, x_test, y_train, y_test


def display_data_summary(x_train, x_test, y_train, y_test):
    summary = Table(title="Data Summary", box=box.SIMPLE, show_header=True, header_style="dim", border_style="dim")
    summary.add_column("Dataset", style="dim", width=20)
    summary.add_column("Samples", justify="right", style="dim", width=15)
    summary.add_column("Features", justify="right", style="dim", width=15)
    summary.add_column("Churn Rate", justify="right", style="dim", width=15)
    
    summary.add_row("Training Set", f"{x_train.shape[0]:,}", f"{x_train.shape[1]}", f"{y_train.mean():.2%}", style="dim")
    summary.add_row("Test Set", f"{x_test.shape[0]:,}", f"{x_test.shape[1]}", f"{y_test.mean():.2%}", style="dim")
    
    console.print("\n")
    console.print(summary)


def train_triboost(x_train, y_train):
    console.print("\nBuilding and Training TriBoost Ensemble...\n", style="dim")
    model = get_triboost_model()
    console.print("Training XGBoost + CatBoost + LightGBM ensemble...", style="dim")
    train_model(model, x_train, y_train)
    console.print("Training complete!", style="dim")
    time.sleep(0.5)
    return model


def print_next_steps():
    next_steps = Text(style="dim")
    next_steps.append("Next Steps\n\n")
    next_steps.append("1. XAI Analysis → jupyter notebook notebooks/xai_churn_analysis.ipynb\n")
    next_steps.append("    SHAP values, feature importance, decision plots\n\n")
    next_steps.append("2. Model Evaluation → jupyter notebook notebooks/model_evaluation.ipynb\n")
    next_steps.append("    Metrics, confusion matrix, ROC curves")
    
    console.print(Panel(
        next_steps,
        title="Next Steps",
        title_align="center",
        box=box.ROUNDED,
        border_style="dim",
        padding=(1, 2)
    ))


def print_footer():
    footer_text = Text(style="dim")
    footer_text.append("TRAINING COMPLETED SUCCESSFULLY\n\n")
    footer_text.append("Model saved and ready for evaluation and deployment")
    
    console.print("\n")
    console.print(Panel(footer_text, box=box.ROUNDED, border_style="dim", padding=(1, 2)))


def main():
    console.clear()
    
    print_header()
    console.print("\n", style="dim")
    
    print_critical_fix()
    console.print("\n", style="dim")
    
    print_preprocessing_pipeline()
    console.print("\n", style="dim")
    
    x_train, x_test, y_train, y_test = execute_preprocessing()
    display_data_summary(x_train, x_test, y_train, y_test)
    console.print("\n", style="dim")
    
    print_model_architecture()
    console.print("\n", style="dim")
    
    train_triboost(x_train, y_train)
    
    print_next_steps()
    print_footer()
    
    console.print("\n" + "─" * 60 + "\n", style="dim")


if __name__ == "__main__":
    main()
