from src.preprocessing import run_preprocessing
from src.model import get_triboost_model
from src.train import train_model

if __name__ == "__main__":
    print("=" * 92)
    print("  XAI-CHURN TRIBOOST — FULLY REPRODUCIBLE IMPLEMENTATION")
    print("  A Data-driven Approach with Explainable AI for Customer Churn Prediction")
    print("  Daniyal Asif¹ · Muhammad Shoaib Ali¹ · Aiman Mukheimer²")
    print("  Results in Engineering 26 (2025) 104629 — Elsevier")
    print("  DOI: 10.1016/j.rineng.2025.104629")
    print("=" * 92)

    print("\nSection 3.2 – 3.6: Data Preprocessing Pipeline (Exact Paper Replication)")
    print("   • Data Cleaning          → Removing irrelevant features (user_id, REGION, TOP_PACK, etc.)")
    print("   • Encoding               → Ordinal encoding for TENURE")
    print("   • Imputation             → IterativeImputer (BayesianRidge, median init, max_iter=20)")
    print("   • Scaling                → Sequential: Robust → Standard → MinMax")
    print("   • Feature Engineering    → OFF_NET = ORANGE + TIGO")
    print("   • Feature Selection      → 10 Boruta-confirmed features (Table 6)")
    print("   • Balancing              → SMOTE (sampling_strategy=0.5)")

    print("\nExecuting preprocessing pipeline...")
    x_train, x_test, y_train, y_test = run_preprocessing()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    print(f"   Training set shape : {x_train.shape}  →  Churn rate after SMOTE: {y_train.mean():.4f}")
    print(f"   Test set shape     : {x_test.shape}")

    print("\nSection 3.7: Proposed XAI-Churn TriBoost Ensemble (Equation 8)")
    print("   • Base models: XGBoost (w=2), CatBoost (w=1), LightGBM (w=3)")
    print("   • Voting: Soft voting with weighted probabilities")
    print("   • Hyperparameters: Exactly from Table 7 (RandomizedSearchCV results)")

    print("\nBuilding TriBoost ensemble...")
    model = get_triboost_model()

    print("\nTraining in progress...")
    train_model(model, x_train, y_train)

    print("\n" + "=" * 92)
    print("  TRAINING COMPLETED SUCCESSFULLY")
