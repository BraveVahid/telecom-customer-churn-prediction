from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# Section 3.7
def get_triboost_model() -> VotingClassifier:
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    cat = CatBoostClassifier(
        iterations=300,
        depth=3,
        learning_rate=0.1,
        l2_leaf_reg=5,
        colsample_bylevel=1.0,
        bagging_temperature=1,
        random_strength=1,
        random_state=42,
        verbose=0,
        thread_count=-1,
    )

    lgb = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0,
        reg_lambda=0,
        min_split_gain=1,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    triboost = VotingClassifier(
        estimators=[
            ("xgboost", xgb),
            ("catboost", cat),
            ("lightgbm", lgb),
        ],
        voting="soft",
        weights=[2, 1, 3],
    )

    return triboost
