import os
import sys
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from config import DATASET_PATH, X_TEST_PATH, X_TRAIN_PATH, Y_TEST_PATH, Y_TRAIN_PATH


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    removed_features = [
        "ZONE1",
        "ZONE2",
        "MRG",
        "user_id",
        "REGION",
        "TOP_PACK",
        "FREQ_TOP_PACK",
    ]
    return data.drop(columns=removed_features)


def encode_data(data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["TENURE"]
    encoder = OrdinalEncoder()
    data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
    return data


def split_data(data: pd.DataFrame) -> tuple:
    x = data.drop("CHURN", axis=1)
    y = data["CHURN"]
    
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.4,
        train_size=0.6,
        random_state=42,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def impute_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple:
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        initial_strategy="median",
        max_iter=30,
        tol=1e-5,
        random_state=42,
        verbose=1,
    )
    
    train_cols = x_train.columns
    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)
    
    x_train = pd.DataFrame(x_train_imputed, columns=train_cols).round().astype(int)
    x_test = pd.DataFrame(x_test_imputed, columns=train_cols).round().astype(int)
    
    return x_train, x_test


def scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple:
    robust_scaler = RobustScaler()
    x_train_scaled = robust_scaler.fit_transform(x_train)
    x_test_scaled = robust_scaler.transform(x_test)
    
    standard_scaler = StandardScaler()
    x_train_scaled = standard_scaler.fit_transform(x_train_scaled)
    x_test_scaled = standard_scaler.transform(x_test_scaled)
    
    minmax_scaler = MinMaxScaler()
    x_train_scaled = minmax_scaler.fit_transform(x_train_scaled)
    x_test_scaled = minmax_scaler.transform(x_test_scaled)
    
    x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    
    return x_train, x_test


def create_off_net_col(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple:
    x_train["OFF_NET"] = x_train["ORANGE"] + x_train["TIGO"]
    x_test["OFF_NET"] = x_test["ORANGE"] + x_test["TIGO"]
    return x_train, x_test


def feature_selection(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series) -> tuple:
    selected_features = [
        "MONTANT",
        "FREQUENCE_RECH",
        "REVENUE",
        "ARPU_SEGMENT",
        "FREQUENCE",
        "DATA_VOLUME",
        "ON_NET",
        "REGULARITY",
        "OFF_NET",
        "TENURE"
    ]
    
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]
    
    return x_train_selected, x_test_selected


def apply_smote_balancing(x_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    smote = SMOTE(
        sampling_strategy=0.5,
        k_neighbors=5,
        random_state=42,
    )
    return smote.fit_resample(x_train, y_train)


def run_preprocessing() -> tuple:
    data = load_data()
    data = clean_data(data)
    data = encode_data(data)
    
    x_train, x_test, y_train, y_test = split_data(data)
    
    x_train, x_test = impute_data(x_train, x_test)
    x_train, x_test = scale_data(x_train, x_test)
    x_train, x_test = create_off_net_col(x_train, x_test)
    x_train, x_test = feature_selection(x_train, x_test, y_train)
    x_train, y_train = apply_smote_balancing(x_train, y_train)
    
    folder = os.path.dirname(X_TRAIN_PATH)
    os.makedirs(folder, exist_ok=True)
    
    x_train.to_csv(X_TRAIN_PATH, index=False)
    x_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    
    return x_train, x_test, y_train, y_test
