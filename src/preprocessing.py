import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from imblearn.over_sampling import SMOTE
import sys
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from config import DATASET_PATH, X_TEST_PATH, X_TRAIN_PATH, Y_TEST_PATH, Y_TRAIN_PATH


# Section 3.1. Dataset
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATASET_PATH)
    return data


# Section 3.2.1. Preprocessing - Data Cleaning
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    removed_features = ["ZONE1", "ZONE2", "MRG", "user_id", "REGION", "TOP_PACK", "FREQ_TOP_PACK"]
    data = data.drop(columns=removed_features)
    return data 


# Section 3.2.2. Preprocessing - Data Encoding
def encode_data(data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["TENURE"]
    ordinal_encoder = OrdinalEncoder()
    data[categorical_cols] = ordinal_encoder.fit_transform(data[categorical_cols])
    return data


# Section 3.2.3. Preprocessing - Data Imputation
def impute_data(data: pd.DataFrame) -> pd.DataFrame:
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        initial_strategy="median",
        max_iter=20,
        tol=1e-5,         
        random_state=42,
        verbose=2
    )
    cols = data.columns
    data = pd.DataFrame(imputer.fit_transform(data), columns=cols).round().astype(int)  
    return data


# Section 3.2.4. Preprocessing - Data Scaling
def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in data.columns if c != "CHURN"]
    data[feature_cols] = RobustScaler().fit_transform(data[feature_cols])
    data[feature_cols] = StandardScaler().fit_transform(data[feature_cols])
    data[feature_cols] = MinMaxScaler().fit_transform(data[feature_cols])
    return data


# Section 3.3. Feature Engineering
def create_off_net_col(data: pd.DataFrame) -> pd.DataFrame:
    data["OFF_NET"] = data["ORANGE"] + data["TIGO"]
    return data
    

# Section 3.4. Feature Selection
def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
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
        "TENURE",
        "CHURN"
    ]
    return data[selected_features]


# Section 3.5. Data Splitting
def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = data.drop("CHURN", axis=1)
    y = data["CHURN"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y, 
        test_size=0.4, 
        random_state=42, 
        stratify=y
    )
    return x_train, x_test, y_train, y_test


# Section 3.6 - Data Balancing
def apply_smote_balancing(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(
        sampling_strategy=0.5,
        k_neighbors=5,
        random_state=42
    )
    x_train, y_train = smote.fit_resample(x_train, y_train)
    return x_train, y_train


def run_preprocessing():
    data = load_data()
    data = clean_data(data)
    data = encode_data(data)
    data = impute_data(data)
    data = scale_data(data)
    data = create_off_net_col(data)
    data = feature_selection(data)     
    x_train, x_test, y_train, y_test = split_data(data)
    x_train, y_train = apply_smote_balancing(x_train, y_train)
    
    x_train.to_csv(X_TRAIN_PATH)
    x_test.to_csv(X_TEST_PATH)
    y_train.to_csv(Y_TRAIN_PATH)
    y_test.to_csv(Y_TEST_PATH)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
