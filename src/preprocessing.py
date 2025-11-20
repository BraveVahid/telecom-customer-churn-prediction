import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from data_loader import load_data


# Sectoin 3.2.1.
def clean_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    removed_features = ["ZONE1", "ZONE2", "MRG", "user_id", "REGION", "TOP_PACK", "FREQ_TOP_PACK"]
    
    train = train.drop(columns=removed_features)
    test = test.drop(columns=removed_features)
    
    return train, test 


# Sectoin 3.2.2.
def encode_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = ["TENURE"]
    ordinal_encoder = OrdinalEncoder()
    
    train[categorical_cols] = ordinal_encoder.fit_transform(train[categorical_cols])
    test[categorical_cols] = ordinal_encoder.transform(test[categorical_cols])
    
    return train, test


# Sectoin 3.2.3.
def impute_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    enable_iterative_imputer

    y_train = train["CHURN"]
    x_train = train.drop(columns=["CHURN"])
    x_test = test

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        initial_strategy="median",
        max_iter=20,
        tol=1e-5,         
        random_state=42,
        verbose=0
    )

    x_train_imputed = imputer.fit_transform(x_train)
    x_test_imputed = imputer.transform(x_test)

    x_train = pd.DataFrame(x_train_imputed, columns=x_train.columns).round().astype(int) 
    x_test = pd.DataFrame(x_test_imputed, columns=x_test.columns).round().astype(int) 
        
    return x_train, x_test, y_train


# Sectoin 3.2.4.
def scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = x_train.columns

    robust_scaler = RobustScaler()
    x_train_scaled = robust_scaler.fit_transform(x_train)
    x_test_scaled = robust_scaler.transform(x_test)

    std_scaler = StandardScaler()
    x_train_scaled = std_scaler.fit_transform(x_train_scaled)
    x_test_scaled = std_scaler.transform(x_test_scaled)

    min_max_scaler = MinMaxScaler()
    x_train_scaled = min_max_scaler.fit_transform(x_train_scaled)
    x_test_scaled = min_max_scaler.transform(x_test_scaled)

    scaled_train = pd.DataFrame(x_train_scaled, columns=feature_columns)
    scaled_test = pd.DataFrame(x_test_scaled, columns=feature_columns)
    
    return scaled_train, scaled_test


def run_preprocessing():
    train_df, test_df = load_data()
    
    cleaned_train, cleaned_test = clean_data(train=train_df, test=test_df)
    
    encoded_train, encoded_test = encode_data(train=cleaned_train, test=cleaned_test)
    
    x_train, x_test, y_train = impute_data(train=encoded_train, test=encoded_test)
    
    scaled_train, scaled_test = scale_data(x_train, x_test)
    
    return scaled_train, scaled_test, y_train


scaled_train, scaled_test, y_train = run_preprocessing()
print(len(scaled_train))
print(len(scaled_test))
print(len(y_train))
