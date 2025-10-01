import pandas as pd
from sklearn.model_selection import train_test_split
import os

def read_and_process(train_path, test_path):
    """
    Read train and test data from given paths and process them.

    Args:
        train_path (str): path to training CSV
        test_path (str): path to test CSV

    Returns:
        train_data (DataFrame)
        test_data (DataFrame)
        X_train, X_val, y_train, y_val
    """
    # Load CSVs
    train_data = pd.read_csv(os.path.join(train_path, 'train.csv')) if os.path.isdir(train_path) else pd.read_csv(train_path)
    test_data  = pd.read_csv(os.path.join(test_path, 'test.csv')) if os.path.isdir(test_path) else pd.read_csv(test_path)

    # Remove first column if needed (like your original code)
    train_data = train_data.iloc[:, 1:]
    test_data = test_data.iloc[:, 1:]

    # Split features and target
    X = train_data.drop(columns=['BeatsPerMinute'])
    y = train_data['BeatsPerMinute']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)

    print("Data loaded and split successfully.")
    return train_data, test_data, X_train, X_val, y_train, y_val
