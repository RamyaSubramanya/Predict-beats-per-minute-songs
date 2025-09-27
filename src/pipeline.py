import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split



def read_and_process():
    """read train and test data from the data_path

    Returns:
        df: train and test data 
    """
    # project_root = Path().resolve().parent

    # train_data_path = os.path.join(project_root, "data", "train.csv")
    train_data = pd.read_csv("data/train.csv")
    train_data = train_data.iloc[:,1:]
    
    # test_data_path = os.path.join(project_root, "data", "test.csv")
    test_data = pd.read_csv("data/test.csv")
    test_data = test_data.iloc[:,1:]
    print("Train and test data have been loaded successfully.")
    return train_data, test_data


def split_train_test(train_data):
    """split train data into train and validation

    Args:
        train_data (df): Includes features and target

    Returns:
        df: X_train, X_val, y_train, y_val
    """
    X = train_data.drop(columns=['BeatsPerMinute'])
    y = train_data['BeatsPerMinute']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
    print("Train data has been split into training and validation set.")
    print()
    return X_train, X_val, y_train, y_val