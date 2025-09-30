import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split



def read_and_process(data_path="data"):
    train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
    train_data = train_data.iloc[:,1:]
    
    test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_data = test_data.iloc[:,1:]
    print("Train and test data have been loaded successfully.")

    #split train, test data
    X = train_data.drop(columns=['BeatsPerMinute'])
    y = train_data['BeatsPerMinute']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42)
    print("Train data has been split into training and validation set.")
    
    # Consider saving the train/validation split as CSV in outputs/ if you want Azure ML run artifact tracking:
    os.makedirs("outputs", exist_ok=True)
    X_train.to_csv("outputs/X_train.csv", index=False)
    X_val.to_csv("outputs/X_val.csv", index=False)
    y_train.to_csv("outputs/y_train.csv", index=False)
    y_val.to_csv("outputs/y_val.csv", index=False)
    return train_data, test_data, X_train, X_val, y_train, y_val