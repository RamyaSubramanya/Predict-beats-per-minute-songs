
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_process, split_train_test

import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def model_and_evaluate(X_train, X_val, y_train, y_val):
    """build a model and evaluate the metrics

    Args:
        X_train (df): train data that has features only
        X_val (df): validation data that has features only
        y_train (df): train data that has targets only
        y_val (df): validation data that has targets only

    Returns:
        model, predictions, mae, mape
    """
    
    model = LinearRegression()
    print(f"Model selected is {model}.")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    mape = round(mean_absolute_percentage_error(y_val, predictions),2)
    print(f"Mean absolute error:{mae:.2f} and model's predictions are off by {mape}%")
    return model, predictions, mae, mape
    
    
def predict_test_data(model, test_data):
    print(f"Predicting on the Test dataset using the {model} model.")
    final_predictions = pd.DataFrame(model.predict(test_data))
    final_predictions.to_csv("predictions.csv", index=False)
    print(f"Final predictions have been made on the test dataset and saved as a csv file.")
    return final_predictions