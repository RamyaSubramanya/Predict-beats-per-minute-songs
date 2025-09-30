import os
from src.pipeline import read_and_process

import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
import json


def model_and_evaluate(X_train, X_val, y_train, y_val):
    model = LinearRegression()
    print(f"Model selected is {model}.")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    mape = round(mean_absolute_percentage_error(y_val, predictions),2)
    print(f"Mean absolute error:{mae:.2f} and model's predictions are off by {mape}%")
    
    # Save model to a file (joblib/pickle/onnx) in outputs/ folder for AzureML artifacts.
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.pkl")
    
    # Save metrics
    with open("outputs/metrics.json", "w") as f:
        json.dump({"mae": mae, "mape": mape}, f)
      
    #save more metrics or info for monitoring   (optional):
    # with open("outputs/model_info.json", "w") as f:
    #     json.dump({"coef": model.coef_.tolist(), "intercept": model.intercept_}, f)
    return model, predictions, mae, mape
    
    
def predict_test_data(model, test_data):
    print(f"Predicting on the Test dataset using the {model} model.")
    final_predictions = pd.DataFrame(model.predict(test_data), columns=["Predicted_BPM"])
    final_predictions.to_csv("outputs/predictions.csv", index=False)
    print(f"Final predictions have been made on the test dataset and saved as a csv file.")
    return final_predictions