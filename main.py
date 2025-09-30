import os
from src.pipeline import read_and_process
from src.modelling import model_and_evaluate, predict_test_data
import joblib
import json

def main():
    print("Starting training pipeline...")
    
    # Data ingestion & processing
    train_data, test_data, X_train, X_val, y_train, y_val = read_and_process("data")
    
    # Model training & evaluation
    model, predictions, mae, mape = model_and_evaluate(X_train, X_val, y_train, y_val)
    
    # Save model & metrics
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.pkl")
    with open("outputs/metrics.json", "w") as f:
        json.dump({"mae": mae, "mape": mape}, f)
        
    # Predict on test data
    final_predictions = predict_test_data(model, test_data)
    final_predictions.to_csv("outputs/final_predictions.csv", index=False)
    
    print("Training & prediction pipeline finished successfully.")

if __name__=="__main__":
    main()