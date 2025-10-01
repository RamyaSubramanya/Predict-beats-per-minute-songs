import os
import argparse
from src.pipeline import read_and_process
from src.modelling import model_and_evaluate, predict_test_data
import joblib
import json

def main(train_path, test_path):
    print("Starting training pipeline...")

    train_data, test_data, X_train, X_val, y_train, y_val = read_and_process(train_path, test_path)

    model, predictions, mae, mape = model_and_evaluate(X_train, X_val, y_train, y_val)

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.pkl")
    with open("outputs/metrics.json", "w") as f:
        json.dump({"mae": mae, "mape": mape}, f)

    final_predictions = predict_test_data(model, test_data)
    final_predictions.to_csv("outputs/final_predictions.csv", index=False)

    print("Training & prediction pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, default="data/train.csv", help="Path to training dataset")
    parser.add_argument("--test-data-path", type=str, default="data/test.csv", help="Path to test dataset")
    args = parser.parse_args()
    main(args.train_data_path, args.test_data_path)
