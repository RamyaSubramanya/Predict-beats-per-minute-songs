import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_process, split_train_test
from src.modelling import model_and_evaluate, predict_test_data

if __name__=="__main__":
    print(f"Executing this from {__name__}")
    train_data, test_data = read_and_process()
    X_train, X_val, y_train, y_val = split_train_test(train_data)
    model, predictions, mae, mape = model_and_evaluate(X_train, X_val, y_train, y_val)
    final_predictions = predict_test_data(model, test_data)
    
