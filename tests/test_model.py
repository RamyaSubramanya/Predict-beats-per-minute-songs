import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_process, split_train_test
from src.modelling import model_and_evaluate, predict_test_data

def test_model():
    print(f"Testing the model...")
    
    train_data, test_data = read_and_process()
    X_train, X_val, y_train, y_val = split_train_test(train_data)
    model, predictions, mae, mape = model_and_evaluate(X_train, X_val, y_train, y_val)
    final_predictions = predict_test_data(model, test_data)
    try:
        #checks for nans in pandas df traindata First .any() → checks per column. Second .any() → checks across all columns.
        assert not train_data.isna().any().any()    
        assert X_train.shape[0]==y_train.shape[0]
        assert len(final_predictions)==len(test_data)
        assert mae>=0 #MAE cannot be negative
        assert isinstance(mae, float)   
    except Exception as e:
        print(f"Error ocurred {e}")
    print("Testing process has been completed.")
