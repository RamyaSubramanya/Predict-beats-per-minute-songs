import json
import joblib
import pandas as pd
from azureml.core.model import Model

# Global model variable
model = None

def init(model_path=None):
    global model
    try:
        if model_path:
            model = joblib.load(model_path)
        else:
            from azureml.core.model import Model
            model_path = Model.get_model_path(model_name="bpm_model")
            model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def run(raw_data):
    """
    Run predictions on input JSON.
    Expected input JSON format:
    {"data": [{feature1: value1, feature2: value2, ...}, {...}]}
    """
    global model
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        if "data" not in data:
            return {"error": "Input JSON must have a 'data' key"}

        df = pd.DataFrame(data["data"])
        preds = model.predict(df)
        return {"predictions": preds.tolist()}

    except Exception as e:
        return {"error": str(e)}
