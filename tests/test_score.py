import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deployment.score import init, run
import json
import os
from src.modelling import model_and_evaluate, predict_test_data  # optional for generating a local model
import joblib
import pandas as pd

# -------------------
# Step 0: Initialize local model
# -------------------
MODEL_PATH = "outputs/model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Local model not found at {MODEL_PATH}. Run training first!")

init(model_path=MODEL_PATH)

# -------------------
# Step 1: Prepare input JSON for testing
# -------------------
test_df = pd.read_csv("data/test.csv").iloc[:2, 1:] # Take first 2 rows and second column onwards
input_json = json.dumps({"data": test_df.to_dict(orient="records")})

# -------------------
# Step 2: Run predictions
# -------------------
output = run(input_json)
print("Prediction output:", output)