import pandas as pd
import json
import requests
import os
import argparse

# Get endpoint and API key from environment variables
SCORING_URI = os.environ.get("SCORING_URI")
API_KEY = os.environ.get("API_KEY")

if not SCORING_URI or not API_KEY:
    raise Exception("Please set SCORING_URI and API_KEY as environment variables")

# Parse input CSV
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()

data = pd.read_csv(args.input)
payload = {"data": data.to_dict(orient="records")}

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

response = requests.post(SCORING_URI, headers=headers, data=json.dumps(payload))
predictions = response.json()

if "predictions" in predictions:
    pd.DataFrame(predictions["predictions"], columns=["Predicted_BPM"]).to_csv("outputs/predictions.csv", index=False)
    print("Predictions saved to outputs/predictions.csv")
else:
    print("Error from endpoint:", predictions)
