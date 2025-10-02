import argparse
import requests
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--endpoint", type=str, required=True)
parser.add_argument("--api_key", type=str, required=True)
args = parser.parse_args()

# Load test data
df = pd.read_csv(args.input)

# Call endpoint
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
response = requests.post(args.endpoint, json={"data": df.to_dict(orient="records")}, headers=headers)

print("Response:", response.json())
