import pandas as pd
import json
import requests

# Load new data
df = pd.read_csv("data/test.csv")    #use new data save it in data folder. but here since it is for testing - we use test.csv only.
input_json = json.dumps({"data": df.to_dict(orient="records")})

# Endpoint details
scoring_uri = "http://60eba419-8d24-4100-a6d5-23dddfb45a9b.centralindia.azurecontainer.io/score"
api_key = "mxi2zitW2vmwt2iQ16ZRRWasrcw77Ejk"

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
response = requests.post(scoring_uri, data=input_json, headers=headers)
predictions = response.json()

# Save predictions
pd.DataFrame(predictions["predictions"], columns=["Predicted_BPM"]).to_csv("outputs/predictions.csv", index=False)
print("Predictions saved to outputs/predictions.csv")
