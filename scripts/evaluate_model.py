# scripts/evaluate_model.py
import argparse
import json
import sys
import os

def main(metrics_file, deployment_file):
    # Read new metrics from latest training
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        sys.exit(1)

    with open(metrics_file, "r") as f:
        new_metrics = json.load(f)

    new_mae = new_metrics.get("mae")
    new_mape = new_metrics.get("mape")

    if new_mae is None or new_mape is None:
        print("Metrics file missing 'mae' or 'mape'")
        sys.exit(1)

    # Read current deployed metrics
    if os.path.exists(deployment_file):
        with open(deployment_file, "r") as f:
            deployment_data = json.load(f)
        current_mae = deployment_data.get("current_mae", float("inf"))
    else:
        current_mae = float("inf")  # First deployment

    print(f"Current MAE: {current_mae}")
    print(f"New MAE: {new_mae}")

    # Output for GitHub Actions workflow
    # Use set-output for workflow if needed
    print(f"::set-output name=new_mae::{new_mae}")
    print(f"::set-output name=new_mape::{new_mape}")

    # Optional: Save to JSON for workflow to read easily
    output = {
        "new_mae": new_mae,
        "new_mape": new_mape,
        "current_mae": current_mae
    }
    os.makedirs("deployment", exist_ok=True)
    with open("deployment/evaluate_metrics.json", "w") as f:
        json.dump(output, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_file", type=str, default="outputs/metrics.json", help="Path to latest training metrics")
    parser.add_argument("--deployment_file", type=str, default="deployment/deployment_output.json", help="Path to current deployment metrics")
    args = parser.parse_args()
    main(args.metrics_file, args.deployment_file)
