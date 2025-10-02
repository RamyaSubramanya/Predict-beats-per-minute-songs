from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig
import os
import json

# # Connect to workspace for vs code to azure 
# ws = Workspace.from_config()

# Use Azure login via secrets in GitHub Actions
if "AZURE_CREDENTIALS" not in os.environ:
    raise Exception("AZURE_CREDENTIALS not set as GitHub Secret")

creds_dict = json.loads(os.environ["AZURE_CREDENTIALS"])

# Connect to Azure ML workspace
ws = Workspace(
    subscription_id=creds_dict["subscriptionId"],
    resource_group=creds_dict["resourceGroup"],
    workspace_name=creds_dict["workspaceName"]
)

# Get the registered model
model = Model(ws, name="bpm_model")

# Define environment for deployment
env = Environment.from_conda_specification(
    name="bpm_inference_env",
    file_path="deployment/inference_env.yml"
)

# Inference configuration
inference_config = InferenceConfig(
    source_directory="deployment",
    entry_script="score.py",
    environment=env
)

# Deployment configuration (ACI - Azure Container Instance)
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    auth_enabled=True,   # requires API key
    tags={"model": "bpm", "type": "linear_regression"},
    description="Predict BPM from song features"
)

# Deploy
service_name = "bpm-predict-service"
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config,
    overwrite=True
)
service.wait_for_deployment(show_output=True)
print(service.get_logs())

print(f"Service state: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")
if service.auth_enabled:
    api_key = service.get_keys()[0]
    print(f"API key: {api_key}")
else:
    api_key = None

# Save dynamic values
output = {
    "scoring_uri": service.scoring_uri,
    "api_key": api_key
}
with open("deployment/deployment_output.json", "w") as f:
    json.dump(output, f)

print("Deployment outputs saved to deployment/deployment_output.json")