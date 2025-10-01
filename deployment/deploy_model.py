from azureml.core import Workspace, Model, Environment
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import InferenceConfig

# Connect to workspace
ws = Workspace.from_config()

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
    print(f"API key: {service.get_keys()[0]}")
