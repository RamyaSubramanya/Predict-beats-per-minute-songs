# Step 1: Azure ML Environment Setup

from azureml.core import Workspace, Environment
import os
import json

# # Connect to workspace ---- below is for vscode to azureml deployment
# ws = Workspace.from_config()  # Ensure config.json exists or pass parameters

# Get Azure credentials from environment variable via github secrets 
# below is when you deploy from vscode on azureml via github 
azure_creds = os.environ["AZURE_CREDENTIALS"]
creds_dict = json.loads(azure_creds)
# Connect to workspace using credentials
ws = Workspace(
    subscription_id=creds_dict["subscriptionId"],
    resource_group=creds_dict["resourceGroup"],
    workspace_name=creds_dict["workspaceName"]
)
# Create Azure ML environment
env = Environment(name="bpm-env")
env.python.conda_dependencies.add_pip_package("pandas")
env.python.conda_dependencies.add_pip_package("numpy")
env.python.conda_dependencies.add_pip_package("scikit-learn")
env.python.conda_dependencies.add_pip_package("joblib")
env.python.conda_dependencies.add_pip_package("azureml-sdk")

# Register environment
env.register(workspace=ws)
print(f"Environment '{env.name}' registered successfully.")

# -----------------------------------------------------------------------------------------------
# Step 2: upload datasets onto AzureML

from azureml.core import Workspace, Datastore, Dataset

# Connect to workspace
datastore = ws.get_default_datastore()

# Upload local files into datastore
datastore.upload_files(
    files=['data/train.csv', 'data/test.csv'],
    target_path='bpm/',   # folder inside datastore
    overwrite=True,
    show_progress=True
)

# Create FileDataset (instead of TabularDataset)
train_dataset = Dataset.File.from_files(path=(datastore, 'bpm/train.csv'))
test_dataset  = Dataset.File.from_files(path=(datastore, 'bpm/test.csv'))

# Register them
train_dataset = train_dataset.register(
    workspace=ws,
    name='bpm_train_file',
    description='Training dataset for BPM prediction (FileDataset)',
    create_new_version=True
)

test_dataset = test_dataset.register(
    workspace=ws,
    name='bpm_test_file',
    description='Test dataset for BPM prediction (FileDataset)',
    create_new_version=True
)

print("FileDatasets uploaded and registered successfully!")

# -----------------------------------------------------------------------------------------------
# step3: Submit training job

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget

# Compute target
cluster_name = "bpm-cluster"
if cluster_name in ws.compute_targets:
    compute_target = ws.compute_targets[cluster_name]
    if compute_target and type(compute_target) is AmlCompute:
        print("Found existing compute cluster:", cluster_name)
else:
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_DS11_V2", max_nodes=2)
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Environment
env = Environment.from_conda_specification(
    name="bpm-env",
    file_path="environment.yml"
)

# Datasets (FileDataset for mount)
train_dataset = ws.datasets.get("bpm_train_file")
test_dataset  = ws.datasets.get("bpm_test_file")

# ScriptRunConfig
src = ScriptRunConfig(
    source_directory=".",
    script="main.py",
    compute_target=compute_target,
    environment=env,
    arguments=[
        "--train-data-path", train_dataset.as_mount(),
        "--test-data-path", test_dataset.as_mount()
    ]
)

# Submit experiment
experiment = Experiment(workspace=ws, name="bpm-training-exp")
run = experiment.submit(src)
run.wait_for_completion(show_output=True)

# -----------------------------------------------------------------------------------------------
# Step4: Register model
model = run.register_model(
    model_name="bpm_model",
    model_path="outputs/model.pkl",
    description="Linear Regression model predicting BPM"
)
print("Model registered:", model.name, model.version)
