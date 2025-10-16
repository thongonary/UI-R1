#!/usr/bin/env python3
"""
Minimal AML job submission wrapper for ppt_train_e_custom.sh
"""
from azure.ai.ml import MLClient, command, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Configure your AML workspace
SUBSCRIPTION_ID = "67aa06b0-2686-40dc-92b0-1316ea0304d9"
RESOURCE_GROUP = "oxoml"
WORKSPACE_NAME = "oaiscience-scus"

# Job configuration
COMPUTE_NAME = "gpu-cluster"
ENVIRONMENT_NAME = "grpo-finetuning-env:1.0.17"  # or use existing environment
EXPERIMENT_NAME = "ppt-font-grounding-training-grpo"

# Create ML client
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME
)

# Create job
job = command(
    code="../..",  # Upload from repository root to include data/
    command="cd src/scripts; bash ppt_train_e_custom.sh",
    environment=ENVIRONMENT_NAME,
    compute=COMPUTE_NAME,
    experiment_name=EXPERIMENT_NAME,
    display_name="ppt-grpo-training",
    instance_count=2,
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 2
    },
    outputs={
        "checkpoints": Output(type="uri_folder", mode="rw_mount")
    },
    environment_variables={
        "SAVE_PATH": "${{outputs.checkpoints}}"
    }
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")
