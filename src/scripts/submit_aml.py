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
COMPUTE_NAME = "gpu-cluster-network"
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
    # Use AML-aware script variant that does NOT invoke torch.distributed.run manually.
    command="cd src/scripts; bash ppt_train_e_custom_aml.sh",
    environment=ENVIRONMENT_NAME,
    compute=COMPUTE_NAME,
    experiment_name=EXPERIMENT_NAME,
    display_name="ppt-grpo-training",
    # Number of nodes
    instance_count=2,
    # Let AML PyTorch launcher spawn processes (no manual torchrun needed in script)
    distribution={
        "type": "PyTorch",
        # Set to number of GPUs per node you want to utilize.
        # Ensure this matches actual GPU count or desired subset.
        "process_count_per_instance": 2
    },
    outputs={
        # Mounted output; SAVE_PATH inside script will be this path (overrides default)
        "checkpoints": Output(type="uri_folder", mode="rw_mount")
    },
    environment_variables={
        # Provided so script can detect externally set SAVE_PATH and not overwrite it.
        "SAVE_PATH": "${{outputs.checkpoints}}",
        "DEBUG_MODE": "true",
        # NCCL/DeepSpeed diagnostics & stability
        "NCCL_DEBUG": "INFO",  # verbose NCCL logs
        "NCCL_ASYNC_ERROR_HANDLING": "1",  # allow async error reporting
        "NCCL_SOCKET_IFNAME": "eth0",  # explicit network interface
        "NCCL_COLLNET_ENABLE": "0",  # disable collnet for simpler topology
        "NCCL_NET_GDR_LEVEL": "2",  # enable GPU Direct RDMA optimizations if available
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # mitigate certain hang scenarios
        "NCCL_IB_DISABLE": "0",  # ensure IB enabled (should already be, explicit for clarity)
        "NCCL_TOPO_FILE": "",  # prevent attempts to read missing topology file
        # Shorten default NCCL timeout for quicker failure (optional):
        "NCCL_BLOCKING_WAIT": "1"
    }
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")
