# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
import random
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist


@pytest.fixture(autouse=True)
def clear_cuda_cache(request: pytest.FixtureRequest):
    """Clear memory between GPU tests."""
    marker = request.node.get_closest_marker('gpu')
    if marker is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Only gc on GPU tests as it 2x slows down CPU tests


@pytest.fixture(autouse=True)
def reset_mlflow_tracking_dir():
    """Reset MLFlow tracking dir so it doesn't persist across tests."""
    try:
        import mlflow
        mlflow.set_tracking_uri(None)  # type: ignore
    except ModuleNotFoundError:
        # MLFlow not installed
        pass


@pytest.fixture(scope='session')
def cleanup_dist():
    """Ensure all dist tests clean up resources properly."""
    yield
    # Avoid race condition where a test is still writing to a file on one rank
    # while the file system is being torn down on another rank.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@pytest.fixture(autouse=True, scope='session')
def configure_dist(request: pytest.FixtureRequest):
    # Configure dist globally when the world size is greater than 1,
    # so individual tests that do not use the trainer
    # do not need to worry about manually configuring dist.

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size == 1:
        return

    uses_gpu = any(item.get_closest_marker('gpu') is not None for item in request.session.items)
    backend = 'nccl' if uses_gpu else 'gloo'
    if uses_gpu:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=300))
    # Hold PyTest until all ranks have reached this barrier. Ensure that no rank starts
    # any test before other ranks are ready to start it, which could be a cause of random timeouts
    # (e.g. rank 1 starts the next test while rank 0 is finishing up the previous test).
    dist.barrier()


@pytest.fixture(autouse=True)
def set_log_levels():
    """Ensures all log levels are set to DEBUG."""
    logging.basicConfig()


@pytest.fixture(autouse=True)
def seed_all(rank_zero_seed: int):
    """Seed Python, NumPy, and PyTorch with a rank-local seed."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    seed = rank_zero_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def remove_run_name_env_var():
    # Remove environment variables for run names in unit tests
    run_name = os.environ.get('RUN_NAME')

    if 'RUN_NAME' in os.environ:
        del os.environ['RUN_NAME']

    yield

    if run_name is not None:
        os.environ['RUN_NAME'] = run_name
