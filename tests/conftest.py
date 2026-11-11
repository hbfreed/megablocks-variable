# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import random

import numpy as np
import pytest
import torch

# Enforce deterministic mode before any tests start.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.use_deterministic_algorithms(True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--seed', default=0, type=int, help='Seed set before each test.')


@pytest.fixture(autouse=True)
def clear_cuda_cache(request: pytest.FixtureRequest):
    """Clear memory between GPU tests."""
    marker = request.node.get_closest_marker('gpu')
    if marker is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()  # Only gc on GPU tests as it 2x slows down CPU tests


@pytest.fixture(autouse=True)
def seed_all(pytestconfig: pytest.Config):
    """Seed Python, NumPy, and PyTorch."""
    seed = pytestconfig.getoption('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    if exitstatus == 5:
        session.exitstatus = 0  # Ignore no-test-ran errors
