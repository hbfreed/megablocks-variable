# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

"""MegaBlocks package setup.

The project metadata (name, version, dependencies, extras) lives in
pyproject.toml. This file only adds the optional CUDA extension.
"""

import os
import warnings

from setuptools import find_packages, setup

# We require torch in setup.py to build cpp extensions "ahead of time"
# More info here: # https://pytorch.org/tutorials/advanced/cpp_extension.html
try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'torch'. `torch` is required to install `MegaBlocks`.",) from e

cmdclass = {}
ext_modules = []
build_extensions = os.environ.get('MEGABLOCKS_BUILD_EXTENSIONS', '0') == '1'

# Build the training extensions only when the user requests them.
if build_extensions and 'cu' in torch.__version__ and CUDA_HOME is not None:

    cmdclass = {'build_ext': BuildExtension}
    nvcc_flags = ['--ptxas-options=-v', '--optimize=2']

    if os.environ.get('TORCH_CUDA_ARCH_LIST'):
        # Let PyTorch builder to choose device to target for.
        device_capability = ''
    else:
        device_capability_tuple = torch.cuda.get_device_capability()
        device_capability = f'{device_capability_tuple[0]}{device_capability_tuple[1]}'

    if device_capability:
        nvcc_flags.append(f'--generate-code=arch=compute_{device_capability},code=sm_{device_capability}',)

    ext_modules = [
        CUDAExtension(
            'nanomoe_ops',
            ['csrc/nanomoe_ops.cu'],
            include_dirs=['csrc'],
            extra_compile_args={
                'cxx': ['-fopenmp'],
                'nvcc': nvcc_flags,
            },
        ),
    ]
elif build_extensions and CUDA_HOME is None:
    warnings.warn(
        'Attempted to install CUDA extensions, but CUDA_HOME was None. ' +
        'Please install CUDA and ensure that the CUDA_HOME environment ' +
        'variable points to the installation location.',
    )
elif build_extensions:
    warnings.warn('Warning: No CUDA devices; cuda code will not be compiled.')

setup(
    packages=find_packages(exclude=['tests*', 'third_party*', 'exp*', '.github*']),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    package_data={'megablocks': ['py.typed']},
)
