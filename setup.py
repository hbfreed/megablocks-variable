# Copyright 2024 Databricks authors
# SPDX-License-Identifier: Apache-2.0

"""MegaBlocks build script.

Project metadata lives in ``pyproject.toml``. This file only finds the
packages and, when ``MEGABLOCKS_BUILD_EXTENSIONS=1``, compiles the legacy
``nanomoe_ops`` CUDA extension. That build needs torch in the build
environment; a default install does not.
"""

import os
import warnings

from setuptools import find_packages, setup

cmdclass = {}
ext_modules = []

if os.environ.get('MEGABLOCKS_BUILD_EXTENSIONS', '0') == '1':
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            'MEGABLOCKS_BUILD_EXTENSIONS=1 needs torch in the build environment. '
            'Build without isolation (pip install --no-build-isolation) or add torch '
            'as an extra build dependency (uv: [tool.uv.extra-build-dependencies]).',
        ) from e

    if 'cu' in torch.__version__ and CUDA_HOME is not None:
        cmdclass = {'build_ext': BuildExtension}
        nvcc_flags = ['--ptxas-options=-v', '--optimize=2']

        if not os.environ.get('TORCH_CUDA_ARCH_LIST'):
            # Otherwise the PyTorch builder chooses the target devices.
            major, minor = torch.cuda.get_device_capability()
            nvcc_flags.append(f'--generate-code=arch=compute_{major}{minor},code=sm_{major}{minor}')

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
    elif CUDA_HOME is None:
        warnings.warn(
            'Attempted to install CUDA extensions, but CUDA_HOME was None. ' +
            'Please install CUDA and ensure that the CUDA_HOME environment ' +
            'variable points to the installation location.',
        )
    else:
        warnings.warn('Warning: No CUDA devices; cuda code will not be compiled.')

setup(
    packages=find_packages(exclude=['tests*', '.github*']),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    package_data={'megablocks': ['py.typed']},
)
