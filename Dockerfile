# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE="none"

FROM $(BASE_IMAGE)

ARG DEFAULT_PYTHON_VERSION="3.12"

RUN yum install -y ninja-build cmake

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace/nixl

# LD_LIBRARY_PATH is needed for auditwheel to find libcuda.so.1
# Set incorrectly to `compat/lib` in cuda-dl-base image
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

ENV CUDA_PATH=/usr/local/cuda

ENV VIRTUAL_ENV=/workspace/nixl/.venv
RUN uv venv $VIRTUAL_ENV --python $DEFAULT_PYTHON_VERSION && \
    uv pip install --upgrade meson pybind11 patchelf

COPY . /workspace/nixl

RUN cp -a /opt/hpcx/ucx/* /usr/local

RUN rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=/usr/local/nixl \
    -Dcudapath_lib="/usr/local/cuda/lib64" \
    -Dcudapath_inc="/usr/local/cuda/include" && \
    cd build && \
    ninja && \
    ninja install

ENV LD_LIBRARY_PATH=/usr/local/nixl/lib64/:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/nixl/lib64/plugins:$LD_LIBRARY_PATH
ENV NIXL_PLUGIN_DIR=/usr/local/nixl/lib64/plugins

RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
    echo "/usr/local/nixl/lib/x86_64-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

# Create the wheel
ARG WHL_PYTHON_VERSIONS="3.12"
ARG WHL_PLATFORM="manylinux_2_27_x86_64"
RUN IFS=',' read -ra PYTHON_VERSIONS <<< "$WHL_PYTHON_VERSIONS" && \
    for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do \
        uv build --wheel --out-dir /tmp/dist --python $PYTHON_VERSION; \
    done
RUN uv pip install auditwheel && \
    uv run auditwheel repair /tmp/dist/nixl-*cp31*.whl --plat $WHL_PLATFORM --wheel-dir /workspace/nixl/dist

RUN uv pip install dist/nixl-*cp${DEFAULT_PYTHON_VERSION//./}*.whl
