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

# Define source files
gds_sources = [
    'gds_utils.cpp',
    'gds_backend.cpp',
    'gds_io_batch.cpp',
    'gds_plugin.cpp',
]

gds_headers = [
    'gds_utils.h',
    'gds_backend.h',
    'gds_io_batch.h',
]

# Configure GDS dependency
gds_path = get_option('gds_path')
cufile_dep = []
if gds_path != ''
    gds_lib_path = gds_path / 'lib'
    gds_inc_path = gds_path / 'include'
    cufile_dep = declare_dependency(
        link_args : ['-L' + gds_lib_path, '-lcufile'],
        include_directories : include_directories(gds_inc_path)
    )
endif

# Common library configuration
gds_lib_deps = [
    nixl_infra,
    cuda_dep,
    cufile_dep,
]

gds_lib_includes = [
    nixl_inc_dirs,
    utils_inc_dirs,
]

# Build either static or shared library based on configuration
if 'GDS' in static_plugins
    gds_backend_lib = static_library('GDS',
        gds_sources + gds_headers,
        dependencies: gds_lib_deps,
        include_directories: gds_lib_includes,
        install: false,
        cpp_args : compile_flags,
        name_prefix: 'libplugin_'
    )
else
    gds_backend_lib = shared_library('GDS',
        gds_sources + gds_headers,
        dependencies: gds_lib_deps,
        include_directories: gds_lib_includes,
        install: true,
        cpp_args : ['-fPIC'],
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir
    )

    # Add plugin to debug build list if needed
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "GDS=' + gds_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

# Export library interface
gds_backend_interface = declare_dependency(link_with: gds_backend_lib)
