/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cufile.h>
#include "common/nixl_status.h"

class nixlGdsIOBatch {
public:
    nixlGdsIOBatch(unsigned int size);
    ~nixlGdsIOBatch();

    nixl_status_t addToBatch(CUfileHandle_t fh, void *buffer,
                            size_t size, size_t file_offset,
                            size_t ptr_offset,
                            CUfileOpcode_t type);
    void destroyBatch();
    nixl_status_t cancelBatch();
    nixl_status_t submitBatch(int flags);
    nixl_status_t checkStatus();

private:
    CUfileBatchHandle_t batch_handle;
    CUfileIOEvents_t* io_batch_events;
    CUfileIOParams_t* io_batch_params;
    CUfileError_t init_err;
    nixl_status_t current_status;
    unsigned int entries_completed;
    unsigned int max_reqs;
    int batch_size;
}; 