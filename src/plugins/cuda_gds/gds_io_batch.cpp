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
#include <iostream>
#include "gds_io_batch.h"

nixlGdsIOBatch::nixlGdsIOBatch(unsigned int size)
{
    max_reqs            = size;
    io_batch_events     = new CUfileIOEvents_t[size];
    io_batch_params     = new CUfileIOParams_t[size];
    current_status      = NIXL_ERR_NOT_POSTED;
    entries_completed   = 0;
    batch_size          = 0;

    init_err = cuFileBatchIOSetUp(&batch_handle, size);
    if (init_err.err != 0) {
        std::cerr << "Error in creating the batch\n";
    }
}

nixlGdsIOBatch::~nixlGdsIOBatch()
{
    if (current_status == NIXL_SUCCESS ||
        current_status == NIXL_ERR_NOT_POSTED) {
            delete io_batch_events;
            delete io_batch_params;
        cuFileBatchIODestroy(batch_handle);
    } else {
            std::cerr<<"Attempting to delete a batch before completion\n";
    }
}

nixl_status_t nixlGdsIOBatch::addToBatch(CUfileHandle_t fh, void *buffer,
                                         size_t size, size_t file_offset,
                                         size_t ptr_offset,
                                         CUfileOpcode_t type)
{
    CUfileIOParams_t    *params = nullptr;

    if (batch_size >= max_reqs)
        return NIXL_ERR_BACKEND;

    params                          = &io_batch_params[batch_size];
    params->mode                    = CUFILE_BATCH;
    params->fh                      = fh;
    params->u.batch.devPtr_base     = buffer;
    params->u.batch.file_offset     = file_offset;
    params->u.batch.devPtr_offset   = ptr_offset;
    params->u.batch.size            = size;
    params->opcode                  = type;
    params->cookie                  = params;
    batch_size++;

    return NIXL_SUCCESS;
}

void nixlGdsIOBatch::destroyBatch()
{
    cuFileBatchIODestroy(batch_handle);
}

nixl_status_t nixlGdsIOBatch::cancelBatch()
{
    CUfileError_t   err;

    err = cuFileBatchIOCancel(batch_handle);
    if (err.err != 0) {
        std::cerr << "Error in canceling batch\n";
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsIOBatch::submitBatch(int flags)
{
    CUfileError_t   err;

    err = cuFileBatchIOSubmit(batch_handle, batch_size,
                              io_batch_params, flags);
    if (err.err != 0) {
        std::cerr << "Error in setting up Batch\n" << std::endl;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlGdsIOBatch::checkStatus()
{
    CUfileError_t       errBatch;
    unsigned int        nr = max_reqs;

    errBatch = cuFileBatchIOGetStatus(batch_handle, 0, &nr,
                                      io_batch_events, NULL);
    if (errBatch.err != 0) {
        std::cerr << "Error in IO Batch Get Status" << std::endl;
        current_status = NIXL_ERR_BACKEND;
    }

    entries_completed += nr;
    if (entries_completed < (unsigned int)max_reqs)
        current_status = NIXL_IN_PROG;
    else if (entries_completed > max_reqs)
        current_status = NIXL_ERR_UNKNOWN;
    else
        current_status = NIXL_SUCCESS;

    return current_status;
} 