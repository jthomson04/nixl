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
#include <cassert>
#include <iostream>
//#include "hf3fs_usrbio.h"
#include "hf3fs_backend.h"
#include "common/str_tools.h"


nixlHf3fsEngine::nixlHf3fsEngine (const nixlBackendInitParams* init_params)
    : nixlBackendEngine (init_params)
{
    hf3fs_utils = new hf3fsUtil();

    this->initErr = false;
    if (hf3fs_utils->openHf3fsDriver() == NIXL_ERR_BACKEND)
        this->initErr = true;

    // TODO handle mount point
    std::string mount_point = init_params->customParams->at("mount_point");
    char mount_point_cstr[256];
    auto ret = hf3fs_extract_mount_point(mount_point_cstr, 256, mount_point.c_str());
    if (ret < 0) {
        std::cerr << "Error in extracting mount point\n";
        this->initErr = true;
    }
    hf3fs_utils->mount_point = mount_point_cstr;

}


nixl_status_t nixlHf3fsEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    nixl_status_t status;
    nixlHf3fsMetadata *md = new nixlHf3fsMetadata();
    if (nixl_mem != FILE_SEG) {
        return NIXL_ERR_BACKEND;
    }
    // if the same file is reused - no need to re-register
    int fd = mem.devId; // TODO: Check if need to open fd?
    auto it = hf3fs_file_map.find(fd);
    if (it != hf3fs_file_map.end()) {
    // should we update the size? and metadata
        md->handle = it->second;
        md->handle.size = mem.len;
        md->handle.metadata = mem.metaInfo;
        md->type = nixl_mem;
        status = NIXL_SUCCESS;
    } else {
        // ADD new file handle
        auto status = hf3fs_utils->registerFileHandle(fd);
        if (status != NIXL_SUCCESS) {
            delete md;
            return status;
        }

        md->handle.fd = fd;
        md->handle.size = mem.len;
        md->handle.metadata = mem.metaInfo;
        md->type = nixl_mem;
        hf3fs_file_map[fd] = md->handle;
        status = NIXL_SUCCESS;
    }


    out = (nixlBackendMD*) md;
    // set value for gds handle here.
    return status;
}

nixl_status_t nixlHf3fsEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlHf3fsMetadata *md = (nixlHf3fsMetadata *)meta;
    if (md->type == FILE_SEG) {
        hf3fs_utils->deregisterFileHandle(md->handle.fd);
        // TODO close fd?
    } else {
        // Unsupported in the backend.
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t nixlHf3fsEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    return NIXL_SUCCESS;
}

nixl_status_t nixlHf3fsEngine::postXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args)
{
    // TODO: Determine the batches and prepare most of the handle
    void                *addr = NULL;
    //int                 fd = 0;
    size_t              size = 0;
    size_t              offset = 0;
    size_t              total_size = 0;
    //int                 rc = 0;
    int              buf_cnt  = local.descCount();
    int              file_cnt = remote.descCount();
    //nixl_status_t       ret = NIXL_ERR_NOT_POSTED;
    hf3fsFileHandle       fh;
    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;

    if ((remote.getType() != FILE_SEG) && (local.getType() != FILE_SEG)) {
        std::cerr <<"Only support I/O between VRAM to file type\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    if ((buf_cnt != file_cnt) ||
            ((operation != NIXL_READ) && (operation != NIXL_WRITE)))  {
        std::cerr <<"Error in count or operation selection\n";
        return NIXL_ERR_INVALID_PARAM;
    }

    bool is_read = operation == NIXL_READ;

    auto status = hf3fs_utils->createIOR(&hf3fs_handle->ior, file_cnt, is_read);
    for (int i = 0; i < file_cnt; i++) {
        addr = (void *) remote[i].addr;
        size = remote[i].len;
        offset = (size_t) local[i].addr;

        auto it = hf3fs_file_map.find(local[i].devId);
        if (it != hf3fs_file_map.end()) {
            fh = it->second;
        } else {
            return NIXL_ERR_NOT_FOUND;
        }
        // TODO: move IOV creation to prepXfer
        nixlHf3fsIO *io = new nixlHf3fsIO();
        status = hf3fs_utils->createIOV(&io->iov, 1, size);
            if (status != NIXL_SUCCESS) {
                return status;
                // TODO: cleanup
            }

        status = hf3fs_utils->prepIO(&hf3fs_handle->ior, &io->iov, addr, offset, size, fh.fd, is_read);
        total_size += size;

        io->fd = fh.fd;
        hf3fs_handle->io_list.push_back(io);
    }

    status = hf3fs_utils->postIOR(&hf3fs_handle->ior);
    if (status != NIXL_SUCCESS) {
        return status;
    }
    return NIXL_SUCCESS;

    // TODO:
}

nixl_status_t nixlHf3fsEngine::checkXfer(nixlBackendReqH* handle)
{
    // TODO: check if the ior is completed
    return NIXL_SUCCESS;
}

nixl_status_t nixlHf3fsEngine::releaseReqH(nixlBackendReqH* handle)
{
    nixlHf3fsBackendReqH *hf3fs_handle = (nixlHf3fsBackendReqH *) handle;
    for (auto io : hf3fs_handle->io_list) {
        hf3fs_dereg_fd(io->fd);
        hf3fs_utils->destroyIOV(&io->iov);
        delete io;
    }

    hf3fs_utils->destroyIOR(&hf3fs_handle->ior);
    delete hf3fs_handle;
    return NIXL_SUCCESS;
}

nixlHf3fsEngine::~nixlHf3fsEngine() {
    hf3fs_utils->closeHf3fsDriver();
    delete hf3fs_utils;
}
