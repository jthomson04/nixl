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
#include "hf3fs_utills.h"


nixl_status_t hf3fsUtil::registerFileHandle(int fd)
{
    int ret = hf3fs_reg_fd(fd, 0);
    if (ret < 0) {
        std::cerr << "file register error:"
                  << std::endl;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::openHf3fsDriver()
{
    return NIXL_SUCCESS;
}

void hf3fsUtil::closeHf3fsDriver()
{
    // nothing to do
}

void hf3fsUtil::deregisterFileHandle(int fd)
{
    hf3fs_dereg_fd(fd);
}

nixl_status_t hf3fsUtil::createIOV(struct hf3fs_iov *iov, int num_ios, size_t block_size)
{
    /* int hf3fs_iovcreate(struct hf3fs_iov *iov,
                    const char *hf3fs_mount_point,
                    size_t size,
                    size_t block_size,
                    int numa); */
    // TODO: numa?
    auto ret = hf3fs_iovcreate(iov, this->mount_point.c_str(), num_ios, block_size, 0);
    if (ret < 0) {
        std::cerr << "hf3fs create iov error:"
                  << std::endl;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::createIOR(struct hf3fs_ior *ior, int num_ios, bool is_read)
{

    /*int hf3fs_iorcreate(struct hf3fs_ior *ior,
                    const char *hf3fs_mount_point,
                    int entries,
                    bool for_read,
                    int io_depth,
                    int numa);*/
    // TODO: numa?, io_depth?
    // TODO: use iorcreate2/3/4?
    auto ret = hf3fs_iorcreate(ior, this->mount_point.c_str(), num_ios, is_read, num_ios, 0);
    if (ret < 0) {
        std::cerr << "hf3fs create ior error:"
                  << std::endl;
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}

// TODO: better name?
nixl_status_t hf3fsUtil::prepIO(struct hf3fs_ior *ior, struct hf3fs_iov *iov, void *addr, size_t fd_offset, size_t size, int fd, bool is_read)
{
    // TODO: userdata?
    // missleadinm prep might send the io
    auto ret = hf3fs_prep_io(ior, iov, is_read, addr, fd, fd_offset, size, nullptr);
    if (ret < 0) {
        std::cerr << "hf3fs prep io error:"
                << std::endl;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}


nixl_status_t hf3fsUtil::postIOR( struct hf3fs_ior *ior)
{
    auto ret = hf3fs_submit_ios(ior);
    if (ret < 0) {
        std::cerr << "hf3fs submit ios error:"
                  << std::endl;
        return NIXL_ERR_BACKEND;
    }
    return NIXL_SUCCESS;
}

// TODO: probably need to chabge
nixl_status_t hf3fsUtil::checkXfer(struct hf3fs_ior *ior)
{
    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::destroyIOR(struct hf3fs_ior *ior)
{
    hf3fs_iordestroy(ior);
    return NIXL_SUCCESS;
}

nixl_status_t hf3fsUtil::destroyIOV(struct hf3fs_iov *iov)
{
    hf3fs_iovdestroy(iov);
    return NIXL_SUCCESS;
}
