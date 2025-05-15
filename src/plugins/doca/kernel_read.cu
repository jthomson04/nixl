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

#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_rdma.cuh>

#include "doca_backend.h"

__global__ void kernel_read(struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
    doca_error_t result;
	struct doca_gpu_buf *lbuf;
	struct doca_gpu_buf *rbuf;
    const int connection_index = 0;
    uint32_t num_ops=0;
    uint32_t curr_position;
	uint32_t mask_max_position;

    //Warmup
    if (xferReqRing == nullptr)
        return;

    if (threadIdx.x < xferReqRing[pos].num) {
        doca_gpu_dev_rdma_get_info(rdma_gpu, 0, &curr_position, &mask_max_position);
        doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].larr[threadIdx.x], 0, &lbuf);
        doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].rarr[threadIdx.x], 0, &rbuf);

        // printf(">>>>>>> CUDA rdma read kernel thread %d pos %d wqe %d size %d\n",
        //         threadIdx.x, pos, (curr_position + threadIdx.x) & mask_max_position, (int)xferReqRing[pos].size[threadIdx.x]);

        result = doca_gpu_dev_rdma_read_weak(rdma_gpu, connection_index, rbuf, 0, lbuf, 0, xferReqRing[pos].size[threadIdx.x], 0, (curr_position + threadIdx.x) & mask_max_position);
        if (result != DOCA_SUCCESS)
            printf("Error %d doca_gpu_dev_rdma_write_strong\n", result);

    }
    __syncthreads();

    if (threadIdx.x == 0) {
        doca_gpu_dev_rdma_commit_weak(rdma_gpu, 0, xferReqRing[pos].num);
        result = doca_gpu_dev_rdma_wait_all(rdma_gpu, &num_ops);
        if (result != DOCA_SUCCESS)
            printf("Error %d doca_gpu_dev_rdma_wait_all\n", result);

        // printf(">>>>>>> CUDA rdma write kernel pos %d num %d completed %d ops\n",
        //                  pos, xferReqRing[pos].num, num_ops);

        xferReqRing[pos].num = 0;
    }
}

extern "C" {

doca_error_t doca_kernel_read(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
    cudaError_t result = cudaSuccess;

    if (rdma_gpu == NULL) {
        fprintf(stderr, "kernel_read_server invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    kernel_read<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(rdma_gpu, xferReqRing, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} /* extern C */
