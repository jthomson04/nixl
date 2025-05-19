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
#include <cuda.h>
#include <cuda/atomic>

#include "doca_backend.h"

#define ENABLE_DEBUG 0

__device__ uint32_t reserve_position(struct docaXferReqGpu *xferReqRing, uint32_t pos) {
	cuda::atomic_ref<uint32_t, cuda::thread_scope_device> index(*xferReqRing[pos].last_rsvd);
	return (index.fetch_add(xferReqRing[pos].num, cuda::std::memory_order_relaxed) & 0xFFFF);
}

__device__ void wait_post(struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos) {
	cuda::atomic_ref<uint32_t, cuda::thread_scope_device> index(*xferReqRing[pos].last_posted);
	while (index.load(cuda::std::memory_order_relaxed) != pos)
		continue;
	// prevents the compiler from reordering
	asm volatile("fence.acquire.gpu;");
	doca_gpu_dev_rdma_commit_weak(rdma_gpu, 0, xferReqRing[pos].num);
	asm volatile("fence.release.gpu;");
	index.store((pos + 1) & DOCA_XFER_REQ_MASK, cuda::std::memory_order_relaxed);
}

__global__ void kernel_read(struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
	doca_error_t result;
	struct doca_gpu_buf *lbuf;
	struct doca_gpu_buf *rbuf;
	__shared__ uint32_t base_position;

	//Warmup
	if (xferReqRing == nullptr)
		return;

	if (threadIdx.x == 0)
		base_position = reserve_position(xferReqRing, pos);
	__syncthreads();

	if (threadIdx.x < xferReqRing[pos].num) {

		doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].larr[threadIdx.x], 0, &lbuf);
		doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].rarr[threadIdx.x], 0, &rbuf);

#if ENABLE_DEBUG == 1
		printf(">>>>>>> CUDA rdma read kernel thread %d pos %d descr %d size %d\n",
		        threadIdx.x, pos, (base_position + threadIdx.x) & 0xFFFF, (int)xferReqRing[pos].size[threadIdx.x]);
#endif
		result = doca_gpu_dev_rdma_read_weak(rdma_gpu, 0, rbuf, 0, lbuf, 0, xferReqRing[pos].size[threadIdx.x], 0, (base_position + threadIdx.x) & 0xFFFF);
		if (result != DOCA_SUCCESS)
			printf("Error %d doca_gpu_dev_rdma_write_strong\n", result);
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		wait_post(rdma_gpu, xferReqRing, pos);

#if ENABLE_DEBUG == 1
		printf(">>>>>>> CUDA rdma read kernel pos %d posted %d buffers\n", pos, xferReqRing[pos].num);
#endif
		xferReqRing[pos].in_use = 0;
	}
}

__global__ void kernel_write(struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
	doca_error_t result;
	struct doca_gpu_buf *lbuf;
	struct doca_gpu_buf *rbuf;
	__shared__ uint32_t base_position;

	//Warmup
	if (xferReqRing == nullptr)
		return;

	if (threadIdx.x == 0)
		base_position = reserve_position(xferReqRing, pos);
	__syncthreads();

	if (threadIdx.x < xferReqRing[pos].num) {

		doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].larr[threadIdx.x], 0, &lbuf);
		doca_gpu_dev_buf_get_buf((struct doca_gpu_buf_arr *)xferReqRing[pos].rarr[threadIdx.x], 0, &rbuf);

#if ENABLE_DEBUG == 1
		printf(">>>>>>> CUDA rdma write kernel thread %d pos %d descr %d size %d\n",
		        threadIdx.x, pos, (base_position + threadIdx.x) & 0xFFFF, (int)xferReqRing[pos].size[threadIdx.x]);
#endif
		result = doca_gpu_dev_rdma_write_weak(rdma_gpu, 0, rbuf, 0, lbuf, 0, xferReqRing[pos].size[threadIdx.x], 0, DOCA_GPU_RDMA_WRITE_FLAG_NONE, (base_position + threadIdx.x) & 0xFFFF);
		if (result != DOCA_SUCCESS)
			printf("Error %d doca_gpu_dev_rdma_write_strong\n", result);
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		wait_post(rdma_gpu, xferReqRing, pos);

#if ENABLE_DEBUG == 1
		printf(">>>>>>> CUDA rdma write kernel pos %d posted %d buffers\n", pos, xferReqRing[pos].num);
#endif
		xferReqRing[pos].in_use = 0;
	}
}

__global__ void kernel_wait(struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
	doca_error_t result;
	uint32_t num_ops=0;

	//Warmup
	if (xferReqRing == nullptr)
		return;

	result = doca_gpu_dev_rdma_wait_all(rdma_gpu, &num_ops);
	if (result != DOCA_SUCCESS) {
		xferReqRing[pos].in_use = 2;
		printf("Error %d doca_gpu_dev_rdma_wait_all\n", result);
	}

	xferReqRing[pos].in_use = 0;
}

extern "C" {

doca_error_t doca_kernel_write(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
{
	cudaError_t result = cudaSuccess;

	if (rdma_gpu == NULL) {
		fprintf(stderr, "kernel_write_server invalid input values");
		return DOCA_ERROR_INVALID_VALUE;
	}

	/* Check no previous CUDA errors */
	result = cudaGetLastError();
	if (result != cudaSuccess) {
		fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	kernel_write<<<1, DOCA_XFER_REQ_SIZE, 0, stream>>>(rdma_gpu, xferReqRing, pos);
	result = cudaGetLastError();
	if (result != cudaSuccess) {
		fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
		return DOCA_ERROR_BAD_STATE;
	}

	return DOCA_SUCCESS;
}

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

doca_error_t doca_kernel_wait(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos)
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

    kernel_wait<<<1, 1, 0, stream>>>(rdma_gpu, xferReqRing, pos);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} /* extern C */
