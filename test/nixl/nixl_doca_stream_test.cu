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
#include <string>
#include <algorithm>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <nixl.h>
#include <cassert>
#include "stream/metadata_stream.h"
#include "serdes/serdes.h"

#define CUDA_THREADS 512
#define NUM_STREAMS 8
#define TRANSFER_NUM_BUFFER 8
#define TRANSFER_NUM 2
#define BUFFER_TOTAL (TRANSFER_NUM * TRANSFER_NUM_BUFFER)
#define SIZE 1024
#define INITIATOR_VALUE 0xbb
#define VOLATILE(x) (*(volatile typeof(x) *)&(x))
#define INITIATOR_THRESHOLD_NS 50000 //50us
#define USE_NVTX 1

#if USE_NVTX
#include <nvtx3/nvToolsExt.h>

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define DEVICE_GET_TIME(globaltimer) asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer))

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

static void checkCudaError(cudaError_t result, const char *message) {
	if (result != cudaSuccess) {
		std::cerr << message << " (Error code: " << result << " - "
				   << cudaGetErrorString(result) << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

__global__ void target_kernel(uintptr_t addr, int transfer_idx)
{
    uint8_t ok = 1;
    uintptr_t transfer_addr = addr + (threadIdx.x * SIZE);

    printf(">>>>>>> CUDA target waiting on transfer %d addr %lx size %d\n",
            transfer_idx, transfer_addr, (uint32_t)SIZE);

    while(VOLATILE(((uint8_t*)transfer_addr)[0]) == 0);

    for (int i = 0; i < (int)SIZE; i++) {
        if (((uint8_t*)transfer_addr)[i] != INITIATOR_VALUE) {
            printf(">>>>>>> CUDA target byte %x is wrong\n", i);
            ok = 1;
        }
    }
    if (ok == 1)
        printf(">>>>>>> CUDA target, all bytes received!\n");
    else
        printf(">>>>>>> CUDA target, not all received bytes are ok!\n");
}

int launch_target_wait_kernel(cudaStream_t stream, uintptr_t addr, int transfer_idx)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    target_kernel<<<1, TRANSFER_NUM, 0, stream>>>(addr, transfer_idx);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

__global__ void initiator_kernel(uintptr_t addr, uint32_t transfer_idx)
{
    unsigned long long start, end;
    // Each block updates a buffer in this transfer
    uintptr_t block_address = (addr + (blockIdx.x * SIZE));

    /* Simulate a longer CUDA kernel to process initiator data */
    DEVICE_GET_TIME(start);

    for (int i = threadIdx.x; i < SIZE; i+=blockDim.x)
        ((uint8_t*)block_address)[i] = INITIATOR_VALUE;

    __syncthreads();

    do {
        DEVICE_GET_TIME(end);
    } while (end - start < INITIATOR_THRESHOLD_NS);
}

int launch_initiator_send_kernel(cudaStream_t stream, uintptr_t addr, uint32_t transfer_idx)
{
    cudaError_t result = cudaSuccess;

    /* Check no previous CUDA errors */
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    // Block = # buffers x transfer
    initiator_kernel<<<TRANSFER_NUM_BUFFER, CUDA_THREADS, 0, stream>>>(addr, transfer_idx);
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        fprintf(stderr, "[%s:%d] cuda failed with %s", __FILE__, __LINE__, cudaGetErrorString(result));
        return -1;
    }

    return 0;
}

/**
 * This test does p2p from using PUT.
 * intitator -> target so the metadata and
 * desc list needs to move from
 * target to initiator
 */

bool allBytesAre(void* buffer, size_t size, uint8_t value) {
    uint8_t* byte_buffer = static_cast<uint8_t*>(buffer); // Cast void* to uint8_t*
    // Iterate over each byte in the buffer
    for (size_t i = 0; i < size; ++i) {
        if (byte_buffer[i] != value) {
            return false; // Return false if any byte doesn't match the value
        }
    }
    return true; // All bytes match the value
}

std::string recvFromTarget(int port) {
    nixlMDStreamListener listener(port);
    listener.setupListenerSync();
    listener.acceptClient();
    return listener.recvFromClient();
}

void sendToInitiator(const char *ip, int port, std::string data) {
    nixlMDStreamClient client(ip, port);
    client.connectListenerSync();
    client.sendData(data);
}

int main(int argc, char *argv[]) {
    int                     initiator_port;
    nixl_status_t           ret = NIXL_SUCCESS;
    uint8_t                 *data_address;
    std::string             role;
    std::string             processing;
    const char              *initiator_ip;
    nixl_blob_t             remote_desc;
    nixl_blob_t             metadata;
    nixl_blob_t             remote_metadata;
    int                     status = 0;

    /** NIXL declarations */
    /** Agent and backend creation parameters */
    nixlAgentConfig cfg(true);
    nixl_b_params_t params;
    nixlBlobDesc    buf[TRANSFER_NUM_BUFFER];
    nixlBackendH    *doca;
    cudaStream_t    stream[NUM_STREAMS];
    /** Serialization/Deserialization object to create a blob */
    nixlSerDes *serdes[TRANSFER_NUM];
    nixlSerDes *remote_serdes[TRANSFER_NUM];
    std::string target_name;

    /** Descriptors and Transfer Request */
    nixl_reg_dlist_t  *dram_for_doca[TRANSFER_NUM];
    nixlXferReqH      *treq[TRANSFER_NUM];

    /** Argument Parsing */
    if (argc < 5) {
        std::cout <<"Enter the required arguments\n" << std::endl;
        std::cout <<"<Role> <Peer IP> <Peer Port> <CPU or GPU processing>"
                  << std::endl;
        exit(-1);
    }

    role = std::string(argv[1]);
    std::transform(role.begin(), role.end(), role.begin(), ::tolower);
    if (!role.compare("initiator") && !role.compare("target")) {
            std::cerr << "Invalid role. Use 'initiator' or 'target'."
                      << "Currently "<< role <<std::endl;
            return 1;
    }

    initiator_ip   = argv[2];
    initiator_port = std::stoi(argv[3]);
    processing = std::string(argv[4]);
    std::transform(processing.begin(), processing.end(), processing.begin(), ::tolower);
    if (!processing.compare("cpu") && !processing.compare("gpu")) {
            std::cerr << "Invalid type of processing. Use 'cpu' or 'gpu'."
                      << "Currently "<< processing <<std::endl;
            return 1;
    }

    /*** End - Argument Parsing */

    checkCudaError(cudaSetDevice(0), "Failed to set device");
	cudaFree(0);

    if (processing.compare("gpu") == 0)
        for (int i = 0; i < NUM_STREAMS; i++)
            checkCudaError(cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking), "Failed to create CUDA stream");

    /** Common to both Initiator and Target */
    std::cout << "Starting Agent for "<< role << "\n";
    nixlAgent     agent(role, cfg);
    params["network_devices"] = "mlx5_0";
	params["gpu_devices"] = "0";
    PUSH_RANGE("createBackend", 0)
    agent.createBackend("DOCA", params, doca);
    POP_RANGE

    nixl_opt_args_t extra_params;
    extra_params.backends.push_back(doca);

    checkCudaError(cudaMalloc(&data_address, SIZE * BUFFER_TOTAL), "Failed to allocate CUDA buffer 0");
    checkCudaError(cudaMemset((void*)data_address, 0, SIZE * BUFFER_TOTAL), "Failed to memset CUDA buffer 0");

    if (role != "target") {
        std::cout << "Allocating for initiator : "
                  << BUFFER_TOTAL << " buffers "
                  << SIZE << " Bytes each "
                  << (void*)data_address << " address "
                  << std::endl;
    } else {
        std::cout << "Allocating for target : "
                  << BUFFER_TOTAL << " buffers "
                  << SIZE << " Bytes each "
                  << (void*)data_address << " address "
                  << std::endl;
    }

    for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
        dram_for_doca[transfer_idx] = new nixl_reg_dlist_t(DRAM_SEG); //, false, TRANSFER_NUM_BUFFER);
        for (int transfer_buffer_idx = 0; transfer_buffer_idx < TRANSFER_NUM_BUFFER; transfer_buffer_idx++) {
            buf[transfer_buffer_idx].addr  = (uintptr_t)(data_address + (transfer_idx * TRANSFER_NUM_BUFFER * SIZE) + (transfer_buffer_idx * SIZE));
            buf[transfer_buffer_idx].len   = SIZE;
            buf[transfer_buffer_idx].devId = 0;
            printf("Register transf %d buf %d addr %lx\n",  transfer_idx, transfer_buffer_idx, buf[transfer_buffer_idx].addr);
            dram_for_doca[transfer_idx]->addDesc(buf[transfer_buffer_idx]);
        }

        /** Register memory in both initiator and target */
        agent.registerMem(dram_for_doca[transfer_idx][0], &extra_params);
    }

    agent.getLocalMD(metadata);

    std::cout << " Start Control Path metadata exchanges \n";
    if (role == "target") {
        nixlMDStreamClient client(initiator_ip, initiator_port);
        client.connectListenerSync();

        std::cout << " Desc List from Target to Initiator\n";
        for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
            dram_for_doca[transfer_idx]->print();
            /** Sending both metadata strings together */
            serdes[transfer_idx] = new nixlSerDes();
            assert(serdes[transfer_idx]->addStr("AgentMD", metadata) == NIXL_SUCCESS);
            assert(dram_for_doca[transfer_idx]->trim().serialize(serdes[transfer_idx]) == NIXL_SUCCESS);

            std::cout << " Serialize Metadata to string and Send to Initiator\n";
            std::cout << " \t -- To be handled by runtime - currently sent via a TCP Stream\n";
            // sendToInitiator(initiator_ip, initiator_port, serdes[transfer_idx]->exportStr());
            client.sendData(serdes[transfer_idx]->exportStr());
            std::cout << " End Control Path metadata exchanges \n";
        }

        std::cout << " Start Data Path Exchanges \n";
        std::cout << " Waiting to receive Data from Initiator\n";

        /* 1 target CUDA kernel per transfer. Each thread will check a single buffer in the transfer */
        for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM-1; transfer_idx++)
            launch_target_wait_kernel(stream[transfer_idx], (uintptr_t)(data_address + (transfer_idx * TRANSFER_NUM_BUFFER * SIZE)), transfer_idx);
        for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM-1; transfer_idx++)
            cudaStreamSynchronize(stream[transfer_idx]);

        std::cout << " DOCA Transfer completed!\n";
    } else {
        std::cout << " Receive metadata from Target \n";
        std::cout << " \t -- To be handled by runtime - currently received via a TCP Stream\n";

        nixlMDStreamListener listener(initiator_port);
        listener.setupListenerSync();
        listener.acceptClient();

        for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
            remote_serdes[transfer_idx] = new nixlSerDes();
            std::string rrstr = listener.recvFromClient(); //recvFromTarget(initiator_port);
            remote_serdes[transfer_idx]->importStr(rrstr);
            remote_metadata = remote_serdes[transfer_idx]->getStr("AgentMD");
            assert (remote_metadata != "");
            agent.loadRemoteMD(remote_metadata, target_name);

            std::cout << " Verify Deserialized Target's Desc List at Initiator\n";
            nixl_xfer_dlist_t dram_target_doca(remote_serdes[transfer_idx]);
            nixl_xfer_dlist_t dram_initiator_doca = dram_for_doca[transfer_idx]->trim();
            dram_target_doca.print();

            std::cout << " Got metadata from " << target_name << " \n";
            std::cout << " Create transfer request with DOCA backend\n ";

            PUSH_RANGE("createXferReq", 1)
            if (processing.compare("gpu") == 0) {
                extra_params.customParam.resize(sizeof(uintptr_t));
                *((uintptr_t*) extra_params.customParam.data()) = (uintptr_t)stream[transfer_idx];
            }

            ret = agent.createXferReq(NIXL_WRITE, dram_initiator_doca, dram_target_doca,
                                "target", treq[transfer_idx], &extra_params);
            if (ret != NIXL_SUCCESS) {
                std::cerr << "Error creating transfer request\n";
                exit(-1);
            }
            POP_RANGE
        }

        std::cout << "Launch initiator send kernel on stream\n";

        /* Synthetic simulation of GPU processing data before sending */
        if (processing.compare("gpu") == 0) {
            for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
                std::cout << " Prepare data, GPU mode, transfer " << transfer_idx << " \n ";
                PUSH_RANGE("InitData", 2)
                launch_initiator_send_kernel(stream[transfer_idx], (uintptr_t)(data_address + (transfer_idx * TRANSFER_NUM_BUFFER * SIZE)), transfer_idx);
                POP_RANGE

                std::cout << " Post the request with DOCA backend transfer " << transfer_idx << " \n ";
                PUSH_RANGE("postXferReq", 3)
                status = agent.postXferReq(treq[transfer_idx]);
                assert(status >= NIXL_SUCCESS);
                POP_RANGE
            } 
        } else {
            /* Synthetic simulation of CPU processing data before sending */
            for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
                std::cout << " Prepare data, CPU mode, transfer " << transfer_idx << " \n ";
                PUSH_RANGE("InitData", 2)
                cudaMemset((void*)(data_address + (transfer_idx * TRANSFER_NUM_BUFFER * SIZE)), INITIATOR_VALUE, TRANSFER_NUM_BUFFER * SIZE);
                POP_RANGE

                std::cout << " Post the request with DOCA backend transfer " << transfer_idx << " \n ";
                PUSH_RANGE("postXferReq", 3)
                status = agent.postXferReq(treq[transfer_idx]);
                assert(status >= NIXL_SUCCESS);
                POP_RANGE
            }
        }

        std::cout << " Initiator posted Data Path transfer\n";
        std::cout << " Waiting for completion\n";

        PUSH_RANGE("getXferStatus", 4)
        for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
            while (status != NIXL_SUCCESS) {
                status = agent.getXferStatus(treq[transfer_idx]);
                assert(status >= NIXL_SUCCESS);
            }
        }
        POP_RANGE
        std::cout << " Completed Sending " << TRANSFER_NUM << " transfers using DOCA backend\n";
        for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++)
            agent.releaseXferReq(treq[transfer_idx]);
    }

    for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
        cudaStreamSynchronize(stream[transfer_idx]);
        cudaStreamDestroy(stream[transfer_idx]);
    }

    std::cout <<"Cleanup.. \n";
    
    for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++)
        agent.deregisterMem(*dram_for_doca[transfer_idx], &extra_params);
    cudaFree(data_address);

    for (int transfer_idx = 0; transfer_idx < TRANSFER_NUM; transfer_idx++) {
        if (role == "target")
            delete serdes[transfer_idx];
        else
            delete remote_serdes[transfer_idx];
    }

    return 0;
}
