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
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cerrno>
#include <cstring>

#define SIZE (1048576 * 16)


std::string generate_timestamped_filename(const std::string& base_name) {
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp),
                  "%Y%m%d%H%M%S", std::localtime(&t));
    std::string return_val = base_name + std::string(timestamp);

    return return_val;
}

int main(int argc, char *argv[])
{
        nixl_status_t           ret = NIXL_SUCCESS;
        void                    *addr[4096];
        std::string             role;
        int                     status = 0;
        int                     i;
        int                     fd[4096];

        nixlAgentConfig         cfg(true);
        nixl_b_params_t         params;
        nixlStringDesc          buf[4096];
        nixlStringDesc          ftrans[4096];
        nixlBackendH	        *gds;

        nixl_reg_dlist_t        vram_for_gds(VRAM_SEG);
        nixl_reg_dlist_t        file_for_gds(FILE_SEG, false);
        nixlXferReqH            *treq, *treq_new;
        std::string             name;
	int			num_transfers = 10;

        std::cout << "Starting Agent for " << "GDS Test Agent" << "\n";
        nixlAgent agent("GDSTester", cfg);

        gds          = agent.createBackend("GDS", params);
        if (gds == nullptr) {
		std::cerr <<"Error creating a new backend\n";
		exit(-1);
	}
        /** Argument Parsing */
        if (argc < 2) {
                std::cout <<"Enter the required arguments\n" << std::endl;
                std::cout <<"Directory path <NUM_REQUESTS>" << std::endl;
                exit(-1);
        }
        std::string path = std::string(argv[1]);
	if (argc > 2)
		num_transfers = std::stoi(argv[2]);

        for (i = 0; i < num_transfers; i++) {
                cudaMalloc((void **)&addr[i], SIZE);
                cudaMemset(addr[i], 'A', SIZE);
		name = generate_timestamped_filename("testfile");
		name = path +"/"+ name +"_"+ std::to_string(i);

		fd[i] = open(name.c_str(), O_RDWR|O_CREAT, 0744);
	        if (fd[i] < 0) {
		   std::cerr<<"Error: "<<strerror(errno)<<std::endl;
		   std::cerr<<"Open call failed to open file\n";
	           std::cerr<<"Cannot run tests\n";
	           return 1;
		}
                buf[i].addr   = (uintptr_t)(addr[i]);
                buf[i].len    = SIZE;
                buf[i].devId  = 0;
                vram_for_gds.addDesc(buf[i]);

                ftrans[i].addr  = 0; // this is offset
                ftrans[i].len   = SIZE;
                ftrans[i].devId = fd[i];
                file_for_gds.addDesc(ftrans[i]);
        }
        agent.registerMem(file_for_gds, gds);
        agent.registerMem(vram_for_gds, gds);

        nixl_xfer_dlist_t vram_for_gds_list = vram_for_gds.trim();
        nixl_xfer_dlist_t file_for_gds_list = file_for_gds.trim();

        ret = agent.createXferReq(vram_for_gds_list, file_for_gds_list,
                                  "GDSTester", "", NIXL_WRITE, treq);
        if (ret != NIXL_SUCCESS) {
                std::cerr << "Error creating transfer request\n" << ret;
                exit(-1);
        }

	std::cout << " **** WRITES with GDS Backend to File ** \n";
        std::cout << " Post the request with GDS backend\n ";
        status = agent.postXferReq(treq);
        std::cout << " GDS File IO has been posted\n";
        std::cout << " Waiting for completion\n";

        while (status != NIXL_XFER_DONE) {
            status = agent.getXferStatus(treq);
            assert(status != NIXL_XFER_ERR);
        }
        std::cout <<" Completed writing data using GDS backend\n";
        agent.invalidateXferReq(treq);

	std::cout <<"**** Reads with GDS Backend to File \n";
        for (i = 0; i < num_transfers; i++) {
	    cudaMemset(addr[i], 0, SIZE);
	}

	ret = agent.createXferReq(vram_for_gds_list, file_for_gds_list,
				  "GDSTester", "", NIXL_READ, treq_new);
	if (ret != NIXL_SUCCESS) {
		std::cerr << "Error in creating transfer request\n" << ret;
		exit(-1);
	}

	std::cout << " Post the request with GDS backend\n ";
        status = agent.postXferReq(treq_new);
        std::cout << " GDS File IO has been posted\n";
        std::cout << " Waiting for completion\n";

        while (status != NIXL_XFER_DONE) {
            status = agent.getXferStatus(treq_new);
            assert(status != NIXL_XFER_ERR);
        }
        std::cout <<" Completed Reading data using GDS backend\n";

	int err = 0;
	std::cout <<"**** Verifying Reads after Writes \n";
	for (i = 0; i < num_transfers; i++) {
	    char *r_buffer = (char *)malloc(SIZE);
	    cudaMemcpy(r_buffer, addr[i], SIZE, cudaMemcpyDeviceToHost);
	    for (size_t i = 0; i < SIZE; i++) {
		if (r_buffer[i] != 'A') {
		   std::cerr<<"Mismatch during READS\n";
		   free(r_buffer);
		   err++;
		   break;
		}
	    }
	}
	if (!err)
	   std::cout << "Write and Read Successful for " << num_transfers
		     <<" Requests\n";
        agent.invalidateXferReq(treq_new);

        std::cout <<"Cleanup.. \n";
        agent.deregisterMem(vram_for_gds, gds);
        agent.deregisterMem(file_for_gds, gds);
	for (i = 0; i < num_transfers; i++) {
	    cudaFree(addr[i]);
	}
        return 0;
}

