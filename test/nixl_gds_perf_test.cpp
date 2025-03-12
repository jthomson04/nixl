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
#include <sys/time.h>

std::string generate_timestamped_filename(const std::string& base_name) {
    std::time_t t = std::time(nullptr);
    char timestamp[100];
    std::strftime(timestamp, sizeof(timestamp),
                  "%Y%m%d%H%M%S", std::localtime(&t));
    std::string return_val = base_name + std::string(timestamp);

    return return_val;
}

void printStats(size_t block_size, size_t batch_size, double write_time,
		double read_time) {


}


int main(int argc, char *argv[]) {

	std::string             role;
        nixl_status_t           status;
        int                     i, option;
        nixlAgentConfig         cfg(true);
        nixl_b_params_t         params;
        nixlStringDesc          *buf;
        nixlStringDesc          *buf_rd;
	nixlStringDesc          *ftrans;
        nixlBackendH	        *gds;
	std::vector<int>	indices;
        void                    **addr, **addr_rd;
	int                     *fd;
	int			num_transfers = 10;
	int			size;
	struct timeval		start_time, end_time;
	struct timeval		write_time, read_time;

	nixl_reg_dlist_t        vram_for_gds_read(VRAM_SEG);
	nixl_reg_dlist_t        vram_for_gds(VRAM_SEG);
	nixl_reg_dlist_t        file_for_gds(FILE_SEG, false);
	nixlXferSideH		*vram_side, *vram_side_rd, *file_side;
	nixlXferReqH            *treq_read, *treq_write;
        std::string             name, path;
	nixl_xfer_state_t	xstatus;

	if (argc < 4) {
		std::cout << "Not enough arguments provided "<<std::endl;
		std::cout << "<nixl_perf_bech> -p <path> -r <num_transfers > "
			  << " -s <size> "<< std::endl;
		exit(-1);
	}

	while ((option = getopt(argc, argv, "p:r:s:")) != -1) {
	   switch (option) {
	     case 'p':
		path = optarg;
		break;
	     case 'r':
		num_transfers = std::stoi(optarg);
		break;
	     case 's':
		size = std::stoi(optarg);
		break;
	     case '?':
		std::cerr <<"Error: Unknown option or missing argument."
			  << std::endl;
		return 1;
	     default:
		return 1;
	   }
	}
	addr	= new void*[num_transfers];
	addr_rd	= new void*[num_transfers];
	buf	= new nixlStringDesc[num_transfers];
	buf_rd	= new nixlStringDesc[num_transfers];
	ftrans  = new nixlStringDesc[num_transfers];
	fd	= new int[num_transfers];

	nixlAgent agent("GDSTester", cfg);
	gds	= agent.createBackend("GDS", params);
        if (gds == nullptr) {
		std::cerr <<"Error creating a new backend\n";
		exit(-1);
	}

	/** Setting up control path configurations */
	for (i = 0; i < num_transfers; i++) {
	    cudaMalloc((void **)&addr[i], size);
	    cudaMalloc((void **)&addr_rd[i], size);
	    cudaMemset(addr[i], 'A', size);
	    cudaMemset(addr_rd[i], 0, size);

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
	    buf[i].len    = size;
	    buf[i].devId  = 0;
	    vram_for_gds.addDesc(buf[i]);
	    buf_rd[i].addr	  = (uintptr_t)(addr_rd[i]);
	    buf_rd[i].len	  = size;
	    buf_rd[i].devId  = 0;
	    vram_for_gds_read.addDesc(buf_rd[i]);
	    ftrans[i].len   = size;
            ftrans[i].devId = fd[i];
            file_for_gds.addDesc(ftrans[i]);
	}

	status = agent.registerMem(file_for_gds, gds);
	assert(status == NIXL_SUCCESS);

	status = agent.registerMem(vram_for_gds, gds);
	assert(status == NIXL_SUCCESS);

	status = agent.registerMem(vram_for_gds_read, gds);
	assert(status == NIXL_SUCCESS);

	nixl_xfer_dlist_t vram_for_gds_list	 = vram_for_gds.trim();
        nixl_xfer_dlist_t file_for_gds_list	 = file_for_gds.trim();
	nixl_xfer_dlist_t vram_for_gds_read_list = vram_for_gds_read.trim();


	status = agent.prepXferSide(file_for_gds_list, "GDSTester", gds,
				    file_side);
	assert(status == NIXL_SUCCESS);

	status = agent.prepXferSide(vram_for_gds_list, "", gds,
				    vram_side);
	assert(status == NIXL_SUCCESS);

	status = agent.prepXferSide(vram_for_gds_read_list, "", gds,
				    vram_side_rd);
	for(int i = 0; i<num_transfers; i++)
		indices.push_back(i);

	status = agent.makeXferReq(vram_side, indices, file_side, indices,
				   "", NIXL_WRITE, treq_write, false);
	assert(status == NIXL_SUCCESS);

	status = agent.makeXferReq(vram_side_rd, indices, file_side, indices,
				   "", NIXL_READ, treq_read, false);
	assert(status == NIXL_SUCCESS);

	gettimeofday(&start_time, NULL);
	xstatus = agent.postXferReq(treq_write);
	assert(xstatus != NIXL_XFER_ERR);
	nixl_xfer_state_t wr_state;
	while (wr_state != NIXL_XFER_DONE) {
            wr_state = agent.getXferStatus(treq_write);
            assert(wr_state != NIXL_XFER_ERR);
        }
	gettimeofday(&end_time, NULL);
	timersub(&end_time, &start_time, &write_time);

	gettimeofday(&start_time, NULL);
	xstatus = agent.postXferReq(treq_read);
	assert(xstatus != NIXL_XFER_ERR);

	nixl_xfer_state_t rd_status;
	while (rd_status != NIXL_XFER_DONE) {
            rd_status = agent.getXferStatus(treq_read);
            assert(rd_status != NIXL_XFER_ERR);
        }
	gettimeofday(&end_time, NULL);
	timersub(&end_time, &start_time, &read_time);

	int err = 0;
	std::cout <<"**** Verifying Reads after Writes \n";
	for (i = 0; i < num_transfers; i++) {
	    char *r_buffer = (char *)malloc(size);
	    cudaMemcpy(r_buffer, addr_rd[i], size, cudaMemcpyDeviceToHost);
	    for (int i = 0; i < size; i++) {
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

	return 0;
}
