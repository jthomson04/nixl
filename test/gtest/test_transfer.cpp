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

#include "common.h"
#include "mem_buffer.h"
#include "gtest/gtest.h"

#include "nixl.h"
#include "nixl_types.h"

#include <gtest/gtest.h>
#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <random>
#include <set>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace gtest {

class TestTransfer : public testing::TestWithParam<std::string> {
protected:
    TestTransfer() : rd(), gen(rd()), distrib() {}

    static nixlAgentConfig
    getConfig(int listen_port) {
        return nixlAgentConfig(true,
                               listen_port > 0,
                               listen_port,
                               nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT,
                               0,
                               100000);
    }

    static int getPort(int i)
    {
        return 9000 + i;
    }

    void SetUp() override
    {
#ifdef HAVE_CUDA
        m_cuda_device = (cudaSetDevice(0) == cudaSuccess);
#endif

        // Create two agents
        for (size_t i = 0; i < 2; i++) {
            agents.emplace_back(std::make_unique<nixlAgent>(getAgentName(i),
                                                            getConfig(getPort(i))));
            nixlBackendH *backend_handle = nullptr;
            nixl_status_t status = agents.back()->createBackend(getBackendName(), {},
                                                                backend_handle);
            ASSERT_EQ(status, NIXL_SUCCESS);
            EXPECT_NE(backend_handle, nullptr);
        }
    }

    void TearDown() override
    {
        agents.clear();
    }

    std::string getBackendName() const
    {
        return GetParam();
    }

    static nixl_opt_args_t extra_params_ip(int remote)
    {
        nixl_opt_args_t extra_params;

        extra_params.ipAddr = "127.0.0.1";
        extra_params.port   = getPort(remote);
        return extra_params;
    }

    nixl_status_t fetchRemoteMD(int local = 0, int remote = 1)
    {
        auto extra_params = extra_params_ip(remote);

        return agents[local]->fetchRemoteMD(getAgentName(remote),
                                            &extra_params);
    }

    nixl_status_t checkRemoteMD(int local = 0, int remote = 1)
    {
        nixl_xfer_dlist_t descs(DRAM_SEG);
        return agents[local]->checkRemoteMD(getAgentName(remote), descs);
    }

    template<typename Desc, nixl_mem_t MemType, typename Iter>
    nixlDescList<Desc>
    makeDescList(Iter begin, Iter end) {
        nixlDescList<Desc> desc_list(MemType);
        for (auto it = begin; it != end; ++it) {
            desc_list.addDesc(Desc(it->data(), it->size(), DEV_ID));
        }
        return desc_list;
    }

    template<nixl_mem_t MemType>
    void registerMem(nixlAgent &agent, const std::vector<MemBuffer<MemType>> &buffers)
    {
        auto reg_list = makeDescList<nixlBlobDesc, MemType>(buffers.begin(), buffers.end());
        agent.registerMem(reg_list);
    }

    void exchangeMDIP()
    {
        for (size_t i = 0; i < agents.size(); i++) {
            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j) {
                    continue;
                }

                auto status = fetchRemoteMD(i, j);
                ASSERT_EQ(NIXL_SUCCESS, status);
                do {
                    status = checkRemoteMD(i, j);
                } while (status != NIXL_SUCCESS);
            }
        }
    }

    void exchangeMD()
    {
        // Connect the existing agents and exchange metadata
        for (size_t i = 0; i < agents.size(); i++) {
            nixl_blob_t md;
            nixl_status_t status = agents[i]->getLocalMD(md);
            ASSERT_EQ(status, NIXL_SUCCESS);

            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                std::string remote_agent_name;
                status = agents[j]->loadRemoteMD(md, remote_agent_name);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_EQ(remote_agent_name, getAgentName(i));
            }
        }
    }

    void invalidateMD()
    {
        // Disconnect the agents and invalidate remote metadata
        for (size_t i = 0; i < agents.size(); i++) {
            for (size_t j = 0; j < agents.size(); j++) {
                if (i == j)
                    continue;
                nixl_status_t status = agents[j]->invalidateRemoteMD(
                        getAgentName(i));
                ASSERT_EQ(status, NIXL_SUCCESS);
            }
        }
    }

    void
    waitForXfer(nixlAgent &from,
                const std::string &from_name,
                nixlAgent &to,
                nixlXferReqH *xfer_req) {
        bool xfer_done;
        do {
            // progress on "from" agent while waiting for completion
            nixl_status_t status = from.getXferStatus(xfer_req);
            EXPECT_TRUE((status == NIXL_SUCCESS) || (status == NIXL_IN_PROG));
            xfer_done = (status == NIXL_SUCCESS);
        } while (!xfer_done);
    }

    template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType>
    void
    doTransfer(nixlAgent &from,
               const std::string &from_name,
               nixlAgent &to,
               const std::string &to_name,
               size_t size,
               size_t count,
               size_t batch_size,
               size_t repeat,
               nixl_xfer_op_t mode,
               std::function<void()> setup_md,
               const std::vector<std::string> &expected_notifs) {
        std::vector<MemBuffer<LocalMemType>> local_buffers;
        std::vector<MemBuffer<RemoteMemType>> remote_buffers;
        for (size_t i = 0; i < count; i++) {
            if (mode == NIXL_WRITE) {
                local_buffers.emplace_back(createRandomData<LocalMemType>(size));
                remote_buffers.emplace_back(size);
            } else {
                local_buffers.emplace_back(size);
                remote_buffers.emplace_back(createRandomData<RemoteMemType>(size));
            }
        }

        registerMem(from, local_buffers);
        registerMem(to, remote_buffers);
        setup_md();

        auto start_time = absl::Now();
        size_t total_transferred = 0;
        size_t notif_idx = 0;

        for (size_t i = 0; i < repeat; i++) {
            for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, count);

                nixl_opt_args_t extra_params;
                extra_params.hasNotif = true;
                extra_params.notifMsg = expected_notifs[notif_idx++];

                nixlXferReqH *xfer_req = nullptr;
                nixl_status_t status = from.createXferReq(
                        mode,
                        makeDescList<nixlBasicDesc, LocalMemType>(local_buffers.begin() + batch_start,
                                                                  local_buffers.begin() + batch_end),
                        makeDescList<nixlBasicDesc, RemoteMemType>(
                                remote_buffers.begin() + batch_start,
                                remote_buffers.begin() + batch_end),
                        to_name,
                        xfer_req,
                        &extra_params);
                ASSERT_EQ(status, NIXL_SUCCESS);
                EXPECT_NE(xfer_req, nullptr);

                status = from.postXferReq(xfer_req);
                ASSERT_GE(status, NIXL_SUCCESS);

                waitForXfer(from, from_name, to, xfer_req);

                status = from.getXferStatus(xfer_req);
                EXPECT_EQ(status, NIXL_SUCCESS);

                // Verify transfer was successful for this batch
                for (size_t j = batch_start; j < batch_end; j++) {
                    EXPECT_EQ(local_buffers[j], remote_buffers[j])
                            << "Transfer validation failed for buffer " << j;
                }

                status = from.releaseXferReq(xfer_req);
                EXPECT_EQ(status, NIXL_SUCCESS);

                total_transferred += (batch_end - batch_start) * size;
            }
        }

        auto total_time = absl::ToDoubleSeconds(absl::Now() - start_time);
        auto bandwidth = total_transferred / total_time / (1024 * 1024 * 1024);
        Logger() << (mode == NIXL_WRITE ? "Write" : "Read") << " transfer: " << size << "x" << count
                 << "x" << repeat << "=" << total_transferred << " bytes in " << total_time
                 << " seconds "
                 << "(" << bandwidth << " GB/s)";

        invalidateMD();
    }

    template<nixl_mem_t LocalMemType, nixl_mem_t RemoteMemType>
    void
    doTransfers(nixlAgent &from,
                const std::string &from_name,
                nixlAgent &to,
                const std::string &to_name,
                size_t size,
                size_t count,
                size_t batch_size,
                size_t repeat,
                nixl_xfer_op_t mode,
                std::function<void()> metadataSetup) {
        std::vector<std::string> expected_notifs;
        for (size_t i = 0; i < repeat; i++) {
            for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
                size_t batch_idx = batch_start / batch_size;
                expected_notifs.push_back(absl::StrFormat("notification_%zu", batch_idx));
            }
        }

        doTransfer<LocalMemType, RemoteMemType>(from,
                                                from_name,
                                                to,
                                                to_name,
                                                size,
                                                count,
                                                batch_size,
                                                repeat,
                                                mode,
                                                metadataSetup,
                                                expected_notifs);

        nixl_notifs_t notif_map;
        nixl_status_t status = to.getNotifs(notif_map);
        ASSERT_EQ(status, NIXL_SUCCESS);

        auto &notif_list = notif_map[from_name];
        EXPECT_EQ(notif_list.size(), expected_notifs.size())
                << "Expected " << expected_notifs.size() << " notifications, got "
                << notif_list.size();

        std::set<std::string> expected_msgs(expected_notifs.begin(), expected_notifs.end());

        for (const auto &msg : notif_list) {
            EXPECT_TRUE(expected_msgs.find(msg) != expected_msgs.end())
                    << "Unexpected notification message: " << msg;
        }
    }

    nixlAgent &getAgent(size_t idx)
    {
        return *agents[idx];
    }

    std::string getAgentName(size_t idx)
    {
        return absl::StrFormat("agent_%d", idx);
    }

    template<nixl_mem_t MemType>
    std::vector<uint8_t>
    createRandomData(size_t size) {
        size_t aligned_size = (size + 7) & ~7;
        std::vector<uint8_t> data(aligned_size);

        for (size_t i = 0; i < aligned_size; i += 8) {
            uint64_t rand_val = distrib(gen);
            for (size_t j = 0; j < 8; ++j) {
                data[i + j] = static_cast<uint8_t>(rand_val >> (j * 8));
            }
        }

        data.resize(size);
        return data;
    }

    bool m_cuda_device = false;

private:
    static constexpr uint64_t DEV_ID = 0;

    std::vector<std::unique_ptr<nixlAgent>> agents;
    std::random_device rd;
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint64_t> distrib;
};

TEST_P(TestTransfer, RandomSizes) {
    // Tuple fields are: size, count, batch_size, repeat
    constexpr std::array<std::tuple<size_t, size_t, size_t, size_t>, 4> test_cases = {
            {{40, 1000, 1, 1}, {4096, 128, 32, 3}, {32768, 32, 4, 3}, {1000000, 8, 1, 3}}};

    for (const auto &[size, count, batch_size, repeat] : test_cases) {
        doTransfers<DRAM_SEG, DRAM_SEG>(getAgent(0),
                                        getAgentName(0),
                                        getAgent(1),
                                        getAgentName(1),
                                        size,
                                        count,
                                        batch_size,
                                        repeat,
                                        NIXL_WRITE,
                                        [this]() { exchangeMD(); });
        doTransfers<DRAM_SEG, DRAM_SEG>(getAgent(0),
                                        getAgentName(0),
                                        getAgent(1),
                                        getAgentName(1),
                                        size,
                                        count,
                                        batch_size,
                                        repeat,
                                        NIXL_READ,
                                        [this]() { exchangeMD(); });
    }
}

TEST_P(TestTransfer, remoteMDFromSocket) {
    constexpr size_t size = 16 * 1024;
    constexpr size_t count = 4;

    if (m_cuda_device) {
        doTransfers<VRAM_SEG, VRAM_SEG>(getAgent(0),
                                        getAgentName(0),
                                        getAgent(1),
                                        getAgentName(1),
                                        size,
                                        count,
                                        1,
                                        1,
                                        NIXL_WRITE,
                                        [this]() { exchangeMDIP(); });
    } else {
        doTransfers<DRAM_SEG, DRAM_SEG>(getAgent(0),
                                        getAgentName(0),
                                        getAgent(1),
                                        getAgentName(1),
                                        size,
                                        count,
                                        1,
                                        1,
                                        NIXL_WRITE,
                                        [this]() { exchangeMDIP(); });
    }
}

INSTANTIATE_TEST_SUITE_P(ucx, TestTransfer, testing::Values("UCX"));
INSTANTIATE_TEST_SUITE_P(ucx_mo, TestTransfer, testing::Values("UCX_MO"));

} // namespace gtest
