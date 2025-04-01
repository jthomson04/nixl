#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
import zmq

from nixl._api import nixl_agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--zmq_ip", type=str, required=True)
    parser.add_argument("--zmq_port", type=int, default=5555)
    parser.add_argument(
        "--mode",
        type=str,
        default="initiator",
        help="Local IP in target, peer IP (target's) in initiator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # zmq as side channel
    _ctx = zmq.Context()
    _socket = _ctx.socket(zmq.PAIR)
    connect_str = f"tcp://{args.zmq_ip}:{args.zmq_port}"

    if args.mode == "target":
        _socket.bind(connect_str)
    else:
        _socket.connect(connect_str)

    # Allocate memory and register with NIXL
    agent = nixl_agent(args.name, None)
    if args.mode == "target":
        tensors = [torch.ones(10, dtype=torch.float32) for _ in range(2)]
    else:
        tensors = [torch.zeros(10, dtype=torch.float32) for _ in range(2)]
    print(f"{args.mode} Tensors: {tensors}")

    reg_descs = agent.register_memory(tensors)
    if not reg_descs:  # Same as reg_descs if successful
        print("Memory registration failed.")
        exit()

    # Exchange metadata. If socket is used (instead of genNotif) for target
    # descriptors, only the initiator needs metadata of the target.
    local_meta = agent.get_agent_metadata()
    if not local_meta:
        print("Acquiring local agent metadata failed.")
        exit()

    _socket.send(local_meta)
    remote_meta = _socket.recv()

    peer_name = agent.add_remote_agent(remote_meta)
    if not peer_name:
        print("Loading metadata from remote agent failed.")
        exit()

    print("My peer name is",peer_name)

    # Blocking send/recv (pull mode)
    if args.mode == "target":
        # Use same addresses for transfer as registration, just convert
        targer_descs = reg_descs.trim()
        # Can also indicate the notification that is expected to be received.
        agent.send_notif(peer_name, str.encode("EXAMPLE_NOTIF, DESCS:") + agent.get_serialized_descs(targer_descs))
        # The received notification can be any python bytes.
        while not agent.check_remote_xfer_done(peer_name, "EXAMPLE_NOTIF"):
            continue
    else:
        # Use same addresses for transfer as registration, just convert
        initiator_descs = reg_descs.trim()
        # wait for msg from target with its transfer descriptors
        msg = {}
        while not msg:
            msg = agent.get_new_notifs()

        if peer_name not in msg:
            print ("Error, wrong message received")
            exit()
        msg = msg[peer_name]
        notif = msg[0:msg.find(str.encode(","))]
        print (notif)
        print (msg[0:msg.find(str.encode(":"))])
        targer_descs = agent.deserialize_descs(msg[msg.find(str.encode(":"))+1:])

        xfer_handle = agent.initialize_xfer(
            "READ", initiator_descs, targer_descs, peer_name, "EXAMPLE_NOTIF"
        )
        if not xfer_handle:
            print("Creating transfer failed.")
            exit()

        state = agent.transfer(xfer_handle)
        if state == "ERR":
            print("Posting transfer failed.")
            exit()
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                print("Transfer got to Error state.")
                exit()
            elif state == "DONE":
                break

        # Verify data after read
        for i, tensor in enumerate(tensors):
            if not torch.allclose(tensor, torch.ones(10)):
                print(f"Data verification failed for tensor {i}.")
                exit()
        print(f"{args.mode} Data verification passed - {tensors}")

    if args.mode != "target":
        agent.remove_remote_agent(peer_name)
        agent.release_xfer_handle(xfer_handle)

    agent.deregister_memory(reg_descs)

    print("Test Complete.")
