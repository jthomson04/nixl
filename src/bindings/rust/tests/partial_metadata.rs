// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nixl_sys::{
    Agent, MemType, NixlError, OptArgs, RegDescList, SystemStorage, XferDescList, XferOp,
};
use std::time::{Duration, Instant};

fn ucx_agent(name: &str) -> Result<Agent, NixlError> {
    let agent = Agent::new(name)?;
    let (_, params) = agent.get_plugin_params("UCX")?;
    agent.create_backend("UCX", &params)?;
    Ok(agent)
}

// This test requires the native UCX backend. Missing native libraries are a
// failure, including when the Rust crate uses runtime C API forwarding.
#[test]
fn partial_metadata_connects_a_fresh_receiver() -> Result<(), NixlError> {
    assert!(!nixl_sys::is_stub(), "the native NIXL backend is required");
    let source = ucx_agent("partial-metadata-source")?;
    let mut selected = SystemStorage::new(65536)?;
    let mut unrelated = SystemStorage::new(65536)?;
    selected.memset(0x5a);
    unrelated.memset(0xa5);
    let _selected_registration = source.register_memory(&selected, None)?;
    let mut unrelated_registration = Some(source.register_memory(&unrelated, None)?);

    let mut descriptors = RegDescList::new(MemType::Dram)?;
    descriptors.add_storage_desc(&selected)?;
    let mut options = OptArgs::new()?;
    options.set_include_connection_info(true)?;
    let metadata = source.get_local_partial_md(&descriptors, Some(&options))?;

    for index in 0..2 {
        let receiver = ucx_agent(&format!("partial-metadata-receiver-{index}"))?;
        let remote = receiver.load_remote_md(&metadata)?;
        let mut destination = SystemStorage::new(selected.as_slice().len())?;
        destination.memset(0);
        let _destination_registration = receiver.register_memory(&destination, None)?;
        let mut local = XferDescList::new(MemType::Dram)?;
        local.add_storage_desc(&destination)?;
        let mut remote_selected = XferDescList::new(MemType::Dram)?;
        remote_selected.add_storage_desc(&selected)?;
        let request =
            receiver.create_xfer_req(XferOp::Read, &local, &remote_selected, &remote, None)?;
        if receiver.post_xfer_req(&request, None)? {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !receiver.get_xfer_status(&request)?.is_success() {
                assert!(Instant::now() < deadline, "native read timed out");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        assert_eq!(destination.as_slice(), selected.as_slice());

        let mut remote_unrelated = XferDescList::new(MemType::Dram)?;
        remote_unrelated.add_storage_desc(&unrelated)?;
        assert!(receiver.check_remote_metadata(&remote, Some(&remote_selected)));
        assert!(
            !receiver.check_remote_metadata(&remote, Some(&remote_unrelated)),
            "partial metadata must exclude unrelated registrations"
        );

        // Release the unrelated registration before importing the same
        // snapshot into a second fresh receiver.
        if index == 0 {
            drop(unrelated_registration.take());
        }
    }
    Ok(())
}
