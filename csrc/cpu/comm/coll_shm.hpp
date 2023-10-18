// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#ifndef _COLL_SHM__HPP_
#define _COLL_SHM__HPP_

#include <torch/extension.h>
#include <oneapi/ccl.hpp>

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  256bit vector ==> 32 bytes, 512bit vector ==> 64 bytes
// If you change implementation of reduce_2_bf16_buffers or reduce_2_fp32_buffers, check
// whether this number needs to be changed
#define VECTOR_LENGTH_IN_BYTES 32

void shm_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type);
void create_shm_workspace(int world_size, int rank, ccl::communicator& comm);

// local reduce functions
void reduce_2_bf16_buffers(int num_elements, void* in_out, void* in)
    __attribute__((target("avx512bw")));

void reduce_bf16_buffers(int num_elements, int num_buffers, struct allreduce_workspace* workspace)
    __attribute__((target("avx512bw")));

void reduce_2_fp32_buffers(int num_elements, void* in_out, void* in)
    __attribute__((target("avx512bw")));

void reduce_fp32_buffers(int num_elements, int num_buffers, struct allreduce_workspace* workspace)
    __attribute__((target("avx512bw")));

#endif //_COLL_SHM__HPP_

