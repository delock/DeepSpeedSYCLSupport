// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "coll_shm.hpp"
#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <oneapi/ccl.hpp>

// states for collectives
enum coll_state {
    coll_begin = 0,
    // coll states for naive allreduce
    coll_allreduce_naive__copy_in_done,   // this state is for local_rank != 0
    coll_allreduce_naive__reduce_done,    // this state is for local_rank == 0
    coll_allreduce_naive__copy_out_done,  // this state is for local_rank != 0
};

// SHM building blocks
struct SharedData {
    const char* name;
    int descriptor;
    void* bytes;
    size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes)
{
    int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
        data->name = name;
        data->descriptor = d;
        data->bytes = bytes;
        data->nbytes = nbytes;
    } else {
        printf("shared_open %s failed\n", name);
        data->descriptor = -1;
    }
}

void shared_create(SharedData* data, const char* name, void* bytes, size_t nbytes)
{
    int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (d != -1) {
        if (nbytes = write(d, bytes, nbytes)) { shared_open(data, name, nbytes); }
    } else {
        printf("shared_create %s failed\n", name);
    }
}

void shared_close(SharedData* data)
{
    if (data->descriptor != -1) {
        munmap(data->bytes, data->nbytes);
        shm_unlink(data->name);
    }
}

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
SharedData allreduce_buffer;
struct allreduce_workspace {
    enum coll_state state;
    char buffer[MAX_BUF_SIZE];
};
struct allreduce_workspace* workspace;

void wait_buffer_state_until(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index].state);

    while (*state_ptr != state)
        ;
}

void wait_buffer_state_until_not(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index].state);

    while (*state_ptr == state)
        ;
}

__m512 cvt_bf16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_bf16_to_fp32(const __m256i src)
{
    auto y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src) __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_bf16(const __m512 src)
{
    __m512i value = _mm512_castps_si512(src);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    // uint32_t lsb = (input >> 16) & 1;
    auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
    // uint32_t rounding_bias = 0x7fff + lsb;
    t_value = _mm512_add_epi32(t_value, vec_bias);
    // input += rounding_bias;
    t_value = _mm512_add_epi32(t_value, value);
    // input = input >> 16;
    t_value = _mm512_srli_epi32(t_value, 16);
    // Check NaN before converting back to bf16
    t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
    return _mm512_cvtusepi32_epi16(t_value);
}

// N_REDUCE_LIMIT is the number of buffers that can be reduced together in one shot.
// Compared with do N-1 2-reduces which needs 2*(N-1) read and N-1 write,
// N-reduce only needs N read and 1 write, this saves 2/3 memory bandwidth.
// When increase N_REDUCE_LIMIT to a bigger number, do the following steps
// 1. Extend REPEAT_<X> macros list down below
// 2. Extend switch cases which call "REPEAT(X, ...)" down below
#define N_REDUCE_LIMIT 8

void reduce_all_buffers(struct allreduce_workspace* workspace,
                        int num_elements,
                        c10::ScalarType scalar_type,
                        int num_buffers)
{
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                reduce_bf16_buffers(num_elements, num_buffers, workspace);
            } else {
                for (int i = 1; i < num_buffers; i++) {
                    reduce_2_bf16_buffers(num_elements, workspace[0].buffer, workspace[i].buffer);
                }
            }
            break;
        case c10::ScalarType::Float:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                reduce_fp32_buffers(num_elements, num_buffers, workspace);
            } else {
                for (int i = 1; i < num_buffers; i++) {
                    reduce_2_fp32_buffers(num_elements, workspace[0].buffer, workspace[i].buffer);
                }
            }
            break;
        default: assert(!"Should not get here");
    }
}

#define REPEAT(N, x) REPEAT_##N(x)
#define REPEAT_1(x) x(1)
#define REPEAT_2(x) \
    REPEAT_1(x);    \
    x(2)
#define REPEAT_3(x) \
    REPEAT_2(x);    \
    x(3)
#define REPEAT_4(x) \
    REPEAT_3(x);    \
    x(4)
#define REPEAT_5(x) \
    REPEAT_4(x);    \
    x(5)
#define REPEAT_6(x) \
    REPEAT_5(x);    \
    x(6)
#define REPEAT_7(x) \
    REPEAT_6(x);    \
    x(7)

#define CVT_ADD_BF16(x)                                                                \
    do {                                                                               \
        auto in##x##_val =                                                             \
            cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(workspace[x].buffer + i))); \
        inout_val = _mm512_add_ps(inout_val, in##x##_val);                             \
    } while (0)

// num_elements must be divisible by 16 (caller check)
void reduce_bf16_buffers(int num_elements, int num_buffers, struct allreduce_workspace* workspace)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 2; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(workspace[0].buffer + i)));
        switch (num_buffers) {
            case 8: REPEAT(7, CVT_ADD_BF16); break;
            case 7: REPEAT(6, CVT_ADD_BF16); break;
            case 6: REPEAT(5, CVT_ADD_BF16); break;
            case 5: REPEAT(4, CVT_ADD_BF16); break;
            case 4: REPEAT(3, CVT_ADD_BF16); break;
            case 3: REPEAT(2, CVT_ADD_BF16); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_si256((__m256i*)(workspace[0].buffer + i), cvt_fp32_to_bf16(inout_val));
    }
}

void reduce_2_bf16_buffers(int num_elements, void* in_out, void* in1)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 2; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in_out + i)));
        auto in1_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in1 + i)));
        inout_val = _mm512_add_ps(inout_val, in1_val);
        _mm256_storeu_si256((__m256i*)((char*)in_out + i), cvt_fp32_to_bf16(inout_val));
    }
}

#define CVT_ADD_F32(x)                                                         \
    do {                                                                       \
        auto in##x##_val = _mm256_loadu_ps((float*)(workspace[x].buffer + i)); \
        inout_val = _mm256_add_ps(inout_val, in##x##_val);                     \
    } while (0)

// num_elements must be divisible by 16 (caller check)
void reduce_fp32_buffers(int num_elements, int num_buffers, struct allreduce_workspace* workspace)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 4; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)(workspace[0].buffer + i));
        switch (num_buffers) {
            case 8: REPEAT(7, CVT_ADD_F32); break;
            case 7: REPEAT(6, CVT_ADD_F32); break;
            case 6: REPEAT(5, CVT_ADD_F32); break;
            case 5: REPEAT(4, CVT_ADD_F32); break;
            case 4: REPEAT(3, CVT_ADD_F32); break;
            case 3: REPEAT(2, CVT_ADD_F32); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_ps((float*)(workspace[0].buffer + i), inout_val);
    }
}

void reduce_2_fp32_buffers(int num_elements, void* in_out, void* in1)
{
#pragma omp parallel for
    for (int i = 0; i < num_elements * 4; i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)((char*)in_out + i));
        auto in1_val = _mm256_loadu_ps((float*)((char*)in1 + i));
        inout_val = _mm256_add_ps(inout_val, in1_val);
        _mm256_storeu_ps((float*)((char*)in_out + i), inout_val);
    }
}

static void parallel_memcpy(void* to, void* from, size_t n_bytes)
    __attribute__((target("avx512bw")));
static void parallel_memcpy(void* to, void* from, size_t n_bytes)
{
#pragma omp parallel for
    for (int i = 0; i < n_bytes; i += VECTOR_LENGTH_IN_BYTES) {
        auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
        _mm256_storeu_si256((__m256i*)((char*)to + i), val);
    }
}

void shm_all_reduce(int local_size, int local_rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
    for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
        auto data_ptr = ((char*)buf + offset);
        size_t chunk_size = data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
        size_t chunk_el = chunk_size / (data_size / numel);

        parallel_memcpy(workspace[local_rank].buffer, data_ptr, chunk_size);
        std::atomic_thread_fence(std::memory_order_release);
        workspace[local_rank].state = coll_allreduce_naive__copy_in_done;

        if (local_rank == 0) {
            // compute allreduce result on local_rank 0
            for (int i = 1; i < local_size; i++) {
                // wait until the other local_rank copy the buffer
                wait_buffer_state_until(i, coll_allreduce_naive__copy_in_done);
            }
            reduce_all_buffers(workspace, chunk_el, scalar_type, local_size);
            std::atomic_thread_fence(std::memory_order_release);
            workspace[local_rank].state = coll_allreduce_naive__reduce_done;
            parallel_memcpy(data_ptr, workspace[0].buffer, chunk_size);
        }
        if (local_rank != 0) {
            wait_buffer_state_until(0, coll_allreduce_naive__reduce_done);
            parallel_memcpy(data_ptr, workspace[0].buffer, chunk_size);
            std::atomic_thread_fence(std::memory_order_release);
            workspace[local_rank].state = coll_allreduce_naive__copy_out_done;
        }
        if (local_rank == 0) {
            for (int i = 1; i < local_size; i++) {
                wait_buffer_state_until(i, coll_allreduce_naive__copy_out_done);
            }
            std::atomic_thread_fence(std::memory_order_release);
            workspace[local_rank].state = coll_begin;
        }
        if (local_rank != 0) {
            // if local_rank 0 spin too fast it could be in state 1 of next allreduce
            // in this case wait_buffer_state_until(0, 0) may cause deadlock
            // what we are certain is when local_rank 0 finishes the state won't be 2
            wait_buffer_state_until_not(0, coll_allreduce_naive__reduce_done);
            workspace[local_rank].state = coll_begin;
        }
    }
}

void create_shm_workspace(int size, int local_rank, ccl::communicator& comm)
{
    auto addr_string = std::getenv("MASTER_ADDR");
    if (addr_string == NULL) { addr_string = ""; }
    auto port_string = std::getenv("MASTER_PORT");
    if (port_string == NULL) { port_string = ""; }
    char shm_name[NAME_BUF_SIZE];
    snprintf(shm_name,
             NAME_BUF_SIZE,
             "%s_%d_%s_%s",
             SHM_BUFFER_NAME,
             getuid(),
             addr_string,
             port_string);

    if (local_rank == 0) {
        workspace =
            (struct allreduce_workspace*)malloc(size * sizeof(struct allreduce_workspace));
        shared_create(
            &allreduce_buffer, shm_name, workspace, size * sizeof(struct allreduce_workspace));
        workspace = (struct allreduce_workspace*)allreduce_buffer.bytes;
        for (int i = 0; i < size; i++) { workspace[i].state = coll_begin; }
    }
    ccl::barrier(comm).wait();
    if (local_rank != 0) {
        shared_open(&allreduce_buffer, shm_name, size * sizeof(struct allreduce_workspace));
    }
    workspace = (struct allreduce_workspace*)allreduce_buffer.bytes;
}
