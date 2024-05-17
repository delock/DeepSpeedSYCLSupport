// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include <ATen/ATen.h>
#include <fcntl.h>
#include <immintrin.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <omp.h>
#include "shm.h"

#define DO_PROFILE
#ifdef DO_PROFILE
#include <cfloat>
#include <chrono>
#endif

// states for collectives
enum coll_state {
    coll_begin = 0,
    coll_allreduce_naive__copy_in_done,
    coll_allreduce_naive__reduce_done,
    coll_allreduce_naive__copy_out_done,
    // alternative state when allreduce is working on alternative buffer
    // of the double buffer.
    coll_alt1_allreduce_naive__copy_in_done,
    coll_alt2_allreduce_naive__copy_in_done,
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
        if (errno != ENOENT) {
            // don't print if shm can not be found because we want to loop over from
            // caller again until the other ranks created the shm
            printf("shared_open %s failed, errno=%d\n", name, errno);
        }
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

#define MAX_OMP_THREAD_NUM 8
// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
    enum coll_state state0; // state for naive_all_reduce
    enum coll_state state1; // state for distributed_naive_all_reduce
    enum coll_state state2[MAX_OMP_THREAD_NUM*64]; // state for serial_all_reduce
    sem_t mutex;
    sem_t turnstile1;
    sem_t turnstile2;
    int counter;
    // double buffer to avoid syncing between rounds
    // offset=0 -- 2*NAIVE_ALLREDUCE_THRESHOLD : buffer for naive_allreduce
    // after that : buffer for distributed_naive_all_reduce
    char buffer[2*NAIVE_ALLREDUCE_THRESHOLD + MAX_BUF_SIZE];
};

#define BUFFER0_OFFSET(current_buffer) current_buffer*NAIVE_ALLREDUCE_THRESHOLD
#define BUFFER1_OFFSET 2*NAIVE_ALLREDUCE_THRESHOLD

static int current_buffer = 0;

struct allreduce_workspace** workspace;

void wait_buffer_state0_until_2(int index, enum coll_state state0,
                               enum coll_state state1)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state0);

    while (1) {
        volatile enum coll_state cur_state = *state_ptr;
        if (cur_state == state0 || cur_state == state1)
            break;
    }
}

void wait_buffer_state2_until_2(int index, int state_number, enum coll_state state0,
                               enum coll_state state1)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state2[state_number*64]);

    while (1) {
        volatile enum coll_state cur_state = *state_ptr;
        if (cur_state == state0 || cur_state == state1)
            break;
    }
}

void wait_buffer_state1_until(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state1);

    while (*state_ptr != state)
        ;
}

void wait_buffer_state1_until_not(int index, enum coll_state state)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state1);

    while (*state_ptr == state)
        ;
}

void wait_buffer_state1_until_range(int index, enum coll_state start, int size)
{
    volatile enum coll_state* state_ptr = &(workspace[index]->state1);
    enum coll_state end = (enum coll_state)(start + size);

    while (1) {
        volatile enum coll_state cur_state = *state_ptr;
        if (cur_state >= start and cur_state < end) break;
    }
}

void barrier_wait(int root_idx, int num_ranks)
{
    // Phase 1: Wait for all threads to enter the barrier
    auto shared = workspace[root_idx];
    sem_wait(&shared->mutex);
    shared->counter++;
    if (shared->counter == num_ranks) {
        for (int i = 0; i < num_ranks; ++i) { sem_post(&shared->turnstile1); }
    }
    sem_post(&shared->mutex);
    sem_wait(&shared->turnstile1);

    // Phase 2: Wait for all threads to exit the barrier
    sem_wait(&shared->mutex);
    shared->counter--;
    if (shared->counter == 0) {
        for (int i = 0; i < num_ranks; ++i) { sem_post(&shared->turnstile2); }
    }
    sem_post(&shared->mutex);
    sem_wait(&shared->turnstile2);
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

void serial_reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));
void reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));

void serial_reduce_bf16_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
    __attribute__((target("avx512bw")));
void reduce_bf16_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
    __attribute__((target("avx512bw")));

void serial_reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));
void reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
    __attribute__((target("avx512bw")));

void serial_reduce_fp32_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
    __attribute__((target("avx512bw")));
void reduce_fp32_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
    __attribute__((target("avx512bw")));

// N_REDUCE_LIMIT is the number of buffers that can be reduced together in one shot.
// Compared with do N-1 2-reduces which needs 2*(N-1) read and N-1 write,
// N-reduce only needs N read and 1 write, this saves 2/3 memory bandwidth.
// When increase N_REDUCE_LIMIT to a bigger number, do the following steps
// 1. Extend REPEAT_<X> macros list down below
// 2. Extend switch cases which call "REPEAT(X, ...)" down below
#define N_REDUCE_LIMIT 16

void serial_reduce_all_buffers(struct allreduce_workspace** workspace,
                        int start_elements,
                        int num_elements,
                        c10::ScalarType scalar_type,
                        int num_buffers,
                        int to_buffer_idx,
                        char* to_buffer,
                        size_t buffer_offset)
{
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                serial_reduce_bf16_buffers(
                    start_elements, num_elements, num_buffers, to_buffer, buffer_offset, workspace);
            } else {
                for (int i = 0; i < num_buffers; i++) {
                    if (i == to_buffer_idx) continue;
                    serial_reduce_2_bf16_buffers_iio(
                        num_elements,
                        workspace[i]->buffer + buffer_offset + start_elements * 2,
                        to_buffer + start_elements * 2,
                        to_buffer + start_elements * 2);
                }
            }
            break;
        case c10::ScalarType::Float:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                serial_reduce_fp32_buffers(
                    start_elements, num_elements, num_buffers, to_buffer, buffer_offset, workspace);
            } else {
                for (int i = 0; i < num_buffers; i++) {
                    if (i == to_buffer_idx) continue;
                    serial_reduce_2_fp32_buffers_iio(
                        num_elements,
                        workspace[i]->buffer + buffer_offset + start_elements * 4,
                        to_buffer + start_elements * 4,
                        to_buffer + start_elements * 4);
                }
            }
            break;
        default: assert(!"Should not get here");
    }
}

void reduce_all_buffers(struct allreduce_workspace** workspace,
                        int start_elements,
                        int num_elements,
                        c10::ScalarType scalar_type,
                        int num_buffers,
                        int to_buffer_idx,
                        char* to_buffer,
                        size_t buffer_offset)
{
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                reduce_bf16_buffers(
                    start_elements, num_elements, num_buffers, to_buffer, buffer_offset, workspace);
            } else {
                for (int i = 0; i < num_buffers; i++) {
                    if (i == to_buffer_idx) continue;
                    reduce_2_bf16_buffers_iio(
                        num_elements,
                        workspace[i]->buffer + buffer_offset + start_elements * 2,
                        to_buffer + start_elements * 2,
                        to_buffer + start_elements * 2);
                }
            }
            break;
        case c10::ScalarType::Float:
            if (num_buffers > 2 && num_buffers <= N_REDUCE_LIMIT) {
                reduce_fp32_buffers(
                    start_elements, num_elements, num_buffers, to_buffer, buffer_offset, workspace);
            } else {
                for (int i = 0; i < num_buffers; i++) {
                    if (i == to_buffer_idx) continue;
                    reduce_2_fp32_buffers_iio(
                        num_elements,
                        workspace[i]->buffer + buffer_offset + start_elements * 4,
                        to_buffer + start_elements * 4,
                        to_buffer + start_elements * 4);
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
#define REPEAT_8(x) \
    REPEAT_7(x);    \
    x(8)
#define REPEAT_9(x) \
    REPEAT_8(x);    \
    x(9)
#define REPEAT_10(x) \
    REPEAT_9(x);     \
    x(10)
#define REPEAT_11(x) \
    REPEAT_10(x);    \
    x(11)
#define REPEAT_12(x) \
    REPEAT_11(x);    \
    x(12)
#define REPEAT_13(x) \
    REPEAT_12(x);    \
    x(13)
#define REPEAT_14(x) \
    REPEAT_13(x);    \
    x(14)
#define REPEAT_15(x) \
    REPEAT_14(x);    \
    x(15)

#define CVT_ADD_BF16(x)                                                                 \
    do {                                                                                \
        auto in##x##_val =                                                              \
            cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(workspace[x]->buffer + buffer_offset + i))); \
        inout_val = _mm512_add_ps(inout_val, in##x##_val);                              \
    } while (0)

// Reduce functions down below use vectorized algorithm, the number of bytes processed each
// iteration depends on vector length.  256bit vector ==> 32 bytes, 512bit vector ==> 64 bytes
// If you change implementation of reduce_2_bf16_buffers_iio or reduce_2_fp32_buffers_iio, check
// whether this number needs to be changed
#define VECTOR_LENGTH_IN_BYTES 32

void serial_reduce_bf16_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(workspace[0]->buffer + buffer_offset + i)));
        switch (num_buffers) {
            case 16: REPEAT(15, CVT_ADD_BF16); break;
            case 15: REPEAT(14, CVT_ADD_BF16); break;
            case 14: REPEAT(13, CVT_ADD_BF16); break;
            case 13: REPEAT(12, CVT_ADD_BF16); break;
            case 12: REPEAT(11, CVT_ADD_BF16); break;
            case 11: REPEAT(10, CVT_ADD_BF16); break;
            case 10: REPEAT(9, CVT_ADD_BF16); break;
            case 9: REPEAT(8, CVT_ADD_BF16); break;
            case 8: REPEAT(7, CVT_ADD_BF16); break;
            case 7: REPEAT(6, CVT_ADD_BF16); break;
            case 6: REPEAT(5, CVT_ADD_BF16); break;
            case 5: REPEAT(4, CVT_ADD_BF16); break;
            case 4: REPEAT(3, CVT_ADD_BF16); break;
            case 3: REPEAT(2, CVT_ADD_BF16); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_si256((__m256i*)(to_buffer + i),
                            cvt_fp32_to_bf16(inout_val));
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < num_buffers; j++) { val += *(at::BFloat16*)(workspace[j]->buffer + buffer_offset + i); }
        *(at::BFloat16*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

void reduce_bf16_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(workspace[0]->buffer + buffer_offset + i)));
        switch (num_buffers) {
            case 16: REPEAT(15, CVT_ADD_BF16); break;
            case 15: REPEAT(14, CVT_ADD_BF16); break;
            case 14: REPEAT(13, CVT_ADD_BF16); break;
            case 13: REPEAT(12, CVT_ADD_BF16); break;
            case 12: REPEAT(11, CVT_ADD_BF16); break;
            case 11: REPEAT(10, CVT_ADD_BF16); break;
            case 10: REPEAT(9, CVT_ADD_BF16); break;
            case 9: REPEAT(8, CVT_ADD_BF16); break;
            case 8: REPEAT(7, CVT_ADD_BF16); break;
            case 7: REPEAT(6, CVT_ADD_BF16); break;
            case 6: REPEAT(5, CVT_ADD_BF16); break;
            case 5: REPEAT(4, CVT_ADD_BF16); break;
            case 4: REPEAT(3, CVT_ADD_BF16); break;
            case 3: REPEAT(2, CVT_ADD_BF16); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_si256((__m256i*)(to_buffer + i),
                            cvt_fp32_to_bf16(inout_val));
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < num_buffers; j++) { val += *(at::BFloat16*)(workspace[j]->buffer + buffer_offset + i); }
        *(at::BFloat16*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

void serial_reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
    for (int i = 0; i < main_elements * element_size; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in0 + i)));
        auto in1_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in1 + i)));
        auto out_val = _mm512_add_ps(in0_val, in1_val);
        _mm256_storeu_si256((__m256i*)((char*)out + i), cvt_fp32_to_bf16(out_val));
    }

    // process remaining part
    int i = main_elements * element_size;
    while (remain_elements > 0) {
        float in0_val = *((at::BFloat16*)((char*)in0 + i));
        float in1_val = *((at::BFloat16*)((char*)in1 + i));
        *((at::BFloat16*)((char*)out + i)) = in0_val + in1_val;
        remain_elements--;
        i += element_size;
    }
}

void reduce_2_bf16_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
    const int element_size = 2;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < main_elements * element_size; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in0 + i)));
        auto in1_val = cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)((char*)in1 + i)));
        auto out_val = _mm512_add_ps(in0_val, in1_val);
        _mm256_storeu_si256((__m256i*)((char*)out + i), cvt_fp32_to_bf16(out_val));
    }

    // process remaining part
    int i = main_elements * element_size;
    while (remain_elements > 0) {
        float in0_val = *((at::BFloat16*)((char*)in0 + i));
        float in1_val = *((at::BFloat16*)((char*)in1 + i));
        *((at::BFloat16*)((char*)out + i)) = in0_val + in1_val;
        remain_elements--;
        i += element_size;
    }
}

#define CVT_ADD_F32(x)                                                          \
    do {                                                                        \
        auto in##x##_val = _mm256_loadu_ps((float*)(workspace[x]->buffer + buffer_offset + i)); \
        inout_val = _mm256_add_ps(inout_val, in##x##_val);                      \
    } while (0)

void serial_reduce_fp32_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
{
    const int element_size = 4;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)(workspace[0]->buffer + buffer_offset + i));
        switch (num_buffers) {
            case 16: REPEAT(15, CVT_ADD_F32); break;
            case 15: REPEAT(14, CVT_ADD_F32); break;
            case 14: REPEAT(13, CVT_ADD_F32); break;
            case 13: REPEAT(12, CVT_ADD_F32); break;
            case 12: REPEAT(11, CVT_ADD_F32); break;
            case 11: REPEAT(10, CVT_ADD_F32); break;
            case 10: REPEAT(9, CVT_ADD_F32); break;
            case 9: REPEAT(8, CVT_ADD_F32); break;
            case 8: REPEAT(7, CVT_ADD_F32); break;
            case 7: REPEAT(6, CVT_ADD_F32); break;
            case 6: REPEAT(5, CVT_ADD_F32); break;
            case 5: REPEAT(4, CVT_ADD_F32); break;
            case 4: REPEAT(3, CVT_ADD_F32); break;
            case 3: REPEAT(2, CVT_ADD_F32); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_ps((float*)(to_buffer + i), inout_val);
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < num_buffers; j++) { val += *(float*)(workspace[j]->buffer + buffer_offset + i); }
        *(float*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

void reduce_fp32_buffers(int start_elements,
                         int num_elements,
                         int num_buffers,
                         char* to_buffer,
                         size_t buffer_offset,
                         struct allreduce_workspace** workspace)
{
    const int element_size = 4;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = start_elements * element_size; i < (start_elements + main_elements) * element_size;
         i += VECTOR_LENGTH_IN_BYTES) {
        auto inout_val = _mm256_loadu_ps((float*)(workspace[0]->buffer + buffer_offset + i));
        switch (num_buffers) {
            case 16: REPEAT(15, CVT_ADD_F32); break;
            case 15: REPEAT(14, CVT_ADD_F32); break;
            case 14: REPEAT(13, CVT_ADD_F32); break;
            case 13: REPEAT(12, CVT_ADD_F32); break;
            case 12: REPEAT(11, CVT_ADD_F32); break;
            case 11: REPEAT(10, CVT_ADD_F32); break;
            case 10: REPEAT(9, CVT_ADD_F32); break;
            case 9: REPEAT(8, CVT_ADD_F32); break;
            case 8: REPEAT(7, CVT_ADD_F32); break;
            case 7: REPEAT(6, CVT_ADD_F32); break;
            case 6: REPEAT(5, CVT_ADD_F32); break;
            case 5: REPEAT(4, CVT_ADD_F32); break;
            case 4: REPEAT(3, CVT_ADD_F32); break;
            case 3: REPEAT(2, CVT_ADD_F32); break;
            default: assert(!"Should not get here.");
        }
        _mm256_storeu_ps((float*)(to_buffer + i), inout_val);
    }

    // process remaining part
    int i = (start_elements + main_elements) * element_size;
    while (remain_elements > 0) {
        float val = 0.0f;
        for (int j = 0; j < num_buffers; j++) { val += *(float*)(workspace[j]->buffer + buffer_offset + i); }
        *(float*)(to_buffer + i) = val;
        remain_elements--;
        i += element_size;
    }
}

void serial_reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
    const int element_size = 4;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
    for (int i = 0; i < main_elements * element_size; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = _mm256_loadu_ps((float*)((char*)in0 + i));
        auto in1_val = _mm256_loadu_ps((float*)((char*)in1 + i));
        auto out_val = _mm256_add_ps(in0_val, in1_val);
        _mm256_storeu_ps((float*)((char*)out + i), out_val);
    }

    // process remaining part
    int i = main_elements * element_size;
    while (remain_elements > 0) {
        float in0_val = *((float*)((char*)in0 + i));
        float in1_val = *((float*)((char*)in1 + i));
        *((float*)((char*)out + i)) = in0_val + in1_val;
        remain_elements--;
        i += element_size;
    }
}

void reduce_2_fp32_buffers_iio(int num_elements, void* in0, void* in1, void* out)
{
    const int element_size = 4;
    const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
    int main_elements = num_elements - (num_elements % vector_length);
    int remain_elements = num_elements % vector_length;

    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < main_elements * element_size; i += VECTOR_LENGTH_IN_BYTES) {
        auto in0_val = _mm256_loadu_ps((float*)((char*)in0 + i));
        auto in1_val = _mm256_loadu_ps((float*)((char*)in1 + i));
        auto out_val = _mm256_add_ps(in0_val, in1_val);
        _mm256_storeu_ps((float*)((char*)out + i), out_val);
    }

    // process remaining part
    int i = main_elements * element_size;
    while (remain_elements > 0) {
        float in0_val = *((float*)((char*)in0 + i));
        float in1_val = *((float*)((char*)in1 + i));
        *((float*)((char*)out + i)) = in0_val + in1_val;
        remain_elements--;
        i += element_size;
    }
}

static bool is_initialized = 0;
static int world_size;
static int world_rank;

void shm_initialize(int size, int rank, char* addr_string, char* port_string)
{
    if (is_initialized) return;
    is_initialized = 1;

    world_size = size;
    world_rank = rank;

    char shm_name_prefix[NAME_BUF_SIZE];
    char shm_name[NAME_BUF_SIZE];
    snprintf(shm_name_prefix,
             NAME_BUF_SIZE,
             "%s_%d_%s_%s",
             SHM_BUFFER_NAME,
             getuid(),
             addr_string,
             port_string);
    // create shared workspace for SHM based allreduce
    SharedData allreduce_buffer;
    // allocate workspace_buf for current rank
    struct allreduce_workspace* workspace_buf;
    struct allreduce_workspace* workspace_buf_other;
    workspace_buf = (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
    snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, rank);
    shared_create(&allreduce_buffer, shm_name, workspace_buf, sizeof(struct allreduce_workspace));
    workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
    workspace_buf->state0 = coll_alt2_allreduce_naive__copy_in_done;
    for (int i=0; i<MAX_OMP_THREAD_NUM; i++) {
        workspace_buf->state2[i*64] = coll_alt2_allreduce_naive__copy_in_done;
    }
    workspace_buf->state1 = coll_begin;

    // create the workspace pointer list
    workspace = (struct allreduce_workspace**)malloc(size * sizeof(struct allreduce_workspace*));

    // map shm of all ranks
    for (int i = 0; i < size; i++) {
        if (i != rank) {
            snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, i);
            // printf("open %s, %d\n", shm_name, rank);
            do {
                shared_open(&allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
            } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
            workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
            workspace[i] = workspace_buf_other;
        } else {
            workspace[i] = workspace_buf;
            workspace_buf->counter = 0;
            sem_init(&workspace_buf->mutex, 1, 1);
            sem_init(&workspace_buf->turnstile1, 1, 0);
            sem_init(&workspace_buf->turnstile2, 1, 0);
        }
    }
}

static void parallel_memcpy(void* to, void* from, size_t n_bytes)
    __attribute__((target("avx512bw")));
static void parallel_memcpy(void* to, void* from, size_t n_bytes)
{
    auto aligned_bytes = n_bytes - (n_bytes % VECTOR_LENGTH_IN_BYTES);
    // process aligned part
#pragma omp parallel for
    for (int i = 0; i < aligned_bytes; i += VECTOR_LENGTH_IN_BYTES) {
        auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
        _mm256_storeu_si256((__m256i*)((char*)to + i), val);
    }

    // process remaining part
    for (int i = aligned_bytes; i < n_bytes; i++) { *((char*)to + i) = *((char*)from + i); }
}

static void serial_memcpy(void* to, void* from, size_t n_bytes)
    __attribute__((target("avx512bw")));
static void serial_memcpy(void* to, void* from, size_t n_bytes)
{
    auto aligned_bytes = n_bytes - (n_bytes % VECTOR_LENGTH_IN_BYTES);
    // process aligned part
    for (int i = 0; i < aligned_bytes; i += VECTOR_LENGTH_IN_BYTES) {
        auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
        _mm256_storeu_si256((__m256i*)((char*)to + i), val);
    }

    // process remaining part
    for (int i = aligned_bytes; i < n_bytes; i++) { *((char*)to + i) = *((char*)from + i); }
}

size_t slice_size(size_t chunk_el, int slice_idx);
char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx);
static void parallel_multi_memcpy(int world_size,
                                  char* data_ptr,
                                  size_t chunk_el,
                                  size_t chunk_size,
                                  int data_size,
                                  size_t buffer_offset)
{
    for (int rank = 0; rank < world_size; rank++) {
        parallel_memcpy(slice_data(data_ptr, chunk_el, data_size, rank),
                        slice_data(workspace[rank]->buffer + buffer_offset, chunk_el, data_size, rank),
                        slice_size(chunk_el, rank) * data_size);
    }
}

#define positive_mod(num, mod) ((((num) % (mod)) + (mod)) % (mod))
#define rank_mod(rank) positive_mod(rank, world_size)
size_t slice_size(size_t chunk_el, int slice_idx)
{
    size_t slice_size = chunk_el / world_size;
    return slice_idx == world_size - 1 ? slice_size + (chunk_el % world_size) : slice_size;
}

char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx)
{
    size_t slice_size = chunk_el / world_size;
    size_t el_offset = slice_size * slice_idx;
    return data_ptr + el_offset * el_size;
}

size_t slice_el_start(size_t chunk_el, int slice_idx)
{
    size_t slice_size = chunk_el / world_size;
    return slice_size * slice_idx;
}

/*
    serial all_reduce
    step 0: before enter the function ith times, state is copy(i-1)
    step 1: each rank copy data from input (data_ptr) to SHM buffer[i]
    step 2: set own state to copy(i)
    step 3: wait each other rank's state equal or later than copy(i)
    step 4: reduce across SHM buffer(ith) directly into output (data_ptr)
*/
void serial_naive_all_reduce(char* data_ptr,
                      c10::ScalarType scalar_type,
                      size_t type_size,
                      size_t chunk_el,
                      int state_number, size_t offset)
{
    static int state_idx[MAX_OMP_THREAD_NUM*64] = {0};

    /*
        We can't have infinite number of buffers and states.  2 sets of buffer
        and 3 sets of states is just enough.  Consider current rank is in step 3,
        with it's own state set to copy(i), the other rank will them have the
        following situations:
        ------------------------------------------------
        my state | can I proceed? | the other rank state
        ================================================
                 |       N        | copy(i-1)
                 |----------------|---------------------
        copy(i)  |       Y        | copy(i)
                 |----------------|---------------------
                 |       Y        | copy(i+1)
        ------------------------------------------------
        * When I have state as copy(i), the other rank cannot have state
          begin(i-1) or before, in that case I'll be in state copy(i-1).
        * The other rank cannot have state begin(i+2) or beyond because my
          state is still copy(i), this is as far as the other rank could go
        * From a rank's POV, all the other ranks can be divided into three sets:
          - Lagging ranks: ranks that are still working on previous iteration
          - Syncing ranks: ranks that are working on current iteration
          - Leading ranks: ranks that are working on next iteration
        * We can have 3 sets of states, one set for syncing ranks; one set for
          lagging ranks; one set of leading ranks.  With 3 sets of states, we can
          distinguish between lagging and leading ranks.
        * Note from any rank's POV, leading ranks and lagging ranks does not
          appear at the same time.  Either all other ranks are syncing or
          lagging, or all other ranks are syncing or leading.
        * So we have 2 sets of buffers, one buffer is used by current iter;
          one buffer used by either lagging ranks or leading ranks.
    */
    auto chunk_size = type_size * chunk_el;
    enum coll_state begin_next,
                    copy_prev, copy_current, copy_next;
    switch (state_idx[state_number*64]) {
    case 0:
        copy_prev =     coll_alt2_allreduce_naive__copy_in_done;
        copy_current =  coll_allreduce_naive__copy_in_done;
        copy_next =     coll_alt1_allreduce_naive__copy_in_done;
        break;
    case 1:
        copy_prev =     coll_allreduce_naive__copy_in_done;
        copy_current =  coll_alt1_allreduce_naive__copy_in_done;
        copy_next =     coll_alt2_allreduce_naive__copy_in_done;
        break;
    case 2:
        copy_prev =     coll_alt1_allreduce_naive__copy_in_done;
        copy_current =  coll_alt2_allreduce_naive__copy_in_done;
        copy_next =     coll_allreduce_naive__copy_in_done;
        break;
    default:
        assert (!"Should not get here.");
    }
    state_idx[state_number*64] = (state_idx[state_number*64] + 1) % 3;

    serial_memcpy(workspace[world_rank]->buffer + BUFFER0_OFFSET(current_buffer) + offset, data_ptr, chunk_size);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state2[state_number*64] = copy_current;

    for (int i = 0; i < world_size; i++) {
        // wait until the other rank copy the buffer
        if (i != world_rank)
            wait_buffer_state2_until_2(i, state_number, copy_current, copy_next);
    }

    // each rank reduce the buffer independently so therre is no need for synchronization afterward
    reduce_all_buffers(workspace, 0, chunk_el, scalar_type, world_size, world_rank, data_ptr, BUFFER0_OFFSET(current_buffer)+offset);
    std::atomic_thread_fence(std::memory_order_release);

    // switch buffer
    current_buffer = 1-current_buffer;
}

/*
    Symmetrical naive all_reduce
    step 0: before enter the function ith times, state is copy(i-1)
    step 1: each rank copy data from input (data_ptr) to SHM buffer[i]
    step 2: set own state to copy(i)
    step 3: wait each other rank's state equal or later than copy(i)
    step 4: reduce across SHM buffer(ith) directly into output (data_ptr)
*/
void naive_all_reduce(char* data_ptr,
                      c10::ScalarType scalar_type,
                      size_t chunk_size,
                      size_t chunk_el)
{
    static int state_idx = 0;
#ifdef DO_PROFILE
    static double total_t1_t0 = 0.0;
    static double total_t2_t1 = 0.0;
    static double total_t3_t2 = 0.0;
    static double total_t4_t3 = 0.0;
    static double total_t5_t4 = 0.0;
    static int count = -16;  // warmup
    auto t0 = std::chrono::system_clock::now();
#endif

    /*
        We can't have infinite number of buffers and states.  2 sets of buffer
        and 3 sets of states is just enough.  Consider current rank is in step 3,
        with it's own state set to copy(i), the other rank will them have the
        following situations:
        ------------------------------------------------
        my state | can I proceed? | the other rank state
        ================================================
                 |       N        | copy(i-1)
                 |----------------|---------------------
        copy(i)  |       Y        | copy(i)
                 |----------------|---------------------
                 |       Y        | copy(i+1)
        ------------------------------------------------
        * When I have state as copy(i), the other rank cannot have state
          begin(i-1) or before, in that case I'll be in state copy(i-1).
        * The other rank cannot have state begin(i+2) or beyond because my
          state is still copy(i), this is as far as the other rank could go
        * From a rank's POV, all the other ranks can be divided into three sets:
          - Lagging ranks: ranks that are still working on previous iteration
          - Syncing ranks: ranks that are working on current iteration
          - Leading ranks: ranks that are working on next iteration
        * We can have 3 sets of states, one set for syncing ranks; one set for
          lagging ranks; one set of leading ranks.  With 3 sets of states, we can
          distinguish between lagging and leading ranks.
        * Note from any rank's POV, leading ranks and lagging ranks does not
          appear at the same time.  Either all other ranks are syncing or
          lagging, or all other ranks are syncing or leading.
        * So we have 2 sets of buffers, one buffer is used by current iter;
          one buffer used by either lagging ranks or leading ranks.
    */
    enum coll_state begin_next,
                    copy_prev, copy_current, copy_next;
    switch (state_idx) {
    case 0:
        copy_prev =     coll_alt2_allreduce_naive__copy_in_done;
        copy_current =  coll_allreduce_naive__copy_in_done;
        copy_next =     coll_alt1_allreduce_naive__copy_in_done;
        break;
    case 1:
        copy_prev =     coll_allreduce_naive__copy_in_done;
        copy_current =  coll_alt1_allreduce_naive__copy_in_done;
        copy_next =     coll_alt2_allreduce_naive__copy_in_done;
        break;
    case 2:
        copy_prev =     coll_alt1_allreduce_naive__copy_in_done;
        copy_current =  coll_alt2_allreduce_naive__copy_in_done;
        copy_next =     coll_allreduce_naive__copy_in_done;
        break;
    default:
        assert (!"Should not get here.");
    }
    state_idx = (state_idx + 1) % 3;

    parallel_memcpy(workspace[world_rank]->buffer + BUFFER0_OFFSET(current_buffer), data_ptr, chunk_size);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state0 = copy_current;

#ifdef DO_PROFILE
    auto t1 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < world_size; i++) {
        // wait until the other rank copy the buffer
        if (i != world_rank)
            wait_buffer_state0_until_2(i, copy_current, copy_next);
    }
#ifdef DO_PROFILE
    auto t2 = std::chrono::system_clock::now();
#endif

    // each rank reduce the buffer independently so therre is no need for synchronization afterward
    reduce_all_buffers(workspace, 0, chunk_el, scalar_type, world_size, world_rank, data_ptr, BUFFER0_OFFSET(current_buffer));
    std::atomic_thread_fence(std::memory_order_release);

    // switch buffer
    current_buffer = 1-current_buffer;

#ifdef DO_PROFILE
    auto t3 = std::chrono::system_clock::now();

    count++;
    if (count > 0) {
        total_t1_t0 += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_t2_t1 += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        total_t3_t2 += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        if (world_rank == 0 && count == 1000) {
            printf("naive_all_reduce time breakdown:\n");
            printf("\tcopy input buffer: %.2f\n", total_t1_t0 / count);
            printf("\twait for copy: %.2f\n", total_t2_t1 / count);
            printf("\treduce: %.2f\n", total_t3_t2 / count);
        }
    }
#endif
}

// naive allreduce distributed, each rank do naive reduce on its slice
void distributed_naive_reduce(char* data_ptr,
                              c10::ScalarType scalar_type,
                              size_t chunk_size,
                              size_t chunk_el)
{
#ifdef DO_PROFILE
    static double total_t1_t0 = 0.0;
    static double total_t2_t1 = 0.0;
    static double total_t3_t2 = 0.0;
    static double total_t4_t3 = 0.0;
    static double total_t5_t4 = 0.0;
    static int count = -16;  // warmup
    auto t0 = std::chrono::system_clock::now();
#endif

    int data_size = chunk_size / chunk_el;
    parallel_memcpy(workspace[world_rank]->buffer + BUFFER1_OFFSET, data_ptr, chunk_size);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state1 = coll_allreduce_naive__copy_in_done;

#ifdef DO_PROFILE
    auto t1 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < world_size; i++) {
        // wait until all the other ranks copy the buffer
        wait_buffer_state1_until_range(i, coll_allreduce_naive__copy_in_done, 2);
    }

#ifdef DO_PROFILE
    auto t2 = std::chrono::system_clock::now();
#endif

    // reduce scatter
    reduce_all_buffers(workspace,
                       slice_el_start(chunk_el, world_rank),
                       slice_size(chunk_el, world_rank),
                       scalar_type,
                       world_size,
                       world_rank,
                       workspace[world_rank]->buffer + BUFFER1_OFFSET,
                       BUFFER1_OFFSET);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state1 = coll_allreduce_naive__reduce_done;

#ifdef DO_PROFILE
    auto t3 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < world_size; i++) {
        int rank = (i + world_rank) % world_size;
        // wait until the other rank reduce the buffer
        wait_buffer_state1_until_range(rank, coll_allreduce_naive__reduce_done, 2);
    }
    parallel_multi_memcpy(world_size, data_ptr, chunk_el, chunk_size, data_size, BUFFER1_OFFSET);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state1 = coll_allreduce_naive__copy_out_done;

#ifdef DO_PROFILE
    auto t4 = std::chrono::system_clock::now();
#endif

    for (int i = 0; i < world_size; i++) {
        wait_buffer_state1_until_not(i, coll_allreduce_naive__reduce_done);
    }

    std::atomic_thread_fence(std::memory_order_release);
    workspace[world_rank]->state1 = coll_begin;

#ifdef DO_PROFILE
    auto t5 = std::chrono::system_clock::now();
    count++;
    if (count > 0) {
        total_t1_t0 += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_t2_t1 += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        total_t3_t2 += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        total_t4_t3 += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        total_t5_t4 += std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
        if (world_rank == 0 && count == 1000) {
            printf("distributed_naive_reduce time breakdown:\n");
            printf("\tcopy input buffer: %.2f\n", total_t1_t0 / count);
            printf("\twait for copy: %.2f\n", total_t2_t1 / count);
            printf("\treduce: %.2f\n", total_t3_t2 / count);
            printf("\tcopy buffer to output: %.2f\n", total_t4_t3 / count);
            printf("\twait finish: %.2f\n", total_t5_t4 / count);
        }
    }
#endif
}

void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size)
{
    for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
        auto data_ptr = ((char*)(data.data_ptr()) + offset);
        size_t chunk_size = data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
        size_t chunk_el = chunk_size / (data_size / numel);
        if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD) {
            #if 0
            naive_all_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
            #else
            int omp_num_threads = omp_get_max_threads();
            if (omp_num_threads > MAX_OMP_THREAD_NUM) omp_num_threads = MAX_OMP_THREAD_NUM;
            size_t sub_chunk_el = chunk_el/omp_num_threads;
            size_t sub_chunk_size = sub_chunk_el * data_size/numel;
            size_t last_chunk_el = sub_chunk_el + chunk_el % omp_num_threads;
#pragma omp parallel for
            for (int i=0; i<omp_num_threads; i++) {
                int tid = omp_get_thread_num();
                serial_naive_all_reduce(data_ptr+tid*sub_chunk_size, data.scalar_type(), data_size/numel, tid==omp_num_threads-1?last_chunk_el:sub_chunk_el, tid, tid*sub_chunk_size);
            }
            #endif
        } else
            distributed_naive_reduce(data_ptr, data.scalar_type(), chunk_size, chunk_el);
    }
}
