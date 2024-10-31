// 1 = handoff, 0 = thread-split
#define HANDOFF 0

#include <mpi.h>
#include "coll_mpi.hpp"

#if !HANDOFF // THREAD_SPLIT
#include <omp.h>
std::vector<MPI_Comm> thread_comm;
static bool thread_comm_inited = false;
#endif

void init_mpi(void)
{
    int mpi_inited;
    MPI_Initialized(&mpi_inited);
    if (!mpi_inited) {
#if HANDOFF
        MPI_Init(NULL, NULL);
#else // THREAD_SPLIT
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
#endif
    }
    init_mpi_thread_comms();
}

void init_mpi_thread_comms(void)
{
#if !HANDOFF
    if (!thread_comm_inited) {
	MPI_Info info;
	char s[16];
	int num_threads = omp_get_max_threads();
	thread_comm.resize(num_threads);
	for (int i = 0; i < num_threads; i++) {
	    MPI_Comm_dup(MPI_COMM_WORLD, &thread_comm[i]);
	    snprintf(s, 16, "%d", i);
	    MPI_Info_create(&info);
	    MPI_Info_set(info, "thread_id", s);
	    MPI_Comm_set_info(thread_comm[i], info);
	    MPI_Info_free(&info);
	}
	thread_comm_inited = true;
    }
#endif
}

/*
char temp_buf[64*1024*1024];

void naive_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
    if (rank == 0) {
        // collect input buf from all other ranks
        MPI_Request *req = (MPI_Request*)malloc(sizeof(MPI_Request)*(world_size-1));
        for (int i=1; i<world_size; i++) {
            MPI_Irecv(temp_buf+data_size*(i-1), data_size, MPI_BYTE, i, 0, MPI_COMM_WORLD, req+i-1);
        }
        for (int i=1; i<world_size; i++) {
            MPI_Wait(req+i-1, MPI_STATUS_IGNORE);
        }

        // reduce all bufs together
        switch (scalar_type) {
            case c10::ScalarType::BFloat16:
                for (int i = 1; i < world_size; i++) {
                    reduce_2_bf16_buffers(numel, buf, temp_buf+data_size*(i-1));
                }
                break;
            case c10::ScalarType::Float:
                for (int i = 1; i < world_size; i++) {
                    reduce_2_fp32_buffers(numel, buf, temp_buf+data_size*(i-1));
                }
                break;
            default: assert(!"Should not get here");
        }

        // send result to all other ranks
        for (int i=1; i<world_size; i++) {
            MPI_Isend(buf, data_size, MPI_BYTE, i, 0, MPI_COMM_WORLD, req+i-1);
        }
        for (int i=1; i<world_size; i++) {
            MPI_Wait(req+i-1, MPI_STATUS_IGNORE);
        }
        free(req);
    } else {
        MPI_Request req;
        MPI_Isend(buf, data_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Irecv(buf, data_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

void rabenseifner_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
    size_t chunk_el = numel/world_size;
    size_t chunk_size = data_size/world_size;
    // reduce scatter
    int recv_chunk_idx = 0;
    int chunk_num = world_size / 2;
    int iter = 0;
    for (int rank_stride = 1; rank_stride<world_size; rank_stride*=2) {
        int dest_rank = rank ^ rank_stride;

        // compute chunk to be send
        recv_chunk_idx = (recv_chunk_idx << 1) | ((rank >> iter) & 0x1);
        int send_chunk_idx = recv_chunk_idx ^ 0x1;

        char* send_buf = (char*)buf + send_chunk_idx * chunk_num * chunk_size;
        char* recv_buf = (char*)buf + recv_chunk_idx * chunk_num * chunk_size;
        MPI_Request req_send;
        MPI_Isend(send_buf, chunk_num * chunk_size, MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD, &req_send);
        MPI_Recv(temp_buf, chunk_num * chunk_size, MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        switch (scalar_type) {
            case c10::ScalarType::BFloat16:
                reduce_2_bf16_buffers(chunk_num * chunk_el, recv_buf, temp_buf);
                break;
            case c10::ScalarType::Float:
                reduce_2_fp32_buffers(chunk_num * chunk_el, recv_buf, temp_buf);
                break;
            default: assert(!"Should not get here");
        }
        MPI_Wait(&req_send, MPI_STATUS_IGNORE);

        chunk_num /= 2;
        iter++;
    }

    chunk_num = 1;
    // allgather
    for (int rank_stride = world_size/2; rank_stride>=1; rank_stride/=2) {
        int dest_rank = rank ^ rank_stride;

        // compute chunk to be send
        int send_chunk_idx = recv_chunk_idx ^ 0x1;
        char* send_buf = (char*)buf + send_chunk_idx * chunk_num * chunk_size;
        char* recv_buf = (char*)buf + recv_chunk_idx * chunk_num * chunk_size;
        MPI_Request req_send;
        MPI_Isend(recv_buf, chunk_num * chunk_size, MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD, &req_send);
        MPI_Recv(send_buf, chunk_num * chunk_size, MPI_BYTE, dest_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&req_send, MPI_STATUS_IGNORE);

        recv_chunk_idx = recv_chunk_idx >> 1;
        chunk_num *= 2;
    }
}

void ring_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
    size_t chunk_el = (numel+world_size-1) / world_size;
    size_t chunk_size = chunk_el * (data_size / numel);
    int send_rank = (rank + 1) % world_size;
    int recv_rank = (rank - 1 + world_size) % world_size;

    // reduce scatter
    int send_chunk = rank;
    for (int stage=0; stage<world_size-1; stage++) {
        MPI_Request req_send;
        int recv_chunk = (world_size + send_chunk - 1) % world_size;

        MPI_Isend((char*)buf+send_chunk*chunk_size, chunk_size, MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &req_send);
        MPI_Recv(temp_buf, chunk_size, MPI_BYTE, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        switch (scalar_type) {
            case c10::ScalarType::BFloat16:
                reduce_2_bf16_buffers(chunk_el, (char*)buf+recv_chunk*chunk_size, temp_buf);
                break;
            case c10::ScalarType::Float:
                reduce_2_fp32_buffers(chunk_el, (char*)buf+recv_chunk*chunk_size, temp_buf);
                break;
            default: assert(!"Should not get here");
        }
        MPI_Wait(&req_send, MPI_STATUS_IGNORE);

        send_chunk = recv_chunk;
    }
    // allgather
    for (int stage=0; stage<world_size-1; stage++) {
        MPI_Request req_send;
        int recv_chunk = (world_size + send_chunk - 1) % world_size;

        MPI_Isend((char*)buf+send_chunk*chunk_size, chunk_size, MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &req_send);
        MPI_Recv((char*)buf+recv_chunk*chunk_size, chunk_size, MPI_BYTE, recv_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Wait(&req_send, MPI_STATUS_IGNORE);

        send_chunk = recv_chunk;
    }
}
*/

void mpi_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
#if HANDOFF
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            MPI_Allreduce(MPI_IN_PLACE, buf, numel, MPIX_C_BF16, MPI_SUM, MPI_COMM_WORLD);
            break;
        case c10::ScalarType::Half:
            MPI_Allreduce(MPI_IN_PLACE, buf, numel, MPIX_C_FLOAT16, MPI_SUM, MPI_COMM_WORLD);
            break;
        case c10::ScalarType::Float:
            MPI_Allreduce(MPI_IN_PLACE, buf, numel, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            break;
        default: assert(!"Should not get here");
    }

#else // THREAD_SPLIT

    // Could tune number of threads for performance based on numel...
    int nthds = std::min((size_t)omp_get_max_threads(), numel);

    #pragma omp parallel for num_threads(nthds) schedule(static) shared(nthds, numel)
    for (int tid = 0; tid < nthds; tid++)
    {
        size_t my_numel = numel / nthds;
        char *my_buf = (char *)buf + (tid * my_numel);
        if (tid == nthds - 1) { // Last thread may have uneven number of elements
            my_numel = numel - (my_numel * (nthds - 1)); // Could balance better...
        }

        switch (scalar_type) {
            case c10::ScalarType::BFloat16:
                MPI_Allreduce(MPI_IN_PLACE, my_buf, my_numel, MPIX_C_BF16, MPI_SUM, thread_comm[tid]);
                break;
            case c10::ScalarType::Half:
                MPI_Allreduce(MPI_IN_PLACE, my_buf, my_numel, MPIX_C_FLOAT16, MPI_SUM, thread_comm[tid]);
                break;
            case c10::ScalarType::Float:
                MPI_Allreduce(MPI_IN_PLACE, my_buf, my_numel, MPI_FLOAT, MPI_SUM, thread_comm[tid]);
                break;
            default: assert(!"Should not get here");
        }
    } // omp parallel for
#endif // THREAD_SPLIT
}

