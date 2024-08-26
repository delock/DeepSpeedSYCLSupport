#include <mpi.h>
#include "coll_mpi.hpp"

void init_mpi(void)
{
    MPI_Init(NULL, NULL);

    //int size, rank;
    //MPI_Comm_size(MPI_COMM_WORLD, &size);
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

char temp_buf[64*1024*1024];

/*
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
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            //naive_all_reduce(world_size, rank, buf, data_size, numel, scalar_type);
            //ring_all_reduce(world_size, rank, buf, data_size, numel, scalar_type);
            //rabenseifner_all_reduce(world_size, rank, buf, data_size, numel, scalar_type);
            break;
        case c10::ScalarType::Float:
            MPI_Allreduce(MPI_IN_PLACE, buf, numel, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            //naive_all_reduce(world_size, rank, buf, data_size, numel, scalar_type);
            //ring_all_reduce(world_size, rank, buf, data_size, numel, scalar_type);
            //rabenseifner_all_reduce(world_size, rank, buf, data_size, numel, scalar_type);
            break;
        default: assert(!"Should not get here");
    }
}

