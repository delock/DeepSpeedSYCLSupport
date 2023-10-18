#include <mpi.h>
#include "coll_shm.hpp"
#include "coll_mpi.hpp"

void init_mpi(void)
{
    //MPI_Init(NULL, NULL);

    //int size, rank;
    //MPI_Comm_size(MPI_COMM_WORLD, &size);
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

void naive_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
    if (rank == 0) {
        // collect input buf from all other ranks
        char *temp_buf = (char*)malloc(data_size*world_size-1);
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
    } else {
        MPI_Request req;
        MPI_Isend(buf, data_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Irecv(buf, data_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

void mpi_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
    naive_all_reduce(world_size, rank, buf, data_size, numel, scalar_type);
    return;
    switch (scalar_type) {
        case c10::ScalarType::BFloat16:
            return;
            break;
        case c10::ScalarType::Float:
            MPI_Allreduce(MPI_IN_PLACE, buf, numel, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            break;
        default: assert(!"Should not get here");
    }
}

