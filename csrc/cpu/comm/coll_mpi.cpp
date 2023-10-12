#include <mpi.h>
#include "coll_mpi.hpp"

void init_mpi(void)
{
    MPI_Init(NULL, NULL);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

void mpi_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type)
{
    MPI_Allreduce(MPI_IN_PLACE, buf, numel, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

