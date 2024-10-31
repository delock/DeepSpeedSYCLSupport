#ifndef _COLL_MPI__HPP_
#define _COLL_MPI__HPP_

#include <torch/extension.h>

void init_mpi(void);
void init_mpi_thread_comms(void);
void mpi_all_reduce(int world_size, int rank, void* buf, size_t data_size, size_t numel, c10::ScalarType scalar_type);

#endif //_COLL_MPI__HPP_
