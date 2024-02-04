#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/extension.h>

void shm_allreduce(torch::Tensor& t_in, py::object op, bool async_op);
