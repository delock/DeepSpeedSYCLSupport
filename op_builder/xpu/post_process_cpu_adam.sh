# delete including torch cuda headers
find ./deepspeed/third-party/csrc -name "context.h" -exec sed -Ei "s:#include <ATen/cuda/CUDAContext.h>:// \0:g" {} +

# fix cublas transpos flag to mkl's
find ./deepspeed/third-party/csrc -name "context.h" -exec sed -i "s/CUBLAS_OP_T/oneapi::mkl::transpose::trans/g" {} +
find ./deepspeed/third-party/csrc -name "context.h" -exec sed -i "s/CUBLAS_OP_N/oneapi::mkl::transpose::nontrans/g" {} +
