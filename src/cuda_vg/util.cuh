#ifndef UTIL_CUH
#define UTIL_CUH

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand_uniform.h>
#include <stdio.h>

#define DEBUG true

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif