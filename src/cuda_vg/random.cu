#include "random.cuh"
#include "util.cuh"

#include <curand_uniform.h>

__global__ void init_rng_kernel(unsigned long seed, int n, curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    curand_init(seed, id, 0, &state[id]);
}

__global__ void gamma_kernel(float *x, int n, float a, curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    if (a == 1.0f) {
        x[id] = device::exponential(&state[id]);
    } else if (a < 1.0f) {
        x[id] = device::jonhk_gamma(a, &state[id]);
    } else {
        x[id] = device::best_gamma(a, &state[id]);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
    CudaRNG* cuda_init_rng(unsigned long seed, int n) {
        CudaRNG* h_state = (CudaRNG*)malloc(sizeof(CudaRNG));

        h_state->n = n;
        CUDA_CHECK(cudaMalloc(&h_state->states, n * sizeof(curandState)));

        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        init_rng_kernel<<<blocks, threadsPerBlock>>>(seed, n, h_state->states);

        CUDA_CHECK(cudaPeekAtLastError());
        #ifdef DEBUG
            CUDA_CHECK(cudaDeviceSynchronize());
        #endif

        return h_state;
    }

    void cuda_cleanup_rng(CudaRNG* state) {
        cudaFree(state->states);
        free(state);
    }

    void cuda_gamma(float *x, int n, float a, CudaRNG* state) {
        if (n > state->n) return;

        if (a < 0.002572f) {
            fprintf(stderr, "WARNING : Parameter 'a' (%f) is below safe threshold for Johnk's method. See inspect_min_safe_gamma_parameter().\n", a);
        }

        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        gamma_kernel<<<blocks, threadsPerBlock>>>(x, n, a, state->states);

        CUDA_CHECK(cudaPeekAtLastError());
        #ifdef DEBUG
            CUDA_CHECK(cudaDeviceSynchronize());
        #endif
    }
#ifdef __cplusplus
}
#endif