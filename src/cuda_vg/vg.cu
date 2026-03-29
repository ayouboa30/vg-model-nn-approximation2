#include "random.cuh"
#include "util.cuh"

#include <curand_uniform.h>

#define BATCH_BLOCK_SIZE 8

namespace device {
    __device__ inline float vg(    
        float T,
        float sigma,
        float theta,
        float kappa,
        curandState *state
    ) {
        float z = kappa * device::gamma(T / kappa, state);
        return theta * z + sigma * sqrtf(z) * curand_normal(state);
    }
}

__global__ void vg_process_kernel(
    float *x,
    float dt,
    float sigma,
    float theta,
    float kappa,
    int n,
    curandState *state
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    curandState local_state = state[id];

    /*
     * We pass a single dt (and a number of steps) instead of
     * a whole time array to avoid having to reach into global
     * mamory. We could without much trouble pass an array of dt(s)
     * instead. But we don't need that.
     *
     * float dt = t[id] - (id > 0 ? t[id - 1] : 0.0f);
     * 
     */
    float dx = device::vg(dt, sigma, theta, kappa, &local_state);

    x[id] = dx;

    /*
     * Note : We don't do the required cumsum here, because it involves a scan,
     * which we have _some_ confidence implementing on a single thread block,
     * but very little across mutliple ones. We'll do the cumsum with torch.
     */

    state[id] = local_state;
}

__global__ void batched_vg_pricing_kernel(
    float *x_mc,
    float *T,
    float *K,
    float *sigma,
    float *theta,
    float *kappa,
    int mc_steps,
    int batch_size,
    curandState *state
) {
    __shared__ float s_omega[BATCH_BLOCK_SIZE];
    __shared__ float s_T[BATCH_BLOCK_SIZE];
    __shared__ float s_K[BATCH_BLOCK_SIZE];
    __shared__ float s_sigma[BATCH_BLOCK_SIZE];
    __shared__ float s_theta[BATCH_BLOCK_SIZE];
    __shared__ float s_kappa[BATCH_BLOCK_SIZE];

    /*
     * Note : Swapped for efficiency, samples from a same batch share parameters and
     * should be kept close in memory. The performance gain on my laptop is significant.
     */
    int sample_id = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_id = threadIdx.y + blockIdx.y * blockDim.y;
    int id = batch_id * mc_steps + sample_id;

    if (threadIdx.x == 0 && sample_id < mc_steps && batch_id < batch_size) {
        s_T[threadIdx.y] = T[batch_id];
        s_K[threadIdx.y] = K[batch_id];
        s_sigma[threadIdx.y] = sigma[batch_id];
        s_theta[threadIdx.y] = theta[batch_id];
        s_kappa[threadIdx.y] = kappa[batch_id];

        s_omega[threadIdx.y] = logf(1.0f - s_theta[threadIdx.y] * s_kappa[threadIdx.y] - s_kappa[threadIdx.y] * s_sigma[threadIdx.y] * s_sigma[threadIdx.y] / 2.0f) / s_kappa[threadIdx.y];
    }

    __syncthreads();

    if (sample_id >= mc_steps) return;
    if (batch_id >= batch_size) return;

    /*
     * Note : `state` points to global memory, keeping the state local to the thread registers
     * avoids back and forths to global memory.
     */
    curandState local_state = state[id];

    x_mc[id] = device::vg(s_T[threadIdx.y], s_sigma[threadIdx.y], s_theta[threadIdx.y], s_kappa[threadIdx.y], &local_state);
    x_mc[id] = expf(s_omega[threadIdx.y] * s_T[threadIdx.y] + x_mc[id]);
    x_mc[id] = x_mc[id] - s_K[threadIdx.y];
    x_mc[id] = fmaxf(x_mc[id], 0.0f);

    state[id] = local_state;
}

#ifdef __cplusplus
extern "C" {
#endif

    void cuda_vg_process(
        float *x,
        float dt,
        float sigma,
        float theta,
        float kappa,
        int n,
        CudaRNG* state
    ) {
        if (n > state->n) return;

        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        vg_process_kernel<<<blocks, threadsPerBlock>>>(x, dt, sigma, theta, kappa, n, state->states);

        CUDA_CHECK(cudaPeekAtLastError());
        #ifdef DEBUG
            CUDA_CHECK(cudaDeviceSynchronize());
        #endif
    }

    void cuda_batched_vg_pricing(
        float *x_mc,
        float *T,
        float *K,
        float *sigma,
        float *theta,
        float *kappa,
        int batch_size,
        int mc_steps,
        CudaRNG* state
    ) {
        if (mc_steps * batch_size > state->n) return;

        dim3 threadsPerBlock;
        threadsPerBlock.x = 32;
        threadsPerBlock.y = BATCH_BLOCK_SIZE;
        threadsPerBlock.z = 1;

        dim3 blocks;
        blocks.x = (mc_steps + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocks.y = (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y;
        blocks.z = 1;

        batched_vg_pricing_kernel<<<blocks, threadsPerBlock>>>(x_mc, T, K, sigma, theta, kappa, mc_steps, batch_size, state->states);

        CUDA_CHECK(cudaPeekAtLastError());
        #ifdef DEBUG
            CUDA_CHECK(cudaDeviceSynchronize());
        #endif

    }

#ifdef __cplusplus
}
#endif
