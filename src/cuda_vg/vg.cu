#include "random.cuh"
#include "util.cuh"

#include <curand_uniform.h>


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

__global__ void _batched_vg_pricing_martingale_constant_kernel(
    float *omega,
    float *sigma,
    float *theta,
    float *kappa,
    int n
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    omega[id] = logf(1.0f - theta[id] * kappa[id] - kappa[id] * sigma[id] * sigma[id] / 2.0f) / kappa[id];
}

__global__ void batched_vg_pricing_kernel(
    float *x_mc,
    float *T,
    float *K,
    float *sigma,
    float *theta,
    float *kappa,
    float *omega,
    int mc_steps,
    int batch_size,
    curandState *state
) {
    /*
     * Note : Swapped for efficiency, samples from a same batch share parameters and
     * should be kept close in memory. The performance gain on my laptop is significant.
     */
    int sample_id = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_id = threadIdx.y + blockIdx.y * blockDim.y;

    if (sample_id >= mc_steps) return;
    if (batch_id >= batch_size) return;

    int id = batch_id * mc_steps + sample_id;

    /*
     * Note : `state` points to global memory, keeping the state local to the thread registers
     * avoids back and forths to global memory.
     */
    curandState local_state = state[id];

    x_mc[id] = device::vg(T[batch_id], sigma[batch_id], theta[batch_id], kappa[batch_id], &local_state);
    x_mc[id] = expf(omega[batch_id] * T[batch_id] + x_mc[id]);
    x_mc[id] = x_mc[id] - K[batch_id];
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

        float* omega;
        CUDA_CHECK(cudaMalloc((void**)&omega, batch_size * sizeof(float)));

        int tpb_omega = 256;
        int blocks_omega = (batch_size + tpb_omega - 1) / tpb_omega;

        _batched_vg_pricing_martingale_constant_kernel<<<blocks_omega, tpb_omega>>>(omega, sigma, theta, kappa, batch_size);

        dim3 threadsPerBlock;
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 1;

        dim3 blocks;
        blocks.x = (mc_steps + threadsPerBlock.x - 1) / threadsPerBlock.x;
        blocks.y = (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y;
        blocks.z = 1;

        batched_vg_pricing_kernel<<<blocks, threadsPerBlock>>>(x_mc, T, K, sigma, theta, kappa, omega, mc_steps, batch_size, state->states);

        CUDA_CHECK(cudaFree(omega));

        CUDA_CHECK(cudaPeekAtLastError());
        #ifdef DEBUG
            CUDA_CHECK(cudaDeviceSynchronize());
        #endif

    }

#ifdef __cplusplus
}
#endif
