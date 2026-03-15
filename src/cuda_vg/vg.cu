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

__global__ void vg_pricing_kernel(
    float *x_mc,
    int n_mc,
    float T,
    float K,
    float sigma,
    float theta,
    float kappa,
    float omega,
    curandState *state
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n_mc) return;

    x_mc[id] = device::vg(T, sigma, theta, kappa, &state[id]);
    x_mc[id] = expf(omega * T + x_mc[id]);
    x_mc[id] = x_mc[id] - K;
    x_mc[id] = fmaxf(x_mc[id], 0.0f);
}

#ifdef __cplusplus
extern "C" {
#endif

    void cuda_vg_pricing(
        float *x_mc,
        float T,
        float K,
        float sigma,
        float theta,
        float kappa,
        int mc_steps,
        CudaRNG* state
    ) {
        if (mc_steps > state->n) return;

        // float* x_mc;
        // CUDA_CHECK(cudaMalloc((void**)&x_mc, mc_steps * sizeof(float)));

        float omega = logf(1.0f - theta * kappa - kappa * sigma * sigma / 2.0f) / kappa;

        int threadsPerBlock = 256;
        int blocks = (mc_steps + threadsPerBlock - 1) / threadsPerBlock;

        vg_pricing_kernel<<<blocks, threadsPerBlock>>>(x_mc, mc_steps, T, K, sigma, theta, kappa, omega, state->states);
        // CUDA_CHECK(cudaDeviceSynchronize());

        // float x_hat_online = 0.f;
        // float x_hat_std_online = 0.f;

        // for (int i = 0; i < mc_steps; i++) {
        //     float x_i = x_mc[i];
        //     float prev_x_hat_online = x_hat_online;
        //     x_hat_online = ((float)i * x_hat_online + x_i) / (float)(i + 1);
        //     x_hat_std_online += (x_i - prev_x_hat_online) * (x_i - x_hat_online);
        // }

        // *x_hat = x_hat_online;

        // if (mc_steps >= 2) {
        //     *x_hat_ic = sqrtf(x_hat_std_online / (float)((mc_steps - 1) * mc_steps));
        // } else {
        //     *x_hat_ic = 0.0f;
        // }

        // CUDA_CHECK(cudaFree(x_mc));

        CUDA_CHECK(cudaPeekAtLastError());
        #ifdef DEBUG
            CUDA_CHECK(cudaDeviceSynchronize());
        #endif
    }

#ifdef __cplusplus
}
#endif
