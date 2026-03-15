#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>
#include "util.cuh"

typedef struct {
    curandState* states;
    int n;
} CudaRNG;

__global__ void init_rng_kernel(unsigned long seed, int n, curandState *state);

namespace device {
    __device__ inline float exponential(curandState *state) {
        return -logf(curand_uniform(state));
    }

    __device__ inline float jonhk_gamma(float a, curandState *state) {
        float b0 = 1.0f / a;
        float b1 = 1.0f / (1.0f - a);

        for (;;) {
            float u = curand_uniform(state);
            float v = curand_uniform(state);

            float y = powf(u, b0);
            float z = powf(v, b1);

            if (y + z <= 1.0f) {
                float e = device::exponential(state);
                return (y * e) / (y + z);
            }
        }
    }

    __device__ inline float best_gamma(float a, curandState *state) {
        float b = a - 1.0f;
        float c = 3.0f * a - 0.75f;

        for (;;){
            float u = curand_uniform(state);
            float v = curand_uniform(state);

            float w = u * (1.0f - u);
            float y = sqrtf(c / w) * (u - 0.5f);
            float x = b + y;

            if (x < 0.0f) continue;

            float z = 64.0f * w * w * w * v * v * v;

            if (logf(z) <= 2.0f * (b * logf(x / b) - y)) return x;
        }
    }

    __device__ inline float gamma(float a, curandState *state) {
        if (a == 1.0f) {
            return device::exponential(state);
        } else if (a < 1.0f) {
            return device::jonhk_gamma(a, state);
        } else {
            return device::best_gamma(a, state);
        }
    }
}

__global__ void gamma_kernel(float *x, int n, float a, curandState *state);

#endif