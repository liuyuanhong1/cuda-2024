#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUFFT_CHECK(status) \
    if (status != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int n = input.size() / (2 * batch);

    std::vector<float> output(input.size(), 0.0f);

    cufftComplex *d_input = nullptr, *d_output = nullptr;

    size_t bytes = input.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CUFFT_CHECK(cufftPlanMany(&plan, 1, &n, NULL, 1, n, NULL, 1, n, CUFFT_C2C, batch));

    CUFFT_CHECK(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));

    CUFFT_CHECK(cufftExecC2C(plan, d_output, d_input, CUFFT_INVERSE));

    CUFFT_CHECK(cufftDestroy(plan));

    float inv_n = 1.0f / static_cast<float>(n);
    int total = n * batch;
    auto normalize_kernel = [] __global__ (cufftComplex* data, int total, float inv_n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total) {
            data[idx].x *= inv_n;
            data[idx].y *= inv_n;
        }
    };

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalize_kernel<<<blocks, threads>>>(d_input, total, inv_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data(), d_input, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}