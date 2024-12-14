#include "fft_cufft.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <iomanip>


#define CHECK_CUDA(call)                                                \
    {                                                                   \
        auto err = call;                                                \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(err);                                             \
        }                                                               \
    }


#define CHECK_CUFFT(call)                                               \
    {                                                                   \
        auto err = call;                                                \
        if (err != CUFFT_SUCCESS) {                                     \
            std::cerr << "cuFFT Error: " << static_cast<int>(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(err);                                             \
        }                                                               \
    }


__global__ void normalizeKernel(float* data, int size, float normFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= normFactor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    if (input.empty()) return {};

    const size_t totalSize = input.size();
    const int n = (totalSize / batch) >> 1;  // Количество комплексных элементов в одной партии
    const size_t byteSize = sizeof(cufftComplex) * n * batch;


    cudaDeviceProp deviceProp;
    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, 0));
    const int threadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int numBlocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> output(totalSize);

 
    cufftComplex* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, byteSize));
    CHECK_CUDA(cudaMemcpy(d_data, input.data(), byteSize, cudaMemcpyHostToDevice));


    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, n, CUFFT_C2C, batch));


    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));


    CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));


    const float normFactor = 1.0f / n;
    normalizeKernel<<<numBlocks, threadsPerBlock>>>(
        reinterpret_cast<float*>(d_data), totalSize, normFactor);
    CHECK_CUDA(cudaDeviceSynchronize());


    CHECK_CUDA(cudaMemcpy(output.data(), d_data, byteSize, cudaMemcpyDeviceToHost));


    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_data));

    return output;
}
