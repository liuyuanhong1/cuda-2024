// Copyright (c) 2024 Vanushkin Dmitry
#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>

__global__ void normKernel(float* input, int size, float norm) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        input[i] *= norm;
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    const int size = input.size();
    std::vector<float> output(size);

    int n = (size / batch) >> 1;
    int sizeInBytes = sizeof(cufftComplex) * n * batch;

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftComplex* data;
    cudaMalloc(&data, sizeInBytes);
    cudaMemcpy(data, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cufftExecC2C(plan, data, data, CUFFT_FORWARD);

    cufftExecC2C(plan, data, data, CUFFT_INVERSE);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    float norm = 1.0f / static_cast<float>(n);
    normKernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<float*>(data), size, norm);


    cudaMemcpy(output.data(), data, sizeInBytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(data);

    return output;
}
