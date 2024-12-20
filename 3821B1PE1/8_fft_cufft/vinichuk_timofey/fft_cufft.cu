// Copyright (c) 2024 Vinichuk Timofey
#include "fft_cufft.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

__global__ void NormalizeKernel(float* input, int size, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        input[index] /= N;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    std::vector<float> result(input.size());

    int N = input.size() / (batch * 2);

    cufftHandle handle;
    cufftComplex* data;

    cufftPlan1d(&handle, N, CUFFT_C2C, batch);

    cudaMalloc(&data, sizeof(cufftComplex) * N * batch);
    cudaMemcpy(
        data,
        input.data(),
        sizeof(cufftComplex) * N * batch,
        cudaMemcpyHostToDevice
    );

    cufftExecC2C(handle, data, data, CUFFT_FORWARD);
    cufftExecC2C(handle, data, data, CUFFT_INVERSE);

    cudaDeviceProp devPropts;
    cudaGetDeviceProperties(&devPropts, 0);
    size_t threadsPerBlock = devPropts.maxThreadsPerBlock;
    size_t blocksCount = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

    NormalizeKernel << <blocksCount, threadsPerBlock >> > (
        reinterpret_cast<float*>(data),
        result.size(),
        N
        );

    cudaMemcpy(
        result.data(),
        data,
        sizeof(cufftComplex) * N * batch,
        cudaMemcpyDeviceToHost
    );

    cufftDestroy(handle);
    cudaFree(data);

    return result;
}