// Copyright (c) 2024 Kirillov Maxim
#include "fft_cufft.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void normalize(float* data, int size, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= n;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    std::vector<float> output(input.size());

    int n = input.size() / (batch * 2);

    cufftHandle handle;
    cufftComplex* data;

    cufftPlan1d(&handle, n, CUFFT_C2C, batch);

    cudaMalloc(&data, n * sizeof(cufftComplex) * batch);
    cudaMemcpy(data,input.data(), n * sizeof(cufftComplex) * batch,cudaMemcpyHostToDevice);

    cufftExecC2C(handle, data, data, CUFFT_FORWARD);
    cufftExecC2C(handle, data, data, CUFFT_INVERSE);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
    size_t blocksCount = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

    normalize<<<blocksCount, threadsPerBlock>>>(reinterpret_cast<float*>(data), output.size(), n);

    cudaMemcpy(output.data(), data, n * sizeof(cufftComplex) * batch, cudaMemcpyDeviceToHost);

    cufftDestroy(handle);
    cudaFree(data);

    return output;
}
