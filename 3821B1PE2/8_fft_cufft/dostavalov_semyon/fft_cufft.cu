// Copyright (c) 2024 Dostavalov Semyon

#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_cufft.h"


__global__ void normalizeKernel(float* __restrict__ input, int size, float normalizationFactor) {
    const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int vectorIndex = threadIndex / 4;

    if (vectorIndex < size / 4) {
        float4* inputVector = reinterpret_cast<float4*>(input);
        float4 data = __ldg(&inputVector[vectorIndex]);
        data.x *= normalizationFactor;
        data.y *= normalizationFactor;
        data.z *= normalizationFactor;
        data.w *= normalizationFactor;
        inputVector[vectorIndex] = data;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) 
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int size = input.size();
    std::vector<float> output(size);

    const int elemPerBatch = size / batch >> 1;
    const int sizeInBytes = sizeof(cufftComplex) * elemPerBatch * batch;
    const int threadsPerBlock = deviceProp.maxThreadsPerBlock;
    const int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    cufftComplex* signal;
    cudaMalloc(&signal, sizeInBytes);
    cudaMemcpy(signal, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cufftHandle handle;
    cufftPlan1d(&handle, elemPerBatch, CUFFT_C2C, batch);
    cufftExecC2C(handle, signal, signal, CUFFT_FORWARD);
    cufftExecC2C(handle, signal, signal, CUFFT_INVERSE);

    normalizeKernel<<<numBlocks, threadsPerBlock>>>(reinterpret_cast<float*>(signal), size, 1.0f / elemPerBatch);

    cudaMemcpy(output.data(), signal, sizeInBytes, cudaMemcpyDeviceToHost);

    cufftDestroy(handle);
    cudaFree(signal);

    return output;
}
