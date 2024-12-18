//  Copyright (c) 2024 Vinokurov Ivan
#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_cufft.h"

__global__ void normalizeKernel(float* __restrict__ dataPtr, int totalSize, float scaleFactor) {
    const int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const int vectorIdx = threadId / 4;

    if (vectorIdx < totalSize / 4) {
        float4* vectorData = reinterpret_cast<float4*>(dataPtr);
        float4 vectorElement = __ldg(&vectorData[vectorIdx]);
        vectorElement.x *= scaleFactor;
        vectorElement.y *= scaleFactor;
        vectorElement.z *= scaleFactor;
        vectorElement.w *= scaleFactor;
        vectorData[vectorIdx] = vectorElement;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch)
{
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    const int inputSize = input.size();
    std::vector<float> result(inputSize);

    const int elementsPerBatch = inputSize / batch >> 1;
    const int memorySize = sizeof(cufftComplex) * elementsPerBatch * batch;
    const int threadsPerBlock = deviceProperties.maxThreadsPerBlock;
    const int totalBlocks = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    cufftComplex* deviceBuffer;
    cudaMalloc(&deviceBuffer, memorySize);
    cudaMemcpy(deviceBuffer, input.data(), memorySize, cudaMemcpyHostToDevice);

    cufftHandle fftPlan;
    cufftPlan1d(&fftPlan, elementsPerBatch, CUFFT_C2C, batch);
    cufftExecC2C(fftPlan, deviceBuffer, deviceBuffer, CUFFT_FORWARD);
    cufftExecC2C(fftPlan, deviceBuffer, deviceBuffer, CUFFT_INVERSE);

    normalizeKernel<<<totalBlocks, threadsPerBlock>>>(reinterpret_cast<float*>(deviceBuffer), inputSize, 1.0f / elementsPerBatch);

    cudaMemcpy(result.data(), deviceBuffer, memorySize, cudaMemcpyDeviceToHost);
    cufftDestroy(fftPlan);
    cudaFree(deviceBuffer);

    return result;
}
