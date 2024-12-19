// Copyright (c) 2024 Sokolova Daria
#include "fft_cufft.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

__global__ void NormalizeKernel(float* data, int totalSize, int signalSize) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex < totalSize) {
        data[threadIndex] /= signalSize;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& inputSignals, int numSignals) {
    size_t totalSize = inputSignals.size();
    std::vector<float> outputSignals(totalSize);

    int signalSize = totalSize / (numSignals * 2);

    cufftHandle fftPlan;
    cufftPlan1d(&fftPlan, signalSize, CUFFT_C2C, numSignals);

    cufftComplex* deviceData = nullptr;
    size_t bufferSize = sizeof(cufftComplex) * signalSize * numSignals;
    cudaMalloc(&deviceData, bufferSize);

    cudaMemcpy(deviceData, inputSignals.data(), bufferSize, cudaMemcpyHostToDevice);

    cufftExecC2C(fftPlan, deviceData, deviceData, CUFFT_FORWARD);
    cufftExecC2C(fftPlan, deviceData, deviceData, CUFFT_INVERSE);

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    size_t threadsPerBlock = deviceProperties.maxThreadsPerBlock;
    size_t blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

    NormalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<float*>(deviceData), totalSize, signalSize
    );

    cudaMemcpy(outputSignals.data(), deviceData, bufferSize, cudaMemcpyDeviceToHost);

    cufftDestroy(fftPlan);
    cudaFree(deviceData);

    return outputSignals;
}
