// Copyright (c) 2024 Morgachev Stepan
#include "fft_cufft.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

__global__ void NormalizeKernel(float* input, int size, int signalLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        input[index] /= signalLength;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    size_t size = input.size();
    std::vector<float> output(size);

    int signalLength = size / (batch * 2);

    cufftHandle handle;
    cufftPlan1d(&handle, signalLength, CUFFT_C2C, batch);

    cufftComplex* complex;
    size_t sizeInBytes = sizeof(cufftComplex) * signalLength * batch;
    cudaMalloc(&complex, sizeInBytes);

    cudaMemcpy(complex, input.data(), sizeInBytes, cudaMemcpyHostToDevice);

    cufftExecC2C(handle, complex, complex, CUFFT_FORWARD);
    cufftExecC2C(handle, complex, complex, CUFFT_INVERSE);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
    size_t blocksPerGrid = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

    NormalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<float*>(complex), size, signalLength
    );

    cudaMemcpy(output.data(), complex, sizeInBytes, cudaMemcpyDeviceToHost);

    cufftDestroy(handle);
    cudaFree(complex);

    return output;
}
