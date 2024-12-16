// Copyright (c) 2024 Tushentsova Karina
#include "fft_cufft.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

__global__ void NormalizeKernel(float* input, int size, int sLen) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size){
        input[i] /= sLen;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    size_t size = input.size();
    std::vector<float> output(size);
    int sLen = size / (batch * 2);

    cufftHandle handle;
    cufftPlan1d(&handle, sLen, CUFFT_C2C, batch);

    cufftComplex* complex;
    size_t sizeBytes = sizeof(cufftComplex) * sLen * batch;
    cudaMalloc(&complex, sizeBytes);

    cudaMemcpy(complex, input.data(), sizeBytes, cudaMemcpyHostToDevice);

    cufftExecC2C(handle, complex, complex, CUFFT_FORWARD);
    cufftExecC2C(handle, complex, complex, CUFFT_INVERSE);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t threadsPerBlock = deviceProp.maxThreadsPerBlock;
    size_t blocksPerGrid = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

    NormalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(
        reinterpret_cast<float*>(complex), size, sLen
    );

    cudaMemcpy(output.data(), complex, sizeBytes, cudaMemcpyDeviceToHost);

    cufftDestroy(handle);
    cudaFree(complex);

    return output;
}