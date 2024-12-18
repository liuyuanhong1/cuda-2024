// Copyright (c) 2024 Zakharov Artem
#include "fft_cufft.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"


__global__ void normalize_signal_kernel(float* signal, int size, float k) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind < size) {
        signal[ind] *= k;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);

    int size = input.size();
    int elements_per_batch = size / batch / 2;
    int bytes_size = sizeof(cufftComplex) * elements_per_batch * batch;

    int threads_per_block = device_prop.maxThreadsPerBlock;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    std::vector<float> output(size);

    cufftComplex* signal;
    cudaMalloc(reinterpret_cast<void**>(&signal), bytes_size);
    cudaMemcpy(reinterpret_cast<void*>(signal),
               reinterpret_cast<const void*>(input.data()),
               bytes_size, cudaMemcpyHostToDevice);

    cufftHandle handle;
    cufftPlan1d(&handle, elements_per_batch, CUFFT_C2C, batch);
    cufftExecC2C(handle, signal, signal, CUFFT_FORWARD);
    cufftExecC2C(handle, signal, signal, CUFFT_INVERSE);
    cufftDestroy(handle);

    normalize_signal_kernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float*>(signal), size, 1.0f / elements_per_batch);
    cudaMemcpy(reinterpret_cast<void*>(output.data()),
               reinterpret_cast<const void*>(signal),
               bytes_size, cudaMemcpyDeviceToHost);
    cudaFree(reinterpret_cast<void*>(signal));

    return output;
}
