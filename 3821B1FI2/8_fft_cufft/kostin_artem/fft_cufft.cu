#include <vector>
#include <cufft.h>
#include <iostream>

__global__ void normalize_kernel(float* data, size_t size, float norm_factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= norm_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    size_t size = input.size();
    int elemPerBatch = size / (2 * batch);

    cufftComplex* d_signal = nullptr;
    size_t complex_size = sizeof(cufftComplex) * elemPerBatch * batch;
    cudaMalloc(&d_signal, complex_size);

    cudaMemcpy(d_signal, input.data(), complex_size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, elemPerBatch, CUFFT_C2C, batch);
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE);

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    normalize_kernel<<<numBlocks, threadsPerBlock>>>(reinterpret_cast<float*>(d_signal), size, 1.0f / elemPerBatch);

    cudaDeviceSynchronize();

    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_signal, complex_size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_signal);

    return output;
}
