#include <vector>
#include <cufft.h>
#include <stdexcept>

__global__ void normalize_kernel(float* data, size_t size, float norm_factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= norm_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {

    int n = input.size() / (2 * batch);

    cufftComplex* d_input;
    cufftComplex* d_output;
    size_t size = input.size() * sizeof(float) / 2;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);

    cufftHandle plan;
    if (cufftPlan1d(&plan, n, CUFFT_C2C, batch) != CUFFT_SUCCESS) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("Failed to create cuFFT plan.");
    }

    if (cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("Failed to execute forward FFT.");
    }

    if (cufftExecC2C(plan, d_output, d_input, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("Failed to execute inverse FFT.");
    }

    int threadsPerBlock = 256;
    int numBlocks = (input.size() + threadsPerBlock - 1) / threadsPerBlock;
    normalize_kernel<<<numBlocks, threadsPerBlock>>>(reinterpret_cast<float*>(d_input), input.size(), 1.0f / n);

    cudaDeviceSynchronize();

    std::vector<float> output(input.size());
    cudaMemcpy(output.data(), d_input, size, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
