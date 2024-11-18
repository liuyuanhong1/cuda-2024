#include "naive_gemm_cuda.h"
#include <vector>

__global__ void NaiveGemmKernel(const float *a, const float *b, float *output,
                                  int n) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n && j < n) {
    float res = 0.0f;
    for (int k = 0; k < n; k++) {
        res = fma(a[i * n + k], b[k * n + j], res);
    }
    output[i * n + j] = res;
  }
  
}

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  std::vector<float> output(n * n);

  float *a_dev, *b_dev, *output_dev;

  cudaMalloc(&a_dev, a.size() * sizeof(float));
  cudaMalloc(&b_dev, b.size() * sizeof(float));
  cudaMalloc(&output_dev, n * n * sizeof(float));

  cudaMemcpy(a_dev, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 64;
  dim3 blockDim(blockSize, blockSize);
  dim3 gridDim((n + blockSize - 1) / blockSize, (n + blockSize - 1) / blockSize);

  NaiveGemmKernel<<<gridDim, blockDim>>>(a_dev, b_dev, output_dev, n);

  cudaMemcpy(output.data(), output_dev, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(output_dev);
  return output;
}