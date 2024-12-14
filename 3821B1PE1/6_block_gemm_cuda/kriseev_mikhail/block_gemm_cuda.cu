#include "block_gemm_cuda.h"

__global__ void BlockGemm(float *a, float *b, float *out, int n, int blockSize, int numBlocks) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < numBlocks && j < numBlocks) {
      for (int k = 0; k < numBlocks; ++k) {
        for (int l = 0; l < blockSize; ++l) {
          for (int m = 0; m < blockSize; ++m) {
            float res = 0.0f;
            for (int k1 = 0; k1 < blockSize; ++k1) {
              res += a[(i * blockSize + l) * n + k * blockSize + k1] *
                     b[(k * blockSize + k1) * n + j * blockSize + m];
            }
            out[(i * blockSize + l) * n + j * blockSize + m] += res;
          }
        }
      }
  }

}

std::vector<float> BlockGemmCUDA(const std::vector<float> &a,
                                 const std::vector<float> &b, int n) {
  std::vector<float> out(n * n);

  float *a_d, *b_d, *out_d;
  cudaMalloc(&a_d, a.size() * sizeof(float));
  cudaMalloc(&b_d, b.size() * sizeof(float));
  cudaMalloc(&out_d, n * n * sizeof(float));

  cudaMemcpy(a_d, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 16;
  int numBlocks = n / blockSize;

  dim3 blockDim(blockSize, blockSize);
  dim3 gridDim(numBlocks, numBlocks);

  BlockGemm<<<gridDim, blockDim>>>(a_d, b_d, out_d, n, blockSize, numBlocks);

  cudaMemcpy(out.data(), out_d, n * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(out_d);

  return out;
}
