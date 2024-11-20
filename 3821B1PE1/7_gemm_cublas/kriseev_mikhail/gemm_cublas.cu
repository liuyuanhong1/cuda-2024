#include "gemm_cublas.h"

#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b, int n) {
  std::vector<float> output(n * n);

  float *a_dev;
  float *b_dev;
  float *output_dev;

  cudaMalloc(&a_dev, a.size() * sizeof(float));
  cudaMalloc(&b_dev, b.size() * sizeof(float));
  cudaMalloc(&output_dev, output.size() * sizeof(float));

  cudaMemcpy(a_dev, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;
  
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n,n,n, &alpha, a_dev, n, b_dev, n, &beta, output_dev, n);

  cudaMemcpy(output.data(), output_dev, output.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  cublasDestroy(handle);
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(output_dev);

  return output;
}
