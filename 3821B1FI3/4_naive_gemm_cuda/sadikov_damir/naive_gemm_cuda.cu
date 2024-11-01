#include "naive_gemm_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void NaiveGemmCUDA_kernel(const float* a, const float* b, float* c, const int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < n && j < n) {
		float res = 0.0f;
		for (int k = 0; k < n; k++) {
			res += a[i * n + k] * b[k * n + j];
		}
		c[i * n + j] = res;
	}
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
	constexpr int threadsPerDim = 32;
	const int blocksCnt = (n + threadsPerDim - 1) / threadsPerDim;
	const dim3 blocksDim(blocksCnt, blocksCnt);
	const dim3 threadsDim(threadsPerDim, threadsPerDim);

	std::vector<float> c(n * n);
	float* ptr_a;
	float* ptr_b;
	float* ptr_c;
	cudaMalloc(&ptr_a, sizeof(float) * n * n);
	cudaMalloc(&ptr_b, sizeof(float) * n * n);
	cudaMalloc(&ptr_c, sizeof(float) * n * n);
	cudaMemcpy(ptr_a, a.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(ptr_b, b.data(), sizeof(float) * n * n, cudaMemcpyHostToDevice);

	NaiveGemmCUDA_kernel<<<blocksDim, threadsDim>>>(ptr_a, ptr_b, ptr_c, n);

	cudaMemcpy(c.data(), ptr_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

	cudaFree(ptr_a);
	cudaFree(ptr_b);
	cudaFree(ptr_c);

	return c;
}
