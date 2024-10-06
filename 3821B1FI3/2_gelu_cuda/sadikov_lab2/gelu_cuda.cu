#include "gelu_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void GuluCUDA_kernel(float* a, const int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		constexpr float twooverpi = 0.7978845608028653;
		float x = a[i];
		a[i] = 0.5f * x * (1.f + tanhf(twooverpi * x * (1.0f + 0.044715f * x * x)));
	}
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
	int threadsPerBlock = deviceProp.maxThreadsPerBlock;
	int blockNum = (input.size() + threadsPerBlock - 1) / threadsPerBlock;
	
	std::vector<float> output(input);
	float* ptr;
	cudaMalloc(&ptr, sizeof(float) * input.size());
	cudaMemcpy(ptr, output.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
	
	GuluCUDA_kernel<<<blockNum, threadsPerBlock>>>(ptr, input.size());
	
	cudaMemcpy(output.data(), ptr, sizeof(float) * input.size(), cudaMemcpyDeviceToHost);
	cudaFree(ptr);
	
	return output;
}
