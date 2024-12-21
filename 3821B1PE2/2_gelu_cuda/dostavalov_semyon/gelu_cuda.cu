// Copyright (c) 2024 Dostavalov Semyon

#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "activation_cuda.h"

__global__ void ActivationKernel(const float* input_data, float* output_data, size_t length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= length) return;

	constexpr float alpha = 1.595769122f;
	constexpr float beta = 0.071354816f;

	float val = input_data[idx];
	output_data[idx] = val * (1.0f - 1.0f / (1.0f + __expf(val * (alpha + val * val * beta))));
}

std::vector<float> GeluCUDA(const std::vector<float>& input_data) {
	if (input_data.empty()) return {};

	cudaDeviceProp device_properties;
	cudaGetDeviceProperties(&device_properties, 0);

	auto length = input_data.size();
	std::vector<float> result(length);

	size_t data_size = length * sizeof(float);
	int max_threads_per_block = device_properties.maxThreadsPerBlock;
	int total_blocks = (length + max_threads_per_block - 1) / max_threads_per_block;

	float* dev_input = nullptr;
	cudaMalloc(&dev_input, data_size);

	float* dev_output = nullptr;
	cudaMalloc(&dev_output, data_size);

	cudaMemcpy(dev_input, input_data.data(), data_size, cudaMemcpyHostToDevice);

	ActivationKernel << <total_blocks, max_threads_per_block >> > (dev_input, dev_output, length);

	cudaDeviceSynchronize();
	cudaMemcpy(result.data(), dev_output, data_size, cudaMemcpyDeviceToHost);

	cudaFree(dev_output);
	cudaFree(dev_input);

	return result;
}
