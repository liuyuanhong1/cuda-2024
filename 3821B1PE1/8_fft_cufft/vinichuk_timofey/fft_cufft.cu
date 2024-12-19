// Copyright (c) 2024 Vinichuk Timofey
#include "fft_cufft.h"

__global__ void normalizeKernel(float* input, int size, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		input[index] /= n;
	}
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
	std::vector<float> output(input.size());

	int n = input.size() / (batch * 2);

	cufftHandle handle;
	cufftPlan1d(&handle, n, CUFFT_C2C, batch);

	cufftComplex* data;
	cudaMalloc(&data, sizeof(cufftComplex) * n * batch);
	cudaMemcpy(
		data,
		input.data(),
		sizeof(cufftComplex) * n * batch,
		cudaMemcpyHostToDevice
	);

	cufftExecC2C(handle, data, data, CUFFT_FORWARD);
	cufftExecC2C(handle, data, data, CUFFT_INVERSE);

	cudaDeviceProp devPropts;
	cudaGetDeviceProperties(&devPropts, 0);
	size_t threadsPerBlock = devPropts.maxThreadsPerBlock;
	size_t blocksCount = (input.size() + threadsPerBlock - 1) / threadsPerBlock;

	normalizeKernel << <blocksCount, threadsPerBlock >> > (
		reinterpret_cast<float*>(data),
		output.size(),
		n);

	cudaMemcpy(
		output.data(),
		data,
		sizeof(cufftComplex) * n * batch,
		cudaMemcpyDeviceToHost
	);

	cufftDestroy(handle);
	cudaFree(data);

	return output;
}