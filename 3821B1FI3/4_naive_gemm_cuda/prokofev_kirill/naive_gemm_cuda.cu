// Copyright (c) 2024 Prokofev Kirill
#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>


/*
__global__ void MulMatrixKernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}
*/


__global__ void MulMatrixKernel(const float* a, const float* b, float* c, int n, int offset) {
    size_t row = offset + blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

std::vector<float> GenVec(const int size){
    std::vector<float> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(size_t i = 0;i < vec.size();++i){
        vec[i] = dist(gen);
    }
    return vec;
}



std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    
    
    const size_t count = n * n;
    std::vector<float> result(count, 0.0f);
    
    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceResult = nullptr;
    cudaMalloc(&deviceA, count * sizeof(float));
    cudaMalloc(&deviceB, count * sizeof(float));
    cudaMalloc(&deviceResult, count * sizeof(float));

   
    const int numStreams = 4; 
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    cudaMemcpy(deviceA, a.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    for (int i = 0; i < numStreams; ++i) {
        int offset = (n / numStreams) * i; 
        int rows = (i == numStreams - 1) ? n - offset : (n / numStreams); 

        MulMatrixKernel<<<gridSize, blockSize, 0, streams[i]>>>(deviceA, deviceB, deviceResult, rows, offset);
    }

    cudaMemcpy(result.data(), deviceResult, count * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return result;

}





int main(){

    std::vector<float> a {1.2,3.4,5.6,3.2,5.5,6.6,7.7,8.8,4.5};
    std::vector<float> b {1.2,3.4,5.6,3.2,5.5,6.6,7.7,8.8,4.5};
    
    //std::vector<float> a = GenVec(64 * 64);
    //std::vector<float> b = GenVec(64 * 64);
    
    // Performance Measuring
    auto start = std::chrono::high_resolution_clock::now();
    auto res = NaiveGemmCUDA(a,b,3);

    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken: " << duration.count() << " s" << std::endl;

    
    for(int i = 0;i < 9;i++){
        std::cout << res[i] << "\t";
    }
    
    
    return 0;
}



/*
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    
    cudaMemcpyAsync(deviceA, a.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(deviceB, b.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream2);

    
    cudaEventRecord(event1, stream1);
    cudaEventRecord(event2, stream2);

    
    cudaStreamWaitEvent(stream3, event1, 0);
    cudaStreamWaitEvent(stream3, event2, 0);

    
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    
    MulMatrixKernel<<<gridSize, blockSize, 0, stream3>>>(deviceA, deviceB, deviceResult, n);

    
    cudaMemcpyAsync(result.data(), deviceResult, count * sizeof(float), cudaMemcpyDeviceToHost, stream3);

    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceResult);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    */
