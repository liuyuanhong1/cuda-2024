// Copyright (c) 2024 Sokolova Daria
#include "gelu_ocl.h"

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <cmath>

#define OCL_CHECK(status, msg) \
    if (status != CL_SUCCESS) { \
        std::cerr << "OpenCL Error: " << msg << " Error Code: " << status << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

const char* geluKernelSource = R"(
__kernel void gelu(__global float* inputBuffer, __global float* outputBuffer, int bufferSize) {
    int globalIndex = get_global_id(0);
    if (globalIndex < bufferSize) {
        float x = inputBuffer[globalIndex];
        outputBuffer[globalIndex] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& inputBuffer) {
    cl_int clError;

    size_t bufferSize = inputBuffer.size();
    std::vector<float> outputBuffer(bufferSize);

    cl_platform_id clPlatform;
    clError = clGetPlatformIDs(1, &clPlatform, nullptr);
    OCL_CHECK(clError, "Getting platform ID");

    cl_device_id clDevice;
    clError = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, nullptr);
    if (clError == CL_DEVICE_NOT_FOUND) {
        std::cerr << "GPU device not found. Switching to CPU..." << std::endl;
        clError = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_CPU, 1, &clDevice, nullptr);
        OCL_CHECK(clError, "Getting CPU device ID");
    } else {
        OCL_CHECK(clError, "Getting GPU device ID");
    }

    cl_context clContext = clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &clError);
    OCL_CHECK(clError, "Creating context");

    cl_command_queue clQueue = clCreateCommandQueue(clContext, clDevice, 0, &clError);
    OCL_CHECK(clError, "Creating command queue");

    cl_mem deviceInputBuffer = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              bufferSize * sizeof(float), (void*)inputBuffer.data(), &clError);
    OCL_CHECK(clError, "Creating input buffer");

    cl_mem deviceOutputBuffer = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY,
                                               bufferSize * sizeof(float), nullptr, &clError);
    OCL_CHECK(clError, "Creating output buffer");

    cl_program clProgram = clCreateProgramWithSource(clContext, 1, &geluKernelSource, nullptr, &clError);
    OCL_CHECK(clError, "Creating program");

    clError = clBuildProgram(clProgram, 1, &clDevice, nullptr, nullptr, nullptr);
    if (clError != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "Error: Building program failed.\nBuild Log:\n" << buildLog.data() << std::endl;
        clReleaseProgram(clProgram);
        clReleaseMemObject(deviceInputBuffer);
        clReleaseMemObject(deviceOutputBuffer);
        clReleaseCommandQueue(clQueue);
        clReleaseContext(clContext);
        exit(EXIT_FAILURE);
    }

    cl_kernel clKernel = clCreateKernel(clProgram, "gelu", &clError);
    OCL_CHECK(clError, "Creating kernel");

    clError = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &deviceInputBuffer);
    OCL_CHECK(clError, "Setting kernel argument 0 (input)");
    clError = clSetKernelArg(clKernel, 1, sizeof(cl_mem), &deviceOutputBuffer);
    OCL_CHECK(clError, "Setting kernel argument 1 (output)");
    clError = clSetKernelArg(clKernel, 2, sizeof(int), &bufferSize);
    OCL_CHECK(clError, "Setting kernel argument 2 (buffer size)");

    size_t globalWorkSize = bufferSize;

    clError = clEnqueueNDRangeKernel(clQueue, clKernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    OCL_CHECK(clError, "Enqueuing kernel");


    clError = clEnqueueReadBuffer(clQueue, deviceOutputBuffer, CL_TRUE, 0, bufferSize * sizeof(float), outputBuffer.data(), 0, nullptr, nullptr);
    OCL_CHECK(clError, "Reading output buffer");

    clReleaseMemObject(deviceInputBuffer);
    clReleaseMemObject(deviceOutputBuffer);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clQueue);
    clReleaseContext(clContext);

    return outputBuffer;
}
