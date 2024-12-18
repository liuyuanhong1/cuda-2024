// Copyright (c) 2024 Chuvashov Andrey
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
__kernel void gelu(__global float* inputArray, __global float* outputArray, int arraySize) {
    int idx = get_global_id(0);
    if (idx < arraySize) {
        float x = inputArray[idx];
        outputArray[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& inputArray) {
    cl_int oclError;

    size_t arraySize = inputArray.size();
    std::vector<float> outputArray(arraySize);

    cl_platform_id oclPlatform;
    oclError = clGetPlatformIDs(1, &oclPlatform, nullptr);
    OCL_CHECK(oclError, "Getting platform ID");

    cl_device_id oclDevice;
    oclError = clGetDeviceIDs(oclPlatform, CL_DEVICE_TYPE_GPU, 1, &oclDevice, nullptr);
    if (oclError == CL_DEVICE_NOT_FOUND) {
        std::cerr << "GPU device not found. Trying CPU..." << std::endl;
        oclError = clGetDeviceIDs(oclPlatform, CL_DEVICE_TYPE_CPU, 1, &oclDevice, nullptr);
        OCL_CHECK(oclError, "Getting CPU device ID");
    }
    else {
        OCL_CHECK(oclError, "Getting GPU device ID");
    }

    cl_context oclContext = clCreateContext(nullptr, 1, &oclDevice, nullptr, nullptr, &oclError);
    OCL_CHECK(oclError, "Creating context");

    cl_command_queue oclQueue = clCreateCommandQueue(oclContext, oclDevice, 0, &oclError);
    OCL_CHECK(oclError, "Creating command queue");

    cl_mem inputBuffer = clCreateBuffer(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        arraySize * sizeof(float), (void*)inputArray.data(), &oclError);
    OCL_CHECK(oclError, "Creating input buffer");

    cl_mem outputBuffer = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY,
        arraySize * sizeof(float), nullptr, &oclError);
    OCL_CHECK(oclError, "Creating output buffer");

    cl_program oclProgram = clCreateProgramWithSource(oclContext, 1, &geluKernelSource, nullptr, &oclError);
    OCL_CHECK(oclError, "Creating program");

    oclError = clBuildProgram(oclProgram, 1, &oclDevice, nullptr, nullptr, nullptr);
    if (oclError != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(oclProgram, oclDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(oclProgram, oclDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "Error: Building program failed.\nBuild Log:\n" << buildLog.data() << std::endl;
        clReleaseProgram(oclProgram);
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseCommandQueue(oclQueue);
        clReleaseContext(oclContext);
        exit(EXIT_FAILURE);
    }

    cl_kernel oclKernel = clCreateKernel(oclProgram, "gelu", &oclError);
    OCL_CHECK(oclError, "Creating kernel");

    oclError = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), &inputBuffer);
    OCL_CHECK(oclError, "Setting kernel argument 0 (input)");
    oclError = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), &outputBuffer);
    OCL_CHECK(oclError, "Setting kernel argument 1 (output)");
    oclError = clSetKernelArg(oclKernel, 2, sizeof(int), &arraySize);
    OCL_CHECK(oclError, "Setting kernel argument 2 (arraySize)");

    size_t globalWorkSize = arraySize;

    oclError = clEnqueueNDRangeKernel(oclQueue, oclKernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    OCL_CHECK(oclError, "Enqueuing kernel");

    oclError = clEnqueueReadBuffer(oclQueue, outputBuffer, CL_TRUE, 0, arraySize * sizeof(float), outputArray.data(), 0, nullptr, nullptr);
    OCL_CHECK(oclError, "Reading output buffer");

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(oclKernel);
    clReleaseProgram(oclProgram);
    clReleaseCommandQueue(oclQueue);
    clReleaseContext(oclContext);

    return outputArray;
}
