// Copyright (c) 2024 Tushentsova Karina
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <cmath>
#include <vector>

#define CHECK_OCL_ERROR(codeError, msg) \
    if (codeError != CL_SUCCESS) { \
        std::cerr << "error: " << msg << " code error: " << codeError << ". At line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

const char* geluKernel = R"(
__kernel void gelu(__global float* input, __global float* output, int arrSize) {
    int i = get_global_id(0);
    if (i < arrSize) {
        float val = input[i];
        output[i] = 0.5f * val * (1.0f + tanh(sqrt(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    size_t arrSize = input.size();
    std::vector<float> output(arrSize);

    cl_int clError;
    cl_platform_id clPlatform;
    clError = clGetPlatformIDs(1, &clPlatform, nullptr);
    CHECK_OCL_ERROR(clError, "Platform ID...");

    cl_device_id clDevice;
    clError = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, nullptr);

    if (clError == CL_DEVICE_NOT_FOUND) {
        std::cerr << "No device available.CPU..." << std::endl;
        clError = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_CPU, 1, &clDevice, nullptr);
        CHECK_OCL_ERROR(clError, "CPU device ID...");
    }
    else {
        CHECK_OCL_ERROR(clError, "GPU device ID...");
    }

    cl_context clContext = clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &clError);
    CHECK_OCL_ERROR(clError, "Context");

    cl_command_queue clQueue = clCreateCommandQueue(clContext, clDevice, 0, &clError);
    CHECK_OCL_ERROR(clError, "Command queue");

    cl_mem inputBuff = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, arrSize * sizeof(float), (void*)input.data(), &clError);
    CHECK_OCL_ERROR(clError, "Input buffer");

    cl_mem outputBuff = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, arrSize * sizeof(float), nullptr, &clError);
    CHECK_OCL_ERROR(clError, "Output buffer");

    cl_program clProgram = clCreateProgramWithSource(clContext, 1, &geluKernel, nullptr, &clError);
    CHECK_OCL_ERROR(clError, "Program");

    clError = clBuildProgram(clProgram, 1, &clDevice, nullptr, nullptr, nullptr);
    if (clError != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        std::cerr << "Error: Build failed.\nLog:\n" << buildLog.data() << std::endl;
        clReleaseProgram(clProgram);
        clReleaseMemObject(inputBuff);
        clReleaseMemObject(outputBuff);
        clReleaseCommandQueue(clQueue);
        clReleaseContext(clContext);
        exit(EXIT_FAILURE);
    }

    cl_kernel clKernel = clCreateKernel(clProgram, "gelu", &clError);
    CHECK_OCL_ERROR(clError, "Kernel");

    clError = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &inputBuff);
    CHECK_OCL_ERROR(clError, "Argument 0 (input)");
    clError = clSetKernelArg(clKernel, 1, sizeof(cl_mem), &outputBuff);
    CHECK_OCL_ERROR(clError, "Argument 1 (output)");
    clError = clSetKernelArg(clKernel, 2, sizeof(int), &arrSize);
    CHECK_OCL_ERROR(clError, "Argument 2 (arrSize)");

    size_t globalSize = arrSize;

    clError = clEnqueueNDRangeKernel(clQueue, clKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    CHECK_OCL_ERROR(clError, "Enqueuing kernel");

    clError = clEnqueueReadBuffer(clQueue, outputBuff, CL_TRUE, 0, arrSize * sizeof(float), output.data(), 0, nullptr, nullptr);
    CHECK_OCL_ERROR(clError, "Reading output buffer");

    clReleaseMemObject(inputBuff);
    clReleaseMemObject(outputBuff);
    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clQueue);
    clReleaseContext(clContext);

    return output;
}