#include "gelu_ocl.h"

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <stdexcept>

std::vector<float> GeluOCL(const std::vector<float>& input) {
    cl_int err;

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platform");
    }

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL GPU device");
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(context);
        throw std::runtime_error("Failed to create OpenCL command queue");
    }

    size_t dataSize = input.size() * sizeof(float);
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dataSize, (void*)input.data(), &err);
    if (err != CL_SUCCESS) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create input buffer");
    }

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create output buffer");
    }

    const char* kernelSource = R"CLC(
    __kernel void gelu_kernel(__global const float* input, __global float* output, const int size) {
        int i = get_global_id(0);
        if (i < size) {
            float x = input[i];
            // Approximate GELU function
            const float c = 0.044715f;
            const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2 / pi)
            float x3 = x * x * x;
            float tanh_arg = sqrt_2_over_pi * (x + c * x3);
            float y = 0.5f * x * (1.0f + tanh(tanh_arg));
            output[i] = y;
        }
    }
    )CLC";

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create OpenCL program");
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Error in kernel:\n" << log.data() << std::endl;

        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to build OpenCL program");
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to create OpenCL kernel");
    }

    int size = static_cast<int>(input.size());
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &size);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to set OpenCL kernel arguments");
    }

    size_t globalWorkSize = input.size();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to enqueue OpenCL kernel");
    }

    std::vector<float> output(input.size());
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, dataSize, output.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        throw std::runtime_error("Failed to read OpenCL output buffer");
    }

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(inputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}