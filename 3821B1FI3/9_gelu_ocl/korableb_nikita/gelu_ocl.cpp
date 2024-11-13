// Copyright (c) 2024 Korablev Nikita
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <cmath>

const char* gelu_kernel_source = R"(
__kernel void gelu(__global float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);

    cl_platform_id platform;
    cl_device_id device;
    cl_int err = clGetPlatformIDs(1, &platform, nullptr);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), (void*)input.data(), &err);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, &err);

    cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    err = clSetKernelArg(kernel, 2, sizeof(int), &size);

    size_t global_work_size = size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);

    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size * sizeof(float), output.data(), 0, nullptr, nullptr);

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}
