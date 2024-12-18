// Copyright 2024 Kachalov Mikhail
#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

const char *kernel_source = R"(
__kernel void gelu(__global const float* input, __global float* output, const unsigned int size) {
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        float c = 0.044715f * x * x * x;
        output[id] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + c)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float> &input)
{

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, input.size() * sizeof(float));
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float));
    queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input.size() * sizeof(float), input.data());
    cl::Program::Sources sources(1, std::string(kernel_source, strlen(kernel_source)));
    cl::Program program(context, sources);

    program.build({device});
    cl::Kernel kernel(program, "gelu");
    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    kernel.setArg(2, static_cast<unsigned int>(input.size()));
    cl::NDRange global(input.size());
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    queue.finish();

    std::vector<float> output(input.size());
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, input.size() * sizeof(float), output.data());

    return output;
}