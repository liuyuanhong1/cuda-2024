// Copyright (c) 2024 Zakharov Artem
#include "gelu_ocl.h"
#include <string>
#include <CL/opencl.hpp>


std::vector<float> GeluOCL(const std::vector<float>& input) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context);

    std::string kernel_code = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int size) {
    int i = get_global_id(0);

    if (i < size) {
        const float x = input[i];
        output[i] = x / (1.0f + exp(-1.59577f * (x + 0.044715f * x * x * x)));
    }
}
)";

    cl::Program::Sources sources;
    sources.emplace_back(std::move(kernel_code));

    cl::Program program(context, sources);
    program.build();

    cl::Kernel kernel(program, "gelu_kernel");

    int size = input.size();
    int bytes_size = size * sizeof(float);

    cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, bytes_size);
    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, bytes_size);

    queue.enqueueWriteBuffer(buffer_input, CL_TRUE,
                             0, bytes_size, input.data());
    kernel.setArg(0, buffer_input);
    kernel.setArg(1, buffer_output);
    kernel.setArg(2, size);

    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange, cl::NDRange(size), cl::NullRange);

    std::vector<float> output(size);
    queue.enqueueReadBuffer(buffer_output, CL_TRUE,
                            0, bytes_size, output.data());

    return output;
}
