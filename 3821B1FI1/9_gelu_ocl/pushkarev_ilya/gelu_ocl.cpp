#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>
#include <string>
#include <utility>

std::vector<float> GeluOCL(const std::vector<float>& input) 
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context);

    std::string kernCode = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int n) {
    int i = get_global_id(0);

    if (i < n) {
        const float x = input[i];
        output[i] = x / (1.0f + exp(-1.59577f * (x + 0.044715f * x * x * x)));
    }
}
)";

    cl::Program::Sources sources;
    sources.emplace_back(std::move(kernCode));

    cl::Program program(context, sources);
    program.build();

    cl::Kernel kernel(program, "gelu_kernel");

    size_t n = input.size();
    size_t size_bytes = n * sizeof(*input.data());

    cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, size_bytes);
    cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, size_bytes);

    queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, size_bytes, input.data());

    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, static_cast<int>(n));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);

    std::vector<float> output(n);
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, size_bytes, output.data());

    return output;
}