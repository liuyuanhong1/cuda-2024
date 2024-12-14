#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>
#include <string>
#include <utility>

std::vector<float> GeluOCL(const std::vector<float>& input) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context);

    std::string kernCode = R"(
__kernel void myKernel(__global const float* input, __global float* output, int size) {
    int i = get_global_id(0);

    if (i < size) {
        const float x = input[i];
        output[i] = x / (1.0f + exp(-1.59577f * (x + 0.044715f * x * x * x)));
    }
}
)";

    cl::Program::Sources sources;
    sources.emplace_back(std::move(kernCode));

    cl::Program program(context, sources);
    program.build();

    cl::Kernel kernel(program, "myKernel");

    size_t size = input.size();
    size_t sizeInBytes = size * sizeof(*input.data());

    cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, sizeInBytes);
    cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, sizeInBytes);

    queue.enqueueWriteBuffer(bufferInput, CL_TRUE,
                             0, sizeInBytes,
                             input.data());

    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, static_cast<int>(size));
    queue.enqueueNDRangeKernel(kernel,
                               cl::NullRange, cl::NDRange(size), cl::NullRange);

    std::vector<float> output(size);
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE,
                            0, sizeInBytes,
                            output.data());

    return output;
}
