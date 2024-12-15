#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <vector>
#include <cmath>
#include <iostream>

#define GELU(x) ((x) * 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * ((x) + 0.044715 * (x) * (x) * (x)))))

std::vector<float> GeluOCL(const std::vector<float>& input) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context);

    const char* kernelSource = R"(
        __kernel void gelu_kernel(__global float* input, __global float* output, int size) {
            int id = get_global_id(0);
            if (id < size) {
                float x = input[id];
                output[id] = 0.5f * x * (1.0f + tanh(sqrt(2.0 / 3.14159265359f) * (x + 0.044715f * x * x * x)));
            }
        }
    )";

    cl::Program::Sources sources;
    sources.emplace_back(kernelSource, strlen(kernelSource));
    cl::Program program(context, sources);
    program.build({device});

    auto size = input.size();
    auto countBytes = size * sizeof(float);

    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, countBytes, (void*)input.data());
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, countBytes);

    cl::Kernel kernel(program, "gelu_kernel");
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, static_cast<int>(size));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

    std::vector<float> output(size);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, countBytes, output.data());

    return output;
}
