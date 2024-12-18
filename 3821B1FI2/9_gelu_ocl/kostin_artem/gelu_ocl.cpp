#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <vector>
#include <cmath>

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
        __kernel void gelu_kernel(__global const float* x, __global float* y, int size) {
            int i = get_global_id(0);
            if (i < size) {
                float val = x[i];
                float tmp = val * (1.595769122f + val * val * 0.071354816f);
                y[i] = val - val / (1.0f + exp(tmp));
            }
        }
    )";

    cl::Program::Sources sources;
    sources.emplace_back(kernelSource);
    cl::Program program(context, sources);
    program.build({device});

    size_t size = input.size();
    size_t countBytes = size * sizeof(float);

    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, countBytes);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, countBytes);

    queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, countBytes, input.data());

    cl::Kernel kernel(program, "gelu_kernel");
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, static_cast<int>(size));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
    queue.finish();

    std::vector<float> output(size);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, countBytes, output.data());

    return output;
}
