// Copyright (c) 2024 Chuvashov Andrey
#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <CL/cl_platform.h>
#include <string>
#include <utility>
#include <math.h>

std::vector<float> GeluOCL(const std::vector<float>& input) {
    std::string gelu_kernel = R"(
__kernel void GeluOCLKernel(__global const float* input, __global float* output, int size) {
            
    int index = get_global_id(0);

    const float pi_c = sqrt(2.0f / (2 * asin(1.0f)));
    const float par_c = 0.044715f;
            
    if (i < size) {
        const float x = input[index];
        output[index] = 0.5f * x * (1.0f + tanhf(pi_c * (x + par_c * x * x * x)));
    }
}
)";

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    size_t input_size = input.size();
    size_t bytesSize = input_size * sizeof(*input.data());

    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context);

    cl::Program::Sources sources;
    sources.emplace_back(std::move(gelu_kernel));

    cl::Program program(context, sources);
    program.build();

    cl::Kernel kernel(program, "GeluOCLKernel");

    cl::Buffer bufferInput(
        context,
        CL_MEM_READ_ONLY,
        bytesSize
    );

    cl::Buffer bufferOutput(
        context,
        CL_MEM_WRITE_ONLY,
        bytesSize
    );

    queue.enqueueWriteBuffer(
        bufferInput,
        CL_TRUE,
        0,
        bytesSize,
        input.data()
    );

    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, static_cast<int>(input_size));

    queue.enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(input_size),
        cl::NullRange
    );

    std::vector<float> result(input_size);

    queue.enqueueReadBuffer(
        bufferOutput,
        CL_TRUE,
        0,
        bytesSize,
        result.data()
    );

    return result;
}
