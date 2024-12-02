#include "gelu_ocl.h"
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

const std::string GELU_KERNEL_SRC = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, const unsigned int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        float x3 = x * x * x;
        float tanh_input = sqrt(2.0f / 3.141592653589793f) * (x + 0.044715f * x3);
        float tanh_value = tanh(tanh_input);
        float result = 0.5f * x * (1.0f + tanh_value);
        output[idx] = result;
    }
}
)";

cl::Device GetGpuDevice() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    for (auto& platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (!devices.empty()) {
            return devices[0];
        }
    }

    throw std::runtime_error("No GPU device found.");
}

std::vector<float> GeluOCL(const std::vector<float>& input) {
    cl::Device device = GetGpuDevice();
    cl::Platform platform = device.getInfo<CL_DEVICE_PLATFORM>();

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    size_t size = input.size();
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, const_cast<float*>(input.data()));
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

    cl::Program::Sources sources(1, std::make_pair(GELU_KERNEL_SRC.c_str(), GELU_KERNEL_SRC.length()));
    cl::Program program(context, sources);

    try {
        program.build({device});
    } catch (cl::Error& e) {
        std::cerr << "OpenCL program build error: " << e.what() << " (" << e.err() << ")\n";
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        throw e;
    }

    cl::Kernel kernel(program, "gelu_kernel");

    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, static_cast<unsigned int>(size));

    cl::NDRange global(size);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    queue.finish();

    std::vector<float> output(size);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * size, output.data());

    return output;
}
