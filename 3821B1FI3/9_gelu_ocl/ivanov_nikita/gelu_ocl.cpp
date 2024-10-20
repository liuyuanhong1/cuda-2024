#include "gelu_ocl.h"
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>

const char* kernel_source = R"(
__kernel void gelu(__global const float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / 3.14159265358979323846f) * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    size_t size = input.size();
    std::vector<float> output(size);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    cl::Program program(context, kernel_source);
    program.build(devices);
    cl::Kernel kernel(program, "gelu");

    cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(float), (void*)input.data());
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float));

    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    kernel.setArg(2, static_cast<int>(size));

    cl::NDRange global_size(size);
    cl::NDRange local_size(256);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size);

    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, size * sizeof(float), output.data());

    return output;
}
