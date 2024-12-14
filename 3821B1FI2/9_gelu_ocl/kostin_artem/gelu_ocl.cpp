#include <CL/cl.hpp>
#include <vector>
#include <cmath>
#include <iostream>

#define GELU(x) ((x) * 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * ((x) + 0.044715 * (x) * (x) * (x)))))

std::vector<float> GeluOCL(const std::vector<float>& input) {
    cl_int err;
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;

    err = cl::Platform::get(&platform);
    err |= platform.getDevices(CL_DEVICE_GPU, &device);
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);

    const char* kernelSource = R"(
        __kernel void gelu_kernel(__global float* input, __global float* output, int size) {
            int id = get_global_id(0);
            if (id < size) {
                float x = input[id];
                output[id] = 0.5f * x * (1.0f + tanh(sqrt(2.0 / 3.14159265359f) * (x + 0.044715f * x * x * x)));
            }
        }
    )";

    cl::Program program(context, kernelSource);
    program.build("-cl-std=CL1.2");

    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(float), (void*)input.data());
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, input.size() * sizeof(float));

    cl::Kernel kernel(program, "gelu_kernel");
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, (int)input.size());

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(256));
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, input.size() * sizeof(float), output.data());

    return output;
}
