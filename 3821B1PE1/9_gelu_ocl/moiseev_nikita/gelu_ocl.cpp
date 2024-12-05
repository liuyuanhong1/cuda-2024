// Copyright (c) 2024 Moiseev Nikita
#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>

#define CHECK_CL_ERROR(callable)                                          \
  {                                                                       \
    auto error_code = callable;                                           \
    if (error_code != CL_SUCCESS) {                                       \
      std::cerr << "\033[1;31mOpenCL error\033[0m: ";                     \
      std::cerr << "Error code: " << static_cast<int>(error_code) << '\n';\
      std::cerr << "Location: " << __FILE__ << '(' << __LINE__ << ")\n";  \
      std::exit(static_cast<int>(error_code));                            \
    }                                                                     \
  }

std::vector<float> GELU_OCL(const std::vector<float>& input_data) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "\033[1;31mError\033[0m: No OpenCL platforms available\n";
        return {};
    }
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "\033[1;31mError\033[0m: No OpenCL devices available\n";
        return {};
    }
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue command_queue(context);

    const std::string kernel_code = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, int element_count) {
    int index = get_global_id(0);
    if (index < element_count) {
        float value = input[index];
        float temp = value * (1.595769122f + value * value * 0.071354816f);
        output[index] = value - value / (1.0f + exp(temp));
    }
}
)";

    cl::Program::Sources sources;
    sources.emplace_back(kernel_code);
    cl::Program program(context, sources);
    if (program.build() != CL_SUCCESS) {
        std::cerr << "\033[1;31mError\033[0m: ";
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
        return {};
    }

    if (input_data.empty()) return {};
    size_t input_size = input_data.size();
    size_t buffer_size = input_size * sizeof(float);

    cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, buffer_size);
    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, buffer_size);

    CHECK_CL_ERROR(command_queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, buffer_size, input_data.data()));

    cl::Kernel kernel(program, "gelu_kernel");
    kernel.setArg(0, buffer_input);
    kernel.setArg(1, buffer_output);
    kernel.setArg(2, static_cast<int>(input_size));

    CHECK_CL_ERROR(command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_size), cl::NullRange));

    std::vector<float> output_data(input_size);
    CHECK_CL_ERROR(command_queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, buffer_size, output_data.data()));

    return output_data;
}
