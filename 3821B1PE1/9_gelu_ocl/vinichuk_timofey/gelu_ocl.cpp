// Copyright (c) 2024 Vinichuk Timofey
#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>

std::vector<float> GeluOCL(const std::vector<float>& input) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    cl::Context context(device);
    cl::CommandQueue queue(context);

    std::string codeKernel = R"(
__kernel void gelu_kernel(__global const float* x, __global float* y, int countElem) {
  int i = get_global_id(0);

  if (i < countElem) {
      float val = x[i];
      float tmp = val * (1.595769122f + val * val * 0.071354816f);
      y[i] = val - val / (1.0f + exp(tmp));
  }
}
)";

    cl::Program::Sources sources;
    sources.emplace_back(std::move(codeKernel));
    cl::Program program(context, sources);

    if (program.build() != CL_SUCCESS) {
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
        return {};
    }

    if (input.empty()) {
        return {};
    }
    auto size = input.size();
    auto numBytes = size * sizeof(float);

    cl::Buffer inBuffer(context, CL_MEM_READ_ONLY, numBytes);
    cl::Buffer outBuffer(context, CL_MEM_WRITE_ONLY, numBytes);

    queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, numBytes, input.data());

    cl::Kernel kernel(program, "gelu_kernel");
    kernel.setArg(0, inBuffer);
    kernel.setArg(1, outBuffer);
    kernel.setArg(2, static_cast<int>(size));
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

    std::vector<float> output(size);
    queue.enqueueReadBuffer(outBuffer, CL_TRUE, 0, numBytes, output.data());

    return output;
}