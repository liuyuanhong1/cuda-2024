// Copyright (c) 2024 Yurin Andrey

#include "gelu_ocl.h"

#include <CL/opencl.hpp>

const  std::string codeKernel = R"(
    #define COEF1 1.595769122f
    #define COEF2 0.071354816f

  __kernel void gelu_kernel(__global const float* x, __global float* y, int countElem) {
    int i = get_global_id(0);

    if (i < countElem) {
        float value = x[i];
        float value_sq = value * value;
        float tmp = value * (COEF1 + value_sq * COEF2);
        y[i] = value - value / (1.0f + native_exp(tmp));
    }
  }
  )";

std::vector<float> GeluOCL(const std::vector<float>& input) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  cl::Device device = devices.front();

  cl::Context context(device);
  cl::CommandQueue queue(context);

  cl::Program::Sources sources;
  sources.emplace_back(std::move(codeKernel));

  cl::Program program(context, sources);

  if (input.empty()) return {};
  
  auto size = input.size();
  auto countBytes = size * sizeof(float);

  cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, countBytes);
  cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, countBytes);

  queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, countBytes, input.data());

  cl::Kernel kernel(program, "gelu_kernel");
  kernel.setArg(0, bufferInput);
  kernel.setArg(1, bufferOutput);
  kernel.setArg(2, static_cast<int>(size));
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);

  std::vector<float> output(size);
  queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, countBytes, output.data());

  return output;
}
