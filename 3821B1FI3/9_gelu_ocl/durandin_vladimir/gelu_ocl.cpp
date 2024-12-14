// Copyright (c) 2024 Durandin Vladimir

#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>

#define VERIFY_CL_CALL(call)                                                   \
  {                                                                            \
    auto errorCode = call;                                                     \
    if (errorCode != CL_SUCCESS) {                                             \
      std::cerr << "\033[1;31mOpenCL Error:\033[0m ";                          \
      std::cerr << "Error Code: " << static_cast<int>(errorCode) << '\n';      \
      std::cerr << "Location: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(static_cast<int>(errorCode));                                  \
    }                                                                          \
  }

std::vector<float> GeluOCL(const std::vector<float> &input) {
  std::vector<cl::Platform> availablePlatforms;
  cl::Platform::get(&availablePlatforms);
  if (availablePlatforms.empty()) {
    std::cerr << "\033[1;31mOpenCL Error:\033[0m No platforms available\n";
    return {};
  }
  cl::Platform selectedPlatform = availablePlatforms.front();

  std::vector<cl::Device> availableDevices;
  selectedPlatform.getDevices(CL_DEVICE_TYPE_GPU, &availableDevices);
  if (availableDevices.empty()) {
    std::cerr << "\033[1;31mOpenCL Error:\033[0m No devices available\n";
    return {};
  }
  cl::Device selectedDevice = availableDevices.front();

  cl::Context clContext(selectedDevice);
  cl::CommandQueue commandQueue(clContext);

  std::string kernelCode = R"(
__kernel void gelu_kernel(__global const float* x, __global float* y, int countElem) {
  int i = get_global_id(0);

  if (i < countElem) {
      float val = x[i];
      float tmp = val * (1.595769122f + val * val * 0.071354816f);
      y[i] = val - val / (1.0f + exp(tmp));
  }
}
)";

  cl::Program::Sources sourceCode;
  sourceCode.emplace_back(std::move(kernelCode));

  cl::Program clProgram(clContext, sourceCode);
  if (clProgram.build() != CL_SUCCESS) {
    std::cerr << "\033[1;31mOpenCL Build Error:\033[0m ";
    std::cerr << clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selectedDevice)
              << '\n';
    return {};
  }

  if (input.empty())
    return {};
  auto elementCount = input.size();
  auto bufferSize = elementCount * sizeof(float);

  cl::Buffer inputBuffer(clContext, CL_MEM_READ_ONLY, bufferSize);
  cl::Buffer outputBuffer(clContext, CL_MEM_WRITE_ONLY, bufferSize);

  VERIFY_CL_CALL(commandQueue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0,
                                                 bufferSize, input.data()));

  cl::Kernel geluKernel(clProgram, "gelu_kernel");
  geluKernel.setArg(0, inputBuffer);
  geluKernel.setArg(1, outputBuffer);
  geluKernel.setArg(2, static_cast<int>(elementCount));
  VERIFY_CL_CALL(commandQueue.enqueueNDRangeKernel(
      geluKernel, cl::NullRange, cl::NDRange(elementCount), cl::NullRange));

  std::vector<float> result(elementCount);
  VERIFY_CL_CALL(commandQueue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0,
                                                bufferSize, result.data()));

  return result;
}
