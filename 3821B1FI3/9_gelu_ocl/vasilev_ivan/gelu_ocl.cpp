#define CL_HPP_TARGET_OPENCL_VERSION 120  // OpenCL 1.2
#define CL_HPP_MINIMUM_OPENCL_VERSION 120  // OpenCL 1.2
#include <CL/opencl.hpp>


#include "gelu_ocl.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <cstddef>





#define CHECK_CL_ERROR(callable)                                          \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != CL_SUCCESS) {                                        \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(static_cast<int>(codeError));                             \
    }                                                                     \
  }

std::vector<float> GeluOCL(const std::vector<float>& input) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    std::cerr << "\033[1;31merror\033[0m: no platform available\n";
    return {};
  }
  cl::Platform platform = platforms.front();

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (devices.empty()) {
    std::cerr << "\033[1;31merror\033[0m: no device available\n";
    return {};
  }
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
  sources.push_back({codeKernel.c_str(), codeKernel.length()});

  cl::Program program(context, sources);
  CHECK_CL_ERROR(program.build());


  cl::Buffer bufferInput(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
  cl::Buffer bufferOutput(context, CL_MEM_WRITE_ONLY, sizeof(float) * input.size());

  CHECK_CL_ERROR(queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, sizeof(float) * input.size(), input.data()));


  cl::Kernel kernel(program, "gelu_kernel");
  CHECK_CL_ERROR(kernel.setArg(0, bufferInput));
  CHECK_CL_ERROR(kernel.setArg(1, bufferOutput));
  CHECK_CL_ERROR(kernel.setArg(2, static_cast<int>(input.size())));

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.size()), cl::NullRange);
  queue.finish();


  std::vector<float> result(input.size());
  CHECK_CL_ERROR(queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, sizeof(float) * result.size(), result.data()));

  return result;
}