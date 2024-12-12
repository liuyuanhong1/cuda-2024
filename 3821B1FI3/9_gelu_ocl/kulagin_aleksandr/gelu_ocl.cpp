// Copyright (c) 2024 Kulagin Aleksandr
#include "gelu_ocl.h"
#define _USE_MATH_DEFINES
#include <CL/opencl.hpp>
#include <iostream>
#include <cmath>

static const float precalc_c_1 = std::sqrt(2.0f / M_PIf);

static const char* gelu_ocl_source = R"(
__kernel void gelu_ocl_kernel(__global float* input_output, const int n, const float precalc_c_1) {
  int i = get_global_id(0);
  if (i < n) {
    const float x = input_output[i];
    input_output[i] = 0.5f * x * (1.0f + tanh(precalc_c_1 * ( x + 0.044715f * (x * x * x) )));
  }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
  if (input.empty()) {
    return std::vector<float>();
  }
  cl_int err;
  std::vector<cl::Platform> platforms;
  err = cl::Platform::get(&platforms);
  if (err) {
    std::cerr << "Error on get platforms " << err << '\n';
    return std::vector<float>();
  }
  if (platforms.empty()) {
    std::cerr << "No platforms are available. Too bad.\n";
    return std::vector<float>();
  }
  cl::Platform& platform = platforms.front();
  std::vector<cl::Device> devices;
  err = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  if (err) {
    std::cerr << "Error on get devices " << err << '\n';
    return std::vector<float>();
  }
  if (devices.empty()) {
    std::cerr << "No devices are available. Too bad.\n";
    return std::vector<float>();
  }
  cl::Device& device = devices.front();
  cl::Context context(device, nullptr, nullptr, nullptr, &err);
  if (err) {
    std::cerr << "Error on create context " << err << '\n';
    return std::vector<float>();
  }
  cl::CommandQueue queue(context, 0, &err);
  if (err) {
    std::cerr << "Error on create queue " << err << '\n';
    return std::vector<float>();
  }
  cl::Program::Sources sources;
  sources.push_back(gelu_ocl_source);
  cl::Program program(context, sources, &err);
  if (err) {
    std::cerr << "Error on create program " << err << '\n';
    return std::vector<float>();
  }
  if ((err = program.build())) {
    std::cerr << "Error on compiling opencl program: " << err << '\n';
    return std::vector<float>();
  }
  unsigned sz = input.size();
  unsigned sz_bytes = sz * sizeof(float);
  std::vector<float> res(sz);
  cl::Buffer buffer(context, CL_MEM_READ_WRITE, sz_bytes, nullptr, &err);
  if (err) {
    std::cerr << "Error on create buffer " << err << '\n';
    return std::vector<float>();
  }
  if ((err = queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sz_bytes, input.data()))) {
    std::cerr << "Error on write buffer " << err << '\n';
    return std::vector<float>();
  }
  cl::Kernel kernel(program, "gelu_ocl_kernel");
  if ((err = kernel.setArg(0, buffer))) {
    std::cerr << "Error on setting arg 0 " << err << '\n';
    return std::vector<float>();
  }
  if ((err = kernel.setArg(1, static_cast<int>(sz)))) {
    std::cerr << "Error on setting arg 1 " << err << '\n';
    return std::vector<float>();
  }
  if ((err = kernel.setArg(2, precalc_c_1))) {
    std::cerr << "Error on setting arg 2 " << err << '\n';
    return std::vector<float>();
  }
  if ((err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(sz), cl::NullRange))) {
    std::cerr << "Error on running kernel " << err << '\n';
    return std::vector<float>();
  }
  if ((err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sz_bytes, res.data()))) {
    std::cerr << "Error on read buffer " << err << '\n';
    return std::vector<float>();
  }
  return res;
}
