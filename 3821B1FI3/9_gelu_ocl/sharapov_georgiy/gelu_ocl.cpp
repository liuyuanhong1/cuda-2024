#include <CL/opencl.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input) {
  const char* kernel_source = R"(
  __kernel void gelu(__global const float* input, __global float* output, int size, float var) {
      int i = get_global_id(0);
      if (i < size) {
          float x = input[i];
          output[i] = 0.5f * x * (1.0f + tanh(var * x * (1.0f + 0.044715f * x * x)));
      }
  }
  )";
  size_t size = input.size();
  size_t bytes_count = size * sizeof(float);
  std::vector<float> result(input.size(), 0.0f);
  float sqrtdvadelpi = sqrt(2.0f / M_PI);

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  cl::Platform platform = platforms[0];
  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
  cl::Device device = devices[0];

  cl::Context context(device);
  cl::CommandQueue queue(context);

  cl::Program::Sources sources;
  sources.emplace_back(kernel_source);
  cl::Program program(context, sources);
  program.build();

  cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, bytes_count);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, bytes_count);

  queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, bytes_count, input.data());

  cl::Kernel kernel(program, "gelu");
  kernel.setArg(0, input_buffer);
  kernel.setArg(1, output_buffer);
  kernel.setArg(2, static_cast<int>(size));
  kernel.setArg(3, static_cast<float>(sqrtdvadelpi));

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size),
                             cl::NullRange);
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, bytes_count,
                          result.data());

  return result;
}
