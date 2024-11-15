#include "gelu_ocl.h"

#include <CL/cl.h>
#include <cmath>

std::vector<float> GeluOCL(const std::vector<float>& input) {
  const char* kernel_source =
R"(__kernel void gelu_kernel(__global float* a, __global float* res, const int n) {
	int i = get_global_id(0);
	if (i < n) {
		float x = a[i];
		res[i] = 0.5f * x * (1.f + tanhf(0.7978845608028653f * x * (1.0f + 0.044715f * x * x)));
	}
})";
  
  size_t sz = static_cast<int>(input.size());
  std::vector<float> output(sz);

  cl_platform_id platform;
  cl_device_id device;

  clGetPlatformIDs(1, &platform, nullptr);

  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);

  cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

  cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * sz, nullptr, nullptr);
  cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * sz, nullptr, nullptr);
  clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float) * sz, input.data(), 0, nullptr, nullptr);

  cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);
  clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  cl_kernel kernel = clCreateKernel(program, "gelu_kernel", nullptr);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
  clSetKernelArg(kernel, 2, sizeof(int), &sz);

  clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &sz, nullptr, 0, nullptr, nullptr);
  clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float) * sz, output.data(), 0, nullptr, nullptr);

  clReleaseMemObject(input_buffer);
  clReleaseMemObject(output_buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}
