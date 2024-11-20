#include "gelu_ocl.h"

#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include <cstddef>
#include <stdexcept>

const char *gelu_kernel = R"(

    #define COEF1 1.595769122f
    #define COEF2 0.071354816f

    __kernel void gelu(__global float *input, __global float *output, int size) {
        int i = get_global_id(0);

        if (i < size) {
            float x = input[i];
            float expon = exp(x * fma(COEF2, pown(x, 2), COEF1));

            output[i] = x * expon / (1.0f + expon);
        }
    }
)";

std::vector<float> GeluOCL(const std::vector<float> &input) {
  std::vector<float> output(input.size());
  cl_platform_id platformId;
  cl_uint num_platforms;
  clGetPlatformIDs(1, &platformId, &num_platforms);

  if (num_platforms == 0) {
    throw std::runtime_error("No OpenCL platforms found");
  }

  cl_device_id device;
  cl_uint numDevices;
  clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
  if (numDevices == 0) {
    throw std::runtime_error("No OpenCL GPU devices found");
  }

  auto context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

  auto queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

  auto inputBuffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     input.size() * sizeof(float), (void *)input.data(), NULL);

  auto outputBuffer =
      clCreateBuffer(context, CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY,
                     output.size() * sizeof(float), NULL, NULL);

  auto program =
      clCreateProgramWithSource(context, 1, &gelu_kernel, NULL, NULL);

  if (clBuildProgram(program, 1, &device, "", NULL, NULL) != CL_SUCCESS) {
    throw std::runtime_error("Error building program");
  }

  auto kernel = clCreateKernel(program, "gelu", NULL);

  auto inputSize = input.size();
  int intSize = inputSize;

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
  clSetKernelArg(kernel, 2, sizeof(int), &intSize);

  size_t workGroupSize = 8;

  auto err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &inputSize, &workGroupSize, 0,
                         NULL, NULL);
  if (err!= CL_SUCCESS) {
    throw std::runtime_error("Error enqueuing kernel: " + std::to_string(err));
  }
  clFinish(queue);

  err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0,
                      inputSize * sizeof(float), output.data(), 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Error reading output buffer: " + std::to_string(err));
  }
  
  clReleaseMemObject(inputBuffer);
  clReleaseMemObject(outputBuffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return output;
}
