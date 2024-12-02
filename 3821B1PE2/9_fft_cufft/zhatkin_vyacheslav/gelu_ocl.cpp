#include "gelu_ocl.h"
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

// OpenCL kernel code (stored as a string)
const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input, __global float* output, const unsigned int size) {
    int id = get_global_id(0);
    if (id < size) {
        float x = input[id];
        float result = 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
        output[id] = result;
    }
}
)";

// Helper function to check OpenCL errors
void checkError(cl_int err, const std::string& msg) {
    if (err != CL_SUCCESS) {
        std::cerr << msg << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Helper function to read OpenCL kernel from a file
std::string readKernelSource(const char* filename) {
    std::ifstream kernelFile(filename);
    std::stringstream buffer;
    buffer << kernelFile.rdbuf();
    return buffer.str();
}

std::vector<float> GeluOCL(const std::vector<float>& input) {
    cl_int err;

    // 1. Select Platform and Device
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkError(err, "Failed to get platform");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_GPU, 1, &device, nullptr);
    checkError(err, "Failed to get device");

    // 2. Create OpenCL context and queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "Failed to create context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Failed to create command queue");

    // 3. Create OpenCL buffer for input and output
    size_t inputSize = input.size() * sizeof(float);
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, inputSize, nullptr, &err);
    checkError(err, "Failed to create input buffer");

    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inputSize, nullptr, &err);
    checkError(err, "Failed to create output buffer");

    // 4. Write input data to buffer
    err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, inputSize, input.data(), 0, nullptr, nullptr);
    checkError(err, "Failed to write to input buffer");

    // 5. Compile and build the OpenCL kernel
    cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
    checkError(err, "Failed to create program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        char* log = new char[logSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        std::cerr << "OpenCL Build Error: " << log << std::endl;
        delete[] log;
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu_kernel", &err);
    checkError(err, "Failed to create kernel");

    // 6. Set kernel arguments
    unsigned int size = static_cast<unsigned int>(input.size());
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    checkError(err, "Failed to set kernel argument 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    checkError(err, "Failed to set kernel argument 1");
    err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &size);
    checkError(err, "Failed to set kernel argument 2");

    // 7. Execute the kernel
    size_t globalWorkSize = input.size();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkError(err, "Failed to enqueue kernel");

    // 8. Read the results from the device
    std::vector<float> output(input.size());
    err = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, inputSize, output.data(), 0, nullptr, nullptr);
    checkError(err, "Failed to read from output buffer");

    // 9. Clean up OpenCL resources
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}