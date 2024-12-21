#include "gelu_ocl.h"

#define OCL_CHECK(status, msg) \
    if (status != CL_SUCCESS) { \
        std::cerr << "OpenCL Error: " << msg << " Error Code: " << status << " at line " << __LINE__ << '\n'; \
        exit(EXIT_FAILURE); \
    }

const char* gelu_kernel_source = R"(
__kernel void gelu(__global float* input, __global float* output, int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
    }
}
)";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    cl_int err;
    size_t size = input.size();
    std::vector<float> output(size);
    cl_platform_id platform;

    err = clGetPlatformIDs(1, &platform, nullptr);
    OCL_CHECK(err, "Getting platform ID");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err == CL_DEVICE_NOT_FOUND) {
        std::cerr << "GPU device not found. Trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        OCL_CHECK(err, "Getting CPU device ID");
    } else {
        OCL_CHECK(err, "Getting GPU device ID");
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    OCL_CHECK(err, "Creating context");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    OCL_CHECK(err, "Creating command queue");
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                        size * sizeof(float), (void*)input.data(), &err);
    OCL_CHECK(err, "Creating input buffer");
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                         size * sizeof(float), nullptr, &err);
    OCL_CHECK(err, "Creating output buffer");
    cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
    OCL_CHECK(err, "Creating program");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        std::cerr << "Error: Building program failed.\nBuild Log:\n" << build_log.data() << std::endl;
        clReleaseProgram(program);
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    OCL_CHECK(err, "Creating kernel");
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    OCL_CHECK(err, "Setting kernel argument 0 (input)");
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    OCL_CHECK(err, "Setting kernel argument 1 (output)");
    err  = clSetKernelArg(kernel, 2, sizeof(int), &size);
    OCL_CHECK(err, "Setting kernel argument 2 (size)");

    size_t global_work_size = size;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr,
                                 0, nullptr, nullptr);
    OCL_CHECK(err, "Enqueuing kernel");
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, size * sizeof(float),
                              output.data(), 0, nullptr, nullptr);
    OCL_CHECK(err, "Reading output buffer");

    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}