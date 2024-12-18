#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <mutex>

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

namespace {
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Device device;
    cl::Buffer bufferInput;
    cl::Buffer bufferOutput;
    std::once_flag initFlag;

    void initializeOpenCL(size_t bufferBytes) {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        cl::Platform platform = platforms.front();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices found.");
        }

        device = devices.front();

        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        const std::string kernelSource = R"CLC(
        __kernel void gelu_kernel(__global const float* x, __global float* y, int countElem) {
            int i = get_global_id(0);

            if (i < countElem) {
                float val = x[i];
                float tmp = val * (1.595769122f + val * val * 0.071354816f);
                y[i] = val - val / (1.0f + exp(tmp));
            }
        }
        )CLC";

        cl::Program::Sources sources;
        sources.emplace_back(std::move(kernelSource));
        program = cl::Program(context, sources);
        if (program.build() != CL_SUCCESS) {
            std::cerr << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw std::runtime_error("Failed to build OpenCL program.");
        }

        kernel = cl::Kernel(program, "gelu_kernel");

        bufferInput = cl::Buffer(context, CL_MEM_READ_ONLY, bufferBytes);
        bufferOutput = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferBytes);
    }
}

std::vector<float> GeluOCL(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    size_t dataSize = input.size();
    size_t bufferBytes = dataSize * sizeof(float);

    try {
        std::call_once(initFlag, initializeOpenCL, bufferBytes);
    } catch (const std::exception& e) {
        std::cerr << "OpenCL initialization error: " << e.what() << std::endl;
        return {};
    }

    if (bufferInput.getInfo<CL_MEM_SIZE>() != bufferBytes) {
        bufferInput = cl::Buffer(context, CL_MEM_READ_ONLY, bufferBytes);
        bufferOutput = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferBytes);
    }

    CHECK_CL_ERROR(queue.enqueueWriteBuffer(bufferInput, CL_TRUE, 0, bufferBytes, input.data()));

    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, static_cast<int>(dataSize));

    size_t localRangeSize = 256;
    size_t maxLocalSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    localRangeSize = std::min(localRangeSize, maxLocalSize);
    size_t globalRangeSize = ((dataSize + localRangeSize - 1) / localRangeSize) * localRangeSize;

    CHECK_CL_ERROR(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(globalRangeSize), cl::NDRange(localRangeSize)));
    queue.finish();

    std::vector<float> output(dataSize);
    CHECK_CL_ERROR(queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, bufferBytes, output.data()));

    return output;
}