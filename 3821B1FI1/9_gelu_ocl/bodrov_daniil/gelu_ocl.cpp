#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <mutex>

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
        // Get all platforms (drivers)
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        // Use platform 0
        cl::Platform platform = platforms[0];

        // Get GPU devices
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices found.");
        }

        device = devices.front();

        // Create context and command queue
        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        // OpenCL kernel code
        const std::string kernelSource = R"CLC(
        __kernel void gelu_activation(__global const float* input, __global float* output, const int size) {
            int gid = get_global_id(0);
            if (gid >= size) return;
            float x = input[gid];
            // GELU computation
            float y = 0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
            output[gid] = y;
        }
        )CLC";

        // Build program
        cl::Program::Sources sources;
        sources.emplace_back(std::move(kernelSource));
        program = cl::Program(context, sources);
        if (program.build() != CL_SUCCESS) {
            std::cerr << "Build log:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        }

        // Create kernel
        kernel = cl::Kernel(program, "gelu_activation");

        // Create buffers
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

    // Initialize OpenCL once
    try {
        std::call_once(initFlag, initializeOpenCL, bufferBytes);
    } catch (const std::exception& e) {
        std::cerr << "OpenCL initialization error: " << e.what() << std::endl;
        return {};
    }

    // If buffers are smaller than needed, re-create them
    if (bufferInput.getInfo<CL_MEM_SIZE>() < bufferBytes) {
        bufferInput = cl::Buffer(context, CL_MEM_READ_ONLY, bufferBytes);
        bufferOutput = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferBytes);
    }

    // Copy data to input buffer
    queue.enqueueWriteBuffer(bufferInput, CL_FALSE, 0, bufferBytes, input.data());

    // Set kernel arguments
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, static_cast<int>(dataSize));

    // Determine global and local work sizes
    size_t localRangeSize = 256;
    size_t globalRangeSize = ((dataSize + localRangeSize - 1) / localRangeSize) * localRangeSize;
    cl::NDRange globalRange(globalRangeSize);
    cl::NDRange localRange(localRangeSize);

    // Execute kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);

    // Read results
    std::vector<float> output(dataSize);
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, bufferBytes, output.data());

    return output;
}