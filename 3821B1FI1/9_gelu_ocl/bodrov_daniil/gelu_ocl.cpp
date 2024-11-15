#include "gelu_ocl.h"

#include <CL/opencl.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>

// Кешированные объекты OpenCL для повторного использования
namespace {
    cl::Context cachedContext;
    cl::CommandQueue cachedQueue;
    cl::Kernel cachedKernel;
    std::once_flag initFlag;

    const std::string kernelSource = R"CLC(
    __kernel void gelu_activation(__global const float* input, __global float* output, const int size) {
        int gid = get_global_id(0);
        if (gid >= size) return;
        float x = input[gid];
        float cdf = 0.5f * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)));
        output[gid] = x * cdf;
    }
    )CLC";

    void initializeOpenCL() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        cl::Platform platform = platforms[0];

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No GPU devices found on the first platform.");
        }

        cl::Device device = devices[0];

        cachedContext = cl::Context(device);
        cachedQueue = cl::CommandQueue(cachedContext, device, CL_QUEUE_PROFILING_ENABLE);

        cl::Program::Sources sources;
        sources.emplace_back(kernelSource);

        cl::Program program(cachedContext, sources);
        program.build({ device });

        cachedKernel = cl::Kernel(program, "gelu_activation");
    }
}

std::vector<float> GeluOCL(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    // Инициализация OpenCL один раз
    try {
        std::call_once(initFlag, initializeOpenCL);
    } catch (const std::exception& e) {
        std::cerr << "OpenCL initialization error: " << e.what() << std::endl;
        return {};
    }

    size_t dataSize = input.size();
    size_t bufferBytes = dataSize * sizeof(float);

    
    // Создание буферов
    cl::Buffer bufferInput(cachedContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferBytes, const_cast<float*>(input.data()));
    cl::Buffer bufferOutput(cachedContext, CL_MEM_WRITE_ONLY, bufferBytes);

    // Установка аргументов ядра
    cachedKernel.setArg(0, bufferInput);
    cachedKernel.setArg(1, bufferOutput);
    cachedKernel.setArg(2, static_cast<int>(dataSize));

    // Определение размера работы
    cl::NDRange globalRange((dataSize + 255) / 256 * 256); // Выравнивание до 256
    cl::NDRange localRange(256);

    // Запуск ядра
    cl::Event event;
    cachedQueue.enqueueNDRangeKernel(cachedKernel, cl::NullRange, globalRange, localRange, nullptr, &event);

    // Ожидание завершения
    event.wait();

    // Чтение результата
    std::vector<float> output(dataSize);
    cachedQueue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, bufferBytes, output.data());

    return output;

}