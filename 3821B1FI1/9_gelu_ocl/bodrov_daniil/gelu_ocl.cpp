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
        // Инициализация платформы и устройства
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        device = devices.front();

        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        // Код ядра
        const std::string kernelSource = R"CLC(
        __kernel void gelu_activation(__global const float* input, __global float* output, const int size) {
            int gid = get_global_id(0);
            if (gid >= size) return;
            float x = input[gid];
            float y = x * (0.7978846f * (1.0f + 0.044715f * x * x));
            output[gid] = 0.5f * x * (1.0f + y);
        }
        )CLC";

        // Компиляция программы
        cl::Program::Sources sources;
        sources.emplace_back(kernelSource);
        program = cl::Program(context, sources);
        program.build({ device });

        kernel = cl::Kernel(program, "gelu_activation");

        // Создание буферов
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

    // Инициализация OpenCL один раз
    try {
        std::call_once(initFlag, initializeOpenCL, bufferBytes);
    } catch (const std::exception& e) {
        std::cerr << "OpenCL initialization error: " << e.what() << std::endl;
        return {};
    }

    // Если буферы меньше необходимого размера, переинициализируем их
    if (bufferInput.getInfo<CL_MEM_SIZE>() < bufferBytes) {
        bufferInput = cl::Buffer(context, CL_MEM_READ_ONLY, bufferBytes);
        bufferOutput = cl::Buffer(context, CL_MEM_WRITE_ONLY, bufferBytes);
    }

    // Копирование данных во входной буфер
    queue.enqueueWriteBuffer(bufferInput, CL_FALSE, 0, bufferBytes, input.data());

    // Установка аргументов ядра
    kernel.setArg(0, bufferInput);
    kernel.setArg(1, bufferOutput);
    kernel.setArg(2, static_cast<int>(dataSize));

    // Оптимизация размеров рабочих групп
    size_t localRangeSize = 256;
    cl::NDRange localRange(localRangeSize);
    cl::NDRange globalRange(((dataSize + localRangeSize - 1) / localRangeSize) * localRangeSize);

    // Запуск ядра
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);

    // Чтение результата
    std::vector<float> output(dataSize);
    queue.enqueueReadBuffer(bufferOutput, CL_TRUE, 0, bufferBytes, output.data());

    return output;
}