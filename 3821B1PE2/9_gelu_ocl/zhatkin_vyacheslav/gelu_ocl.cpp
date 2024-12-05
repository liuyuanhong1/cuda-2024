#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// OpenCL GELU Kernel as a string constant
const char* geluKernelSource = R"CLC(
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const unsigned int n)
{
    int id = get_global_id(0);
    if (id < n) {
        float x = input[id];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float c1 = 0.7978845608f; // sqrt(2/pi)
        float c2 = 0.035677408f;  // 0.044715
        float gelu = 0.5f * x * (1.0f + tanh(c1 * (x + c2 * x * x * x)));
        output[id] = gelu;
    }
}
)CLC";

std::vector<float> GeluOCL(const std::vector<float>& input) {
    // Размер входных данных
    size_t n = input.size();
    std::vector<float> output(n, 0.0f);

    cl_int err;

    // Получение платформ
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return output;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get OpenCL platform IDs." << std::endl;
        return output;
    }

    // Выбор первой платформы
    cl_platform_id platform = platforms[0];

    // Получение устройств GPU на платформе
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        std::cerr << "Failed to find any GPU devices." << std::endl;
        return output;
    }

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to get GPU device IDs." << std::endl;
        return output;
    }

    // Выбор первого устройства
    cl_device_id device = devices[0];

    // Создание контекста
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return output;
    }

    // Создание командного очереди
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL command queue." << std::endl;
        clReleaseContext(context);
        return output;
    }

    // Создание буферов
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(float) * n, const_cast<float*>(input.data()), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create input buffer." << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        sizeof(float) * n, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create output buffer." << std::endl;
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    // Создание и компиляция программы
    cl_program program = clCreateProgramWithSource(context, 1, &geluKernelSource, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL program." << std::endl;
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Получение и вывод логов компиляции
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Failed to build OpenCL program. Build log:\n" << log.data() << std::endl;
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    // Создание ядра
    cl_kernel kernel = clCreateKernel(program, "gelu", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL kernel." << std::endl;
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    // Установка аргументов ядра
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &n);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set OpenCL kernel arguments." << std::endl;
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    // Определение размера глобальной рабочей группы
    size_t global_work_size = ((n + 255) / 256) * 256; // Выравнивание до ближайшего 256

    // Запуск ядра
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to enqueue OpenCL kernel." << std::endl;
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(outputBuffer);
        clReleaseMemObject(inputBuffer);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return output;
    }

    // Чтение результатов
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(float) * n, output.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to read OpenCL buffer." << std::endl;
        // Продолжаем, так как данные могут быть частично заполнены
    }

    // Освобождение ресурсов
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(inputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return output;
}