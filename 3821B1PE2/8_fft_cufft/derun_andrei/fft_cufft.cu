#include "fft_cufft.h"

void convertFloatToCufftComplex(float *d_input, cufftComplex *d_input_complex, int size)
{
    for (int i = 0; i < size; i++)
    {
        d_input_complex[i].x = d_input[2 * i];
        d_input_complex[i].y = d_input[2 * i + 1];
    }
}

void convertCufftComplexToFloat(cufftComplex *d_input_complex, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[2 * i] = d_input_complex[i].x;
        output[2 * i + 1] = d_input_complex[i].y;
    }
}

void normalizeCufftComplex(cufftComplex *data, float factor, int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i].x /= factor;
        data[i].y /= factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch)
{
    assert(input.size() % (2 * batch) == 0);
    int n = input.size() / (2 * batch);

    float *d_input;
    cudaMalloc((void **)&d_input, input.size() * sizeof(float));
    cufftComplex *d_output;
    cudaMalloc((void **)&d_output, n * batch * sizeof(cufftComplex));
    cufftHandle plan;
    cufftPlanMany(&plan,
                  1,         // rank (number of dimensions in the multidimensional FFT)
                  &n,        // *n (array of sizes for each dimension)
                  NULL,      // inembed (in-embedding attributes for complex inputs, or NULL for default)
                  1,         // istride (stride for complex inputs, or 1 for default sequential storage)
                  n,         // idist (distance between first elements of two consecutive batches, or n for default)
                  NULL,      // onembed (out-embedding attributes for complex outputs, or NULL for default)
                  1,         // ostride (stride for complex outputs, or 1 for default sequential storage)
                  n,         // odist (distance between first elements of two consecutive batches in output, or n for default)
                  CUFFT_C2C, // type (transform type, here: C2C for complex-to-complex)
                  batch);    // batch (number of batches)

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    cufftComplex *d_input_complex;
    cudaMalloc((void **)&d_input_complex, n * batch * sizeof(cufftComplex));
    convertFloatToCufftComplex(d_input, d_input_complex, n * batch);

    cufftExecC2C(plan, d_input_complex, d_output, CUFFT_FORWARD);

    cufftExecC2C(plan, d_output, d_input_complex, CUFFT_INVERSE);

    normalizeCufftComplex(d_input_complex, n, n * batch);

    std::vector<float> output(2 * n * batch);
    convertCufftComplexToFloat(d_input_complex, output.data(), n * batch);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input_complex);
    cufftDestroy(plan);

    return output;
}
