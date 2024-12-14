#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cufft.h>
#include <cassert>

std::vector<float> FffCUFFT(const std::vector<float> &input, int batch);

#endif // __FFT_CUFFT_H
