// Copyright (c) 2024 Moiseev Nikita
#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <vector>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch);

#endif  // __FFT_CUFFT_H