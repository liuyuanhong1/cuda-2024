// Copyright (c) 2024 Moiseev Nikita
#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <vector>

std::vector<float> FFTCUFFT(const std::vector<float>& input_data, int batch_size);

#endif  // __FFT_CUFFT_H
