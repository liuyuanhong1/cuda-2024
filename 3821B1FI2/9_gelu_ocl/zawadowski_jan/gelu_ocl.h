#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <cmath>

std::vector<float> GeluOCL(const std::vector<float>& input);

#endif // __GELU_OCL_H