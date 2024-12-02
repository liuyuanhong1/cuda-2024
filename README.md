# Content
- [How To](#how-to)
- [Configuration](#configuration)
- [Time Measurement](#time-measurement)
- [Tasks](#tasks)
- [Results](#results)

# How To
1. Create [github](https://github.com/) account (if not exists);
2. Make sure SSH clone & commit is working ([Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh));
3. Fork this repo (just click **Fork** button on the top of the page, detailed instructions [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project))
4. Clone your forked repo into your local machine, use your user instead of `username`:
```sh
git clone git@github.com:username/cuda-2024.git
cd cuda-2024
```
5. Go to your group folder, e.g.:
```sh
cd 3821B1FI1
```
6. Go to needed task folder, e.g.:
```sh
cd 1_gelu_omp
```
7. Create new folder with your surname and name (**make sure it's the same for all tasks**), e.g.:
```sh
mkdir petrov_ivan
```
8. Copy your task source/header files (including main program) into this folder (use `copy` instead of `cp` on Windows), e.g.:
```sh
cd petrov_ivan
cp /home/usr/lab/*.cpp .
cp /home/usr/lab/*.h .
```
8. Push your sources to github repo, e.g.:
```sh
cd ..
git add .
git commit -m "1_gelu_omp task"
git push
```
9. Go to your repo in browser, click **Contribute** button on the top of page, then **Open pull request**. Provide meaningfull request title and description, then **Create pull request** (see details [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)).
10. Go to Pull Requests [page](https://github.com/avgorshk/cuda-2024/pulls) in course repo, find your pull request and check if there are no any merge conflicts occur. If merge conflicts happen - resolve it following the instruction provided by github.

# Time Measurement
The following scheme is used to measure task execution time:
```cpp
int main() {
    // ...

    // Warming-up
    Task(input, size / 8);

    // Performance Measuring
    auto start = std::chrono::high_resolution_clock::now();
    auto c = Task(input, size);
    auto end = std::chrono::high_resolution_clock::now();

    // ...
}
```

# Configuration
- CPU: Intel Core i5 12600K (4 cores, 4 threads)
- RAM: 16 GB
- GPU: NVIDIA RTX 4060 (8 GB)
- Host Compiler: GCC 11.4.0
- CUDA: 12.6

# Tasks
## Task #1: OpenMP GELU Implementation
The **Gaussian Error Linear Unit (GELU)** is an activation function frequently used in Deep Neural Networks (DNNs) and can be thought of as a smoother ReLU.

To approximate GELU function, use the following formula:

GELU(x) =  $0.5x(1 + tanh(\sqrt{2 / \pi}(x + 0.044715 * x^3)))$

Implement the function with the following interface in C++:
```cpp
std::vector<float> GeluOMP(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use OpenMP technology to make your function parallel & fast.

Two files are expected to be uploaded:
- gelu_omp.h
```cpp
#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H
```
- gelu_omp.cpp
```cpp
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // Place your implementation here
}
```
## Task #2: CUDA GELU Implementation
Implement the function with the following interface in CUDA C++ using the formula described above:
```cpp
std::vector<float> GeluCUDA(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use CUDA technology to make your function work on NVIDIA GPU. Try to make it fast.

Two files are expected to be uploaded:
- gelu_cuda.h
```cpp
#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H
```
- gelu_cuda.cu
```cpp
#include "gelu_cuda.h"

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    // Place your implementation here
}
```

## Task #3: Naive Matrix Multiplication using OpenMP
General matrix multiplication (GEMM) is a very basic and broadly used linear algebra operation applied in high performance computing (HPC), statistics, deep learning and other domains. There are a lot of GEMM algorithms with different mathematical complexity form $O(n^3)$ for naive and block approaches to $O(n^{2.371552})$ for the method descibed by Williams et al. in 2024 [[1](https://epubs.siam.org/doi/10.1137/1.9781611977912.134)]. But despite a variety of algorithms with low complexity, block matrix multiplication remains the most used implementation in practice since it fits to modern HW better.

To start learning matrix multiplication smoother, let us start with naive approach here. To compute matrix multiplication result C for matricies A and B, where C = A * B and the size for all matricies are $n*n$, one should use the following formula for each element of C (will consider only square matricies for simplicity):

$c_{ij}=\sum_{k=1}^na_{ik}b_{kj}$

To complete the task one should implement a function that multiplies two square matricies using OpenMP with the following interface:
```cpp
std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);
```
Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- naive_gemm_omp.h:
```cpp
#ifndef __NAIVE_GEMM_OMP_H
#define __NAIVE_GEMM_OMP_H

#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);

#endif // __NAIVE_GEMM_OMP_H
```
- naive_gemm_omp.cpp:
```cpp
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    // Place your implementation here
}
```

## Task #4: Naive Matrix Multiplication using CUDA
In this task one should implement naive approach for matrix multiplication in CUDA trying to make it fast enough *(pay attention to global memory accesses in your code)*.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- naive_gemm_cuda.h:
```cpp
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __NAIVE_GEMM_CUDA_H
```
- naive_gemm_cuda.cu:
```cpp
#include "naive_gemm_cuda.h"

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```

## Task #5: Block Matrix Multiplication using OpenMP
In real applications block-based approach for matrix multiplication can get multiple times faster execution comparing with naive version due to cache friendly approach. To prove this in practice, implement such a version in C++ using OpenMP.

In block version algorithm could be divided into three stages:
1. Split matricies into blocks (block size normally affects performance significantly so choose it consciously);
2. Multiply two blocks to get partial result;
3. Replay step 2 for all row/column blocks accumulating values into a single result block.

From math perspective, block matrix multiplication could be described by the following formula, where $C_{IJ}$, $A_{IK}$ and $B_{KJ}$ are sub-matricies with the size $block\_size*block\_size$:

$C_{IJ}=\sum_{k=1}^{block_count}A_{IK}B_{KJ}$

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- block_gemm_omp.h:
```cpp
#ifndef __BLOCK_GEMM_OMP_H
#define __BLOCK_GEMM_OMP_H

#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n);

#endif // __BLOCK_GEMM_OMP_H
```
- block_gemm_omp.cpp:
```cpp
#include "block_gemm_omp.h"

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    // Place your implementation here
}
```

As in previous task, let us consider all matricies are square.

## Task #6: Block Matrix Multiplication using CUDA
In CUDA C++ block-based approach looks similar. But to get better performance one should use CUDA shared memory to store each particular block while computations. With this consideration, algorithm will be the following:
1. A single CUDA block should compute a single block of result matrix C, a single CUDA thread - a single matrix C element;
2. For each A block in a row and B block in a column:
    1. Load A block into shared memory;
    2. Load B block into shared memory;
    3. Synchronize over all threads in block;
    4. Compute BlockA * BlockB and accumulate into C block in shared memory;
    5. Synchronize over all threads in block;
3. Dump block C from shared to global memory.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- block_gemm_cuda.h:
```cpp
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __BLOCK_GEMM_CUDA_H
```
- block_gemm_cuda.cu:
```cpp
#include "block_gemm_cuda.h"

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```

## Task #7: Matrix Multiplication using cuBLAS
The most performant way to multiply two matrices on particular hardware is to use vendor-provided library for this purpose. In CUDA it's [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html). Try to use cuBLAS API to implement general matrix multiplication in most performant way.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Note, that in cuBLAS API matrix is expected to be stored by columns, so additional transpose may be required.

Two files are expected to be uploaded:
- gemm_cublas.h:
```cpp
#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n);

#endif // __GEMM_CUBLAS_H
```
- gemm_cublas.cu:
```cpp
#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Place your implementation here
}
```

## Task #8: FFT (Fast Fourier Transform) using cuFFT
Another widely used operation in HPC & signal processing is discrete [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform). Naive approach (by definition) has $O(n^2)$ complexity and is not used in practice due to its slowness. Better way is [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform) algorithm with $O(n*log(n))$ complexity.

Due to its frequent use, FFT algorithm implementation is normally a part of vendor-optimized solutions for various hardware chips. For NVIDIA GPUs one should take [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) library.

To pass the task one should implement a funtion that takes $batch$ signals of $n$ complex elements, and performs complex-to-complex forward and than inverse Fourier transform for them. For better performance use cuFFT API.

Required function should have the following prototype:
```cpp
std::vector<float> FffCUFFT(const std::vector<float>& input, int batch);
```
Here $batch$ is a number of independent signals, $input$ contains complex values in the format of $(real, imaginary)$ pairs of floats storing pair by pair. So $input$ array size must be equal to $2 * n * batch$.

The function should perform the following actions:
1. Compute forward Fourier transform for $input$;
2. Compute inverse Fourier transform for the result of step 1;
3. Normalize result of step 2 by $n$.

Returned array must store result of step 3 in the same format of $(real, imaginary)$ pairs as $input$ and have the same size.

Note, that due to Fourier Transform math properties, result array will have the same values as input one. This specificity could be used for self-checking.

Two files are expected to be uploaded:
- fft_cufft.h:
```cpp
#ifndef __FFT_CUFFT_H
#define __FFT_CUFFT_H

#include <vector>

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch);

#endif // __FFT_CUFFT_H
```
- fft_cufft.cu:
```cpp
#include "fft_cufft.h"

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    // Place your implementation here
}
```

## Task #9: OpenCL GELU Implementation
Implement GELU function with the following interface in OpenCL using the formula described in task #1:
```cpp
std::vector<float> GeluOCL(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use OpenCL technology to make your function work on NVIDIA GPU. Try to make it fast.

Use `CL_DEVICE_GPU` flag to choose GPU device. Use zero platform and zero device. Store your OpenCL kernel in a string constant.

Two files are expected to be uploaded:
- gelu_ocl.h
```cpp
#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input);

#endif // __GELU_OCL_H
```
- gelu_ocl.cpp
```cpp
#include "gelu_ocl.h"

std::vector<float> GeluOCL(const std::vector<float>& input) {
    // Place your implementation here
}
```

# Results
## 1_gelu_omp (134217728 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI3|kuznetsov_artyom|0.2679|
|3821B1FI3|polozov_vladislav|0.2766|
|3821B1FI3|kulikov_artem|0.2794|
|3821B1FI2|kostanyan_arsen|0.2820|
|3821B1FI1|shipitsin_alex|0.2916|
|3821B1PE1|yurin_andrey|0.2945|
|3821B1FI3|durandin_vladimir|0.2964|
|3821B1FI2|petrov_maksim|0.3031|
|3821B1FI2|kostin_artem|0.3777|
|3821B1PE1|chuvashov_andrey|0.4449|
|3821B1PE1|kriseev_mikhail|0.4687|
|3821B1FI3|simonyan_suren|0.4708|
|3821B1FI3|kulagin_aleksandr|0.4800|
|3821B1PE1|vinichuk_timofey|0.5684|
|3821B1FI2|zakharov_artem|0.6916|
|3821B1FI3|sadikov_damir|0.6978|
|3821B1FI1|kashirin_alexander|0.6999|
|3821B1FI1|bodrov_daniil|0.7005|
|3821B1FI3|sharapov_georgiy|0.7042|
|3821B1PE3|Kokin_Ivan|0.7088|
|3821B1FI1|veselov_ilya|0.7132|
|3821B1FI3|volodin_evgeniy|0.7137|
|3821B1PE2|belan_vadim|0.7160|
|3821B1FI2|travin_maksim|0.7180|
|3821B1FI3|korablev_nikita|0.7191|
|3821B1FI3|benduyzhko_tatiana|0.7471|
|3821B1FI1|alexseev_danila|0.7535|
|3821B1FI1|akopyan_zal|0.7544|
|3821B1FI3|kulaev_zhenya|0.7553|
|3821B1FI3|prokofev_kirill|0.7670|
|3821B1FI3|tyulkina_olga|0.7711|
|3821B1PE1|smirnov_leonid|0.7726|
|3821B1PE1|khramov_ivan|0.7728|
|3821B1PE1|pozdnyakov_vasya|0.7753|
|3821B1PE3|Musaev_Ilgar|0.7776|
|3821B1PE1|khodyrev_fedor|0.7787|
|3821B1PE1|kiselev_igor|0.7866|
|3821B1FI3|ivanov_nikita|0.7888|
|REF|REF|0.8126|
|3821B1PE3|smirnov_pavel|1.4899|
|3821B1PE1|vanushkin_dmitry|1.5001|
|3821B1FI1|mirzakhmedov_alexander|TEST FAILED|
|3821B1PE1|kashin_stepan|BUILD FAILED|
|3821B1PE2|zhatkin_vyacheslav|TEST FAILED|

## 2_gelu_cuda (134217728 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1PE1|kriseev_mikhail|0.2370|
|3821B1PE2|savchuk_anton|0.2378|
|3821B1FI3|kulikov_artem|0.2385|
|3821B1FI1|bodrov_daniil|0.2399|
|3821B1FI3|prokofev_kirill|0.2402|
|3821B1PE1|vinichuk_timofey|0.2407|
|3821B1FI3|kulaev_zhenya|0.2410|
|3821B1FI3|polozov_vladislav|0.2413|
|3821B1FI1|shipitsin_alex|0.2419|
|3821B1PE1|chuvashov_andrey|0.2420|
|3821B1FI3|kuznetsov_artyom|0.2452|
|3821B1FI2|zakharov_artem|0.2455|
|3821B1FI1|akopyan_zal|0.2461|
|3821B1PE3|smirnov_pavel|0.2461|
|3821B1FI3|sharapov_georgiy|0.2466|
|3821B1FI1|veselov_ilya|0.2470|
|3821B1FI3|tyulkina_olga|0.2471|
|3821B1FI2|kostanyan_arsen|0.2477|
|3821B1FI2|kostin_artem|0.2482|
|3821B1FI2|travin_maksim|0.2498|
|3821B1PE1|smirnov_leonid|0.2500|
|3821B1PE3|Musaev_Ilgar|0.2500|
|3821B1FI3|kulagin_aleksandr|0.2506|
|3821B1FI3|volodin_evgeniy|0.2509|
|3821B1FI1|alexseev_danila|0.2509|
|3821B1FI2|petrov_maksim|0.2540|
|3821B1FI1|kashirin_alexander|0.2571|
|3821B1FI3|sadikov_damir|0.2584|
|3821B1FI3|benduyzhko_tatiana|0.2593|
|REF|REF|0.2598|
|3821B1PE1|kiselev_igor|0.2607|
|3821B1FI3|ivanov_nikita|0.2627|
|3821B1FI3|korablev_nikita|0.2653|
|3821B1PE1|kashin_stepan|0.2667|
|3821B1FI3|simonyan_suren|0.2671|
|3821B1PE1|khodyrev_fedor|0.2716|
|3821B1PE1|yurin_andrey|0.2718|
|3821B1PE1|khramov_ivan|0.2879|
|3821B1PE1|pozdnyakov_vasya|0.2983|
|3821B1PE1|vanushkin_dmitry|0.3671|
|3821B1PE2|zhatkin_vyacheslav|0.4486|
|3821B1PE2|derun_andrei|TEST FAILED|
|3821B1PE2|belan_vadim|BUILD FAILED|

## 3_naive_gemm_omp (1024 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI1|shipitsin_alex|0.0973|
|3821B1PE1|yurin_andrey|0.1144|
|3821B1FI1|kashirin_alexander|0.1256|
|3821B1FI1|akopyan_zal|0.1274|
|3821B1FI3|sharapov_georgiy|0.1553|
|3821B1FI2|travin_maksim|0.1666|
|3821B1FI3|ivanov_nikita|0.1681|
|3821B1FI1|bodrov_daniil|0.1690|
|3821B1FI2|petrov_maksim|0.1707|
|3821B1PE1|vanushkin_dmitry|0.1717|
|3821B1FI3|sadikov_damir|0.1747|
|3821B1FI3|kulaev_zhenya|0.1750|
|3821B1FI3|prokofev_kirill|0.5781|
|3821B1FI3|polozov_vladislav|0.6071|
|3821B1FI2|kostin_artem|0.6207|
|3821B1PE1|pozdnyakov_vasya|0.6863|
|3821B1FI3|korablev_nikita|0.7212|
|3821B1PE1|khodyrev_fedor|0.7400|
|3821B1FI1|alexseev_danila|0.7757|
|3821B1PE3|Musaev_Ilgar|0.7770|
|3821B1PE2|zhatkin_vyacheslav|0.7813|
|3821B1FI1|veselov_ilya|0.7819|
|3821B1FI3|simonyan_suren|0.7843|
|3821B1PE3|smirnov_pavel|0.7883|
|3821B1PE2|savchuk_anton|0.7977|
|3821B1PE1|chuvashov_andrey|0.8009|
|3821B1FI3|benduyzhko_tatiana|0.8010|
|3821B1PE2|belan_vadim|0.8026|
|3821B1FI3|volodin_evgeniy|0.8086|
|3821B1PE1|khramov_ivan|0.8116|
|3821B1FI3|kulagin_aleksandr|0.8122|
|3821B1PE1|kiselev_igor|0.8134|
|3821B1FI3|kulikov_artem|0.8203|
|3821B1PE1|vinichuk_timofey|0.8271|
|3821B1PE1|smirnov_leonid|0.8334|
|3821B1PE1|kriseev_mikhail|0.8339|
|REF|REF|0.8379|
|3821B1FI3|tyulkina_olga|0.8383|
|3821B1FI2|zakharov_artem|0.8446|
|3821B1FI3|kuznetsov_artyom|0.9010|
|3821B1PE1|kashin_stepan|TOO SLOW|

## 4_naive_gemm_cuda (4096 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI1|shipitsin_alex|0.1586|
|3821B1PE1|yurin_andrey|0.1635|
|3821B1FI3|kulaev_zhenya|0.1762|
|3821B1FI1|veselov_ilya|0.1763|
|3821B1FI3|ivanov_nikita|0.1789|
|3821B1FI1|bodrov_daniil|0.1792|
|3821B1PE1|kriseev_mikhail|0.1813|
|3821B1FI3|sharapov_georgiy|0.1824|
|3821B1PE1|chuvashov_andrey|0.1844|
|3821B1PE1|vanushkin_dmitry|0.1865|
|3821B1PE1|kiselev_igor|0.1866|
|REF|REF|0.1877|
|3821B1FI2|kostin_artem|0.1878|
|3821B1FI2|petrov_maksim|0.1901|
|3821B1FI3|polozov_vladislav|0.2280|
|3821B1FI3|simonyan_suren|0.2324|
|3821B1FI3|kuznetsov_artyom|0.2329|
|3821B1PE1|pozdnyakov_vasya|0.2331|
|3821B1FI3|kulikov_artem|0.2340|
|3821B1FI3|benduyzhko_tatiana|0.2353|
|3821B1FI1|akopyan_zal|0.2355|
|3821B1PE1|smirnov_leonid|0.2357|
|3821B1FI3|tyulkina_olga|0.2358|
|3821B1FI3|prokofev_kirill|0.2369|
|3821B1FI1|alexseev_danila|0.2506|
|3821B1FI2|zakharov_artem|0.2689|
|3821B1FI3|volodin_evgeniy|0.2805|
|3821B1PE3|Musaev_Ilgar|0.2810|
|3821B1FI3|korablev_nikita|0.2824|
|3821B1PE3|smirnov_pavel|0.3040|
|3821B1FI1|kashirin_alexander|0.3984|
|3821B1PE1|khramov_ivan|0.4043|
|3821B1FI3|sadikov_damir|0.4133|
|3821B1PE1|khodyrev_fedor|0.4135|
|3821B1FI2|travin_maksim|0.5205|
|3821B1FI3|kulagin_aleksandr|0.5823|
|3821B1PE2|derun_andrei|1.0889|
|3821B1PE1|kashin_stepan|BUILD FAILED|
|3821B1PE2|savchuk_anton|BUILD FAILED|
|3821B1PE2|belan_vadim|BUILD FAILED|

## 5_block_gemm_omp (1024 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI1|bodrov_daniil|0.0512|
|3821B1FI3|benduyzhko_tatiana|0.1023|
|3821B1FI1|alexseev_danila|0.1026|
|3821B1FI3|korablev_nikita|0.1520|
|3821B1FI3|sadikov_damir|0.1878|
|REF|REF|0.1980|
|3821B1FI3|sharapov_georgiy|0.2053|
|3821B1PE1|chuvashov_andrey|0.2088|
|3821B1FI3|polozov_vladislav|0.2145|
|3821B1PE1|khramov_ivan|0.2183|
|3821B1FI1|kashirin_alexander|0.2185|
|3821B1PE1|smirnov_leonid|0.2192|
|3821B1PE1|pozdnyakov_vasya|0.2221|
|3821B1FI3|kuznetsov_artyom|0.2230|
|3821B1PE3|Musaev_Ilgar|0.2258|
|3821B1PE1|khodyrev_fedor|0.2271|
|3821B1FI3|ivanov_nikita|0.2273|
|3821B1PE1|yurin_andrey|0.2314|
|3821B1FI2|travin_maksim|0.2338|
|3821B1FI1|akopyan_zal|0.2378|
|3821B1FI3|kulikov_artem|0.2386|
|3821B1PE3|smirnov_pavel|0.2406|
|3821B1FI3|prokofev_kirill|0.2551|
|3821B1PE1|kiselev_igor|0.2596|
|3821B1FI1|veselov_ilya|0.2624|
|3821B1PE1|kriseev_mikhail|0.2634|
|3821B1PE1|vanushkin_dmitry|0.2708|
|3821B1FI2|kostin_artem|0.2776|
|3821B1FI2|petrov_maksim|0.3179|
|3821B1FI2|zakharov_artem|0.3504|
|3821B1PE1|kashin_stepan|0.3548|
|3821B1FI1|shipitsin_alex|0.3734|
|3821B1PE2|savchuk_anton|0.3778|
|3821B1PE2|belan_vadim|0.3897|
|3821B1FI3|tyulkina_olga|0.5011|
|3821B1FI3|kulaev_zhenya|0.5147|
|3821B1FI3|simonyan_suren|0.5419|

## 6_block_gemm_cuda (4096 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI1|alexseev_danila|0.1397|
|3821B1FI2|travin_maksim|0.1420|
|3821B1FI3|benduyzhko_tatiana|0.1439|
|3821B1FI3|kulaev_zhenya|0.1442|
|3821B1PE1|chuvashov_andrey|0.1444|
|3821B1FI1|bodrov_daniil|0.1447|
|3821B1FI1|akopyan_zal|0.1502|
|3821B1FI3|sadikov_damir|0.1502|
|3821B1PE1|pozdnyakov_vasya|0.1503|
|3821B1FI3|kuznetsov_artyom|0.1516|
|3821B1FI3|simonyan_suren|0.1521|
|3821B1FI3|polozov_vladislav|0.1522|
|REF|REF|0.1524|
|3821B1PE1|smirnov_leonid|0.1529|
|3821B1FI3|tyulkina_olga|0.1541|
|3821B1FI3|sharapov_georgiy|0.1541|
|3821B1FI3|kulikov_artem|0.1550|
|3821B1FI2|petrov_maksim|0.1577|
|3821B1PE1|khramov_ivan|0.1595|
|3821B1FI3|ivanov_nikita|0.1609|
|3821B1PE1|khodyrev_fedor|0.1609|
|3821B1PE1|vanushkin_dmitry|0.1644|
|3821B1PE1|yurin_andrey|0.2017|
|3821B1FI1|shipitsin_alex|0.2031|
|3821B1FI2|kostin_artem|0.2782|
|3821B1PE3|Musaev_Ilgar|0.3034|
|3821B1FI1|kashirin_alexander|0.3156|
|3821B1PE3|smirnov_pavel|0.3197|
|3821B1FI1|veselov_ilya|0.3456|
|3821B1FI3|korablev_nikita|0.3486|
|3821B1PE1|kriseev_mikhail|0.7497|
|3821B1PE2|derun_andrei|TEST FAILED|
|3821B1PE2|belan_vadim|BUILD FAILED|

## 7_gemm_cublas (4096 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI1|shipitsin_alex|0.0408|
|3821B1FI3|ivanov_nikita|0.0454|
|3821B1PE1|chuvashov_andrey|0.0473|
|3821B1PE1|vanushkin_dmitry|0.0475|
|3821B1FI3|kulikov_artem|0.0478|
|3821B1FI1|alexseev_danila|0.0496|
|3821B1FI3|benduyzhko_tatiana|0.0497|
|3821B1PE1|yurin_andrey|0.0498|
|3821B1PE3|Musaev_Ilgar|0.0498|
|3821B1FI1|akopyan_zal|0.0547|
|3821B1FI1|kashirin_alexander|0.0557|
|3821B1FI3|kuznetsov_artyom|0.0559|
|3821B1FI1|veselov_ilya|0.0566|
|3821B1PE1|khodyrev_fedor|0.0566|
|3821B1FI1|bodrov_daniil|0.0579|
|3821B1FI3|simonyan_suren|0.0579|
|3821B1FI3|tyulkina_olga|0.0582|
|3821B1PE1|pozdnyakov_vasya|0.0583|
|3821B1FI3|sadikov_damir|0.0585|
|3821B1FI3|korablev_nikita|0.0590|
|3821B1FI2|travin_maksim|0.0592|
|3821B1FI3|polozov_vladislav|0.0596|
|3821B1PE1|kriseev_mikhail|0.0599|
|3821B1FI3|kulaev_zhenya|0.0600|
|3821B1PE1|smirnov_leonid|0.0601|
|3821B1PE3|smirnov_pavel|0.0601|
|REF|REF|0.0601|
|3821B1PE1|khramov_ivan|0.0608|
|3821B1FI3|sharapov_georgiy|0.0800|
|3821B1PE2|derun_andrei|TEST FAILED|
|3821B1PE2|belan_vadim|BUILD FAILED|

## 8_fft_cufft (131072 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1PE1|vanushkin_dmitry|0.1077|
|3821B1FI1|bodrov_daniil|0.1173|
|3821B1FI3|ivanov_nikita|0.1220|
|3821B1FI3|sadikov_damir|0.1257|
|3821B1PE1|kriseev_mikhail|0.1326|
|3821B1FI3|simonyan_suren|0.1332|
|3821B1FI3|kulaev_zhenya|0.1375|
|3821B1FI3|kuznetsov_artyom|0.1375|
|3821B1FI3|kulikov_artem|0.1375|
|3821B1FI3|polozov_vladislav|0.1379|
|3821B1FI3|benduyzhko_tatiana|0.1388|
|3821B1FI1|akopyan_zal|0.1396|
|3821B1PE1|chuvashov_andrey|0.1400|
|3821B1PE3|Musaev_Ilgar|0.1402|
|3821B1PE1|smirnov_leonid|0.1420|
|3821B1PE1|yurin_andrey|0.1426|
|3821B1PE1|khodyrev_fedor|0.1440|
|3821B1FI1|shipitsin_alex|0.1447|
|3821B1PE1|khramov_ivan|0.1460|
|3821B1PE1|pozdnyakov_vasya|0.1470|
|3821B1FI2|travin_maksim|0.1476|
|3821B1FI3|sharapov_georgiy|0.1574|
|3821B1FI3|tyulkina_olga|0.1595|
|3821B1FI3|korablev_nikita|0.1595|
|3821B1FI1|alexseev_danila|0.1727|
|3821B1FI1|kashirin_alexander|0.1991|
|3821B1FI1|veselov_ilya|0.1996|
|3821B1PE3|smirnov_pavel|0.2139|
|REF|REF|0.2309|
|3821B1PE2|derun_andrei|RUN FAILED|
|3821B1PE2|belan_vadim|BUILD FAILED|

## 9_gelu_ocl (134217728 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI3|kulaev_zhenya|0.2314|
|REF|REF|0.2621|
|3821B1FI3|kuznetsov_artyom|0.2646|
|3821B1FI3|polozov_vladislav|0.2720|
|3821B1FI3|kulikov_artem|0.2766|
|3821B1FI3|sadikov_damir|0.2768|
|3821B1FI3|simonyan_suren|0.2802|
|3821B1PE3|Musaev_Ilgar|0.2804|
|3821B1FI1|akopyan_zal|0.2809|
|3821B1PE1|khramov_ivan|0.2844|
|3821B1FI3|benduyzhko_tatiana|0.2856|
|3821B1FI3|sharapov_georgiy|0.2870|
|3821B1PE1|yurin_andrey|0.2876|
|3821B1FI3|ivanov_nikita|0.2909|
|3821B1FI1|shipitsin_alex|0.2946|
|3821B1PE3|smirnov_pavel|0.2974|
|3821B1PE1|smirnov_leonid|0.2976|
|3821B1PE1|khodyrev_fedor|0.2990|
|3821B1PE1|pozdnyakov_vasya|0.3008|
|3821B1FI1|kashirin_alexander|0.3011|
|3821B1FI3|tyulkina_olga|0.3032|
|3821B1FI1|alexseev_danila|0.3053|
|3821B1FI3|korableb_nikita|0.3918|
|3821B1PE1|kriseev_mikhail|0.4140|
|3821B1FI3|korablev_nikita|0.4283|
|3821B1FI2|travin_maksim|0.4418|
|3821B1FI1|bodrov_daniil|TOO SLOW|
|3821B1PE1|chuvashov_andrey|BUILD FAILED|
|3821B1PE2|belan_vadim|BUILD FAILED|

# Tasks Done
## 3821B1FI1
|Group|Name|Passed|
|-----|----|------|
|3821B1FI1|akopyan_zal|**9/9**|
|3821B1FI1|alexseev_danila|**9/9**|
|3821B1FI1|bodrov_daniil|8/9|
|3821B1FI1|kashirin_alexander|**9/9**|
|3821B1FI1|mirzakhmedov_alexander|0/9|
|3821B1FI1|shipitsin_alex|**9/9**|
|3821B1FI1|veselov_ilya|8/9|

## 3821B1FI2
|Group|Name|Passed|
|-----|----|------|
|3821B1FI2|kostanyan_arsen|2/9|
|3821B1FI2|kostin_artem|6/9|
|3821B1FI2|petrov_maksim|6/9|
|3821B1FI2|travin_maksim|**9/9**|
|3821B1FI2|zakharov_artem|5/9|

## 3821B1FI3
|Group|Name|Passed|
|-----|----|------|
|3821B1FI3|benduyzhko_tatiana|**9/9**|
|3821B1FI3|durandin_vladimir|1/9|
|3821B1FI3|ivanov_nikita|**9/9**|
|3821B1FI3|korableb_nikita|1/9|
|3821B1FI3|korablev_nikita|**9/9**|
|3821B1FI3|kulaev_zhenya|**9/9**|
|3821B1FI3|kulagin_aleksandr|4/9|
|3821B1FI3|kulikov_artem|**9/9**|
|3821B1FI3|kuznetsov_artyom|**9/9**|
|3821B1FI3|polozov_vladislav|**9/9**|
|3821B1FI3|prokofev_kirill|5/9|
|3821B1FI3|sadikov_damir|**9/9**|
|3821B1FI3|sharapov_georgiy|**9/9**|
|3821B1FI3|simonyan_suren|**9/9**|
|3821B1FI3|tyulkina_olga|**9/9**|
|3821B1FI3|volodin_evgeniy|4/9|

## 3821B1PE1
|Group|Name|Passed|
|-----|----|------|
|3821B1PE1|chuvashov_andrey|8/9|
|3821B1PE1|kashin_stepan|2/9|
|3821B1PE1|khodyrev_fedor|**9/9**|
|3821B1PE1|khramov_ivan|**9/9**|
|3821B1PE1|kiselev_igor|5/9|
|3821B1PE1|kriseev_mikhail|**9/9**|
|3821B1PE1|pozdnyakov_vasya|**9/9**|
|3821B1PE1|smirnov_leonid|**9/9**|
|3821B1PE1|vanushkin_dmitry|8/9|
|3821B1PE1|vinichuk_timofey|3/9|
|3821B1PE1|yurin_andrey|**9/9**|

## 3821B1PE2
|Group|Name|Passed|
|-----|----|------|
|3821B1PE2|belan_vadim|3/9|
|3821B1PE2|derun_andrei|1/9|
|3821B1PE2|savchuk_anton|3/9|
|3821B1PE2|zhatkin_vyacheslav|2/9|

## 3821B1PE3
|Group|Name|Passed|
|-----|----|------|
|3821B1PE3|Kokin_Ivan|1/9|
|3821B1PE3|Musaev_Ilgar|**9/9**|
|3821B1PE3|smirnov_pavel|**9/9**|

