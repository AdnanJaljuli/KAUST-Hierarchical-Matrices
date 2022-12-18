
#ifndef HELPER_KERNELS_H
#define HELPER_KERNELS_H

#include "kDTree.cuh"
#include "TLRMatrix.cuh"

#include <assert.h>
#include <ctype.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include <curand_kernel.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// void generateMaxRanks(unsigned int numLevels, unsigned int leafSize, unsigned int *maxRanks) {
//     for(unsigned int i = 0; i < numLevels - 2; ++i) {
//         maxRanks[i] = leafSize*(1 << i);
//         if(i > 5) {
//             maxRanks[i]/=4;
//         }
//     }
// }

template<class T>
struct UnaryAoAAssign : public thrust::unary_function<int, T*>
{
    T* original_array;
    int stride;
    UnaryAoAAssign(T* original_array, int stride) { this->original_array = original_array; this->stride = stride; }
    __host__ __device__
    T* operator()(const unsigned int& thread_id) const { return original_array + thread_id * stride; }
};

template<class T>
static void generateArrayOfPointersT(T* original_array, T** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{
    
    thrust::device_ptr<T*> dev_data(array_of_arrays);
    thrust::transform(
        thrust::cuda::par.on(stream),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_arrays),
        dev_data,
        UnaryAoAAssign<T>(original_array, stride)
    );
    cudaGetLastError();
}

static __host__ __device__ int ij2Morton(unsigned int x, unsigned int y){
    static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
    static const unsigned int S[] = {1, 2, 4, 8};

    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];

    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    int z = x | (y << 1);
    return z;
}

static __host__ __device__ int columnMajor2Morton(int numSegments, int n){
    unsigned int i = n%numSegments;
    unsigned int j = n/numSegments;
    return ij2Morton(j, i);
}

static __host__ __device__ uint32_t morton2columnMajor_helper(uint32_t x) {
    x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

static __host__ __device__ void morton2columnMajor(uint32_t d, uint32_t &x, uint32_t &y) {
    x = morton2columnMajor_helper(d);
    y = morton2columnMajor_helper(d >> 1);
}

static __device__ __host__ int upperPowerOfTwo(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

#endif
