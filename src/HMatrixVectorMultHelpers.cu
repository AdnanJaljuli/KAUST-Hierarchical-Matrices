
#include "HMatrixVectorMultHelpers.cuh"
#include <vector>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/gemm/device/gemm_batched.h>

cudaError_t cutlass_strided_batched_sgemm(
  int m, 
  int n,
  int k,
  double alpha,
  double const *A,
  int lda,
  long long int batch_stride_A,
  double const *B,
  int ldb,
  long long int batch_stride_B,
  double *C,
  int ldc,
  long long int batch_stride_C,
  double beta,
  int batch_count) {

  using Gemm = cutlass::gemm::device::GemmBatched<
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor,
    double, cutlass::layout::ColumnMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    {A, lda}, 
    batch_stride_A,
    {B, ldb}, 
    batch_stride_B,
    {C, ldc}, 
    batch_stride_C,
    {C, ldc}, 
    batch_stride_C,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

cudaError_t cutlassDiagonalXVec(
    unsigned int numberOfInputPoints, unsigned int  bucketSize, 
    unsigned int  numSegments, unsigned int  vectorWidth, H2Opus_Real *diagonal,
    H2Opus_Real *inputVectors, H2Opus_Real *resultVectors) {
        // Arbitrary problem size
        int const m = bucketSize;
        int const n = vectorWidth;
        int const k = bucketSize;
        int const batch_count = numSegments;

        // A, B are non-transpose, column major
        int const lda = m;
        int const ldb = k * batch_count;
        int const ldc = m;

        // the memory is batched along K dimension
        long long int batch_stride_A = static_cast<long long int>(lda) * static_cast<long long int>(k);
        long long int batch_stride_B = static_cast<long long int>(k);
        long long int batch_stride_C = static_cast<long long int>(ldc) * static_cast<long long int>(n);

        // alpha and beta
        double alpha = 1.0f;
        double beta = 0.0f;

        cudaError_t result = cudaSuccess;

        result = cutlass_strided_batched_sgemm(
            m, n, k, alpha, diagonal, lda, batch_stride_A, inputVectors, ldb, batch_stride_B, resultVectors, ldc, batch_stride_C,
            beta, batch_count);
        if (result != cudaSuccess)
            return result;
}

// __global__ void diagonalXVec(
//     unsigned int numberOfInputPoints, unsigned int  bucketSize, 
//     unsigned int  numSegments, unsigned int  vectorWidth, H2Opus_Real *diagonal,
//     H2Opus_Real *inpuVectors, H2Opus_Real *resultVectors) {
        
//         extern __shared__ H2Opus_Real shared_mem[];
//         H2Opus_Real *firstTile_s = shared_mem;
//         H2Opus_Real *secondTile_s = &shared_mem[blockDim.x*blockDim.y];

//         unsigned int blockRow = blockIdx.y*blockDim.y + threadIdx.y;
//         unsigned int blockCol = blockIdx.x*blockDim.x + threadIdx.x;
//         H2Opus_Real sum = 0.0lf;

//         // loop over tiles
//         unsigned int numTiles = bucketSize/blockDim.y;
//         for(unsigned int tile = 0; tile < numTiles; ++tile) {
//             // load blocks to shared memory
//             secondTile_s[threadIdx.x*blockDim.y + threadIdx.y] = inpuVectors[blockCol*numberOfInputPoints + blockIdx.y*blockDim.y + tile*];

//             // firstTile_s[threadIdx.x*blockDim.y + threadIdx.y] = diagonal[blockIdx.z*bucketSize*bucketSize + (tile*blockDim.x + threadIdx.x)*bucketSize + row];
//             __syncthreads();

//             // multiply and store in register

//         }
//         // store back to results array
// }