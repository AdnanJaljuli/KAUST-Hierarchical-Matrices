
#include "cutlassDiagonalXVector.cuh"
#include <vector>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/gemm/device/gemm_batched.h>

cudaError_t cutlass_strided_batched_dgemm(
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
			batch_count});

		if (status != cutlass::Status::kSuccess) {
			return cudaErrorUnknown;
		}

  		return cudaSuccess;
}

cudaError_t cutlassDiagonalXVec(
    unsigned int numberOfInputPoints, unsigned int bucketSize, 
    unsigned int numSegments, unsigned int vectorWidth, H2Opus_Real *diagonal,
    H2Opus_Real *inputVectors, H2Opus_Real *resultVectors) {

		int const lda = bucketSize;
		int const ldb = numberOfInputPoints;
		int const ldc = numberOfInputPoints;

		long long int batch_stride_A = static_cast<long long int>(bucketSize)*static_cast<long long int>(bucketSize);
		long long int batch_stride_B = static_cast<long long int>(bucketSize);
		long long int batch_stride_C = static_cast<long long int>(bucketSize);

		double alpha = 1;
		double beta = 0;

		cudaError_t result = cudaSuccess;
		result = cutlass_strided_batched_dgemm(
			bucketSize, vectorWidth, bucketSize, alpha, diagonal, lda, batch_stride_A, inputVectors, ldb, batch_stride_B, resultVectors, ldc, batch_stride_C,
			beta, numSegments);

		return result;
}
