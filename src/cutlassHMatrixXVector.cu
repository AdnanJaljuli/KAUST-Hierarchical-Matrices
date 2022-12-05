
#include "cutlassHMatrixXVector.cuh"
#include "HMatrix.cuh"

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/command_line.h>
#include <cutlass/util/distribution.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/tensor_view_io.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm_complex.h>
#include <cutlass/util/reference/device/gemm_complex.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_norm.h>

struct Result {
	double runtime_ms;
	double initialization_time_ms;
	double gflops;
	cutlass::Status status;
	cudaError_t error;
	bool passed;

	Result(
		double runtime_ms = 0,
		double initialization_time_ms = 0,
		double gflops = 0,
		cutlass::Status status = cutlass::Status::kSuccess,
		cudaError_t error = cudaSuccess
	):
	runtime_ms(runtime_ms), initialization_time_ms(initialization_time_ms), gflops(gflops),
	status(status), error(error), passed(true) { }
};

void preprocessGroupedGEMM(
	unsigned int numberOfInputPoints, unsigned int level, int numLevels,
	int problemCount, unsigned int bucketSize, unsigned int vectorWidth,
	HMatrixLevel matrixLevel, 
	std::vector<cutlass::gemm::GemmCoord> *h_problemSizes, cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> *d_problemSizes,
	std::vector<int64_t> *h_lda, std::vector<int64_t> *h_ldb, std::vector<int64_t> *h_ldc,
	cutlass::DeviceAllocation<int64_t> *lda, cutlass::DeviceAllocation<int64_t> *ldb, cutlass::DeviceAllocation<int64_t> *ldc,
	H2Opus_Real *AMatrix, H2Opus_Real *BMatrix, H2Opus_Real *CMatrix,
	std::vector<H2Opus_Real*> *h_ptrA, std::vector<H2Opus_Real*> *h_ptrB, std::vector<H2Opus_Real*> *h_ptrC,
	cutlass::DeviceAllocation<H2Opus_Real *> *ptr_A, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_B, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_C,
	unsigned int iteration) {

		int *h_tileScanRanks = (int*)malloc(problemCount*sizeof(int));
		cudaMemcpy(h_tileScanRanks, matrixLevel.tileScanRanks, problemCount*sizeof(int), cudaMemcpyDeviceToHost);

		int previousTileScanRank = 0;
		for (unsigned int tile = 0; tile < problemCount; ++tile) {
			int tileRank = h_tileScanRanks[tile] - previousTileScanRank;
			unsigned int tileDimension = (1<<(numLevels - (level + 1)))*bucketSize;

			printf("tile rank: %d   tile dimension: %d\n", tileRank, tileDimension);
			if(iteration == 0) {
				cutlass::gemm::GemmCoord problem(tileRank, vectorWidth, tileDimension);
				// cutlass::gemm::GemmCoord problem(tileDimension, vectorWidth, tileRank);
				h_problemSizes->push_back(problem);
			}
			else {
				cutlass::gemm::GemmCoord problem(tileDimension, vectorWidth, tileRank);
				h_problemSizes->push_back(problem);
			}

			if(iteration == 0) {
				h_lda->at(tile) = (int64_t)tileRank;
				// h_lda->at(tile) = (int64_t)tileDimension;
				h_ldb->at(tile) = (int64_t)numberOfInputPoints;
				h_ldc->at(tile) = (int64_t)tileRank;
			}
			else {
				h_lda->at(tile) = (int64_t)tileDimension;
				h_ldb->at(tile) = (int64_t)tileRank;
				h_ldc->at(tile) = (int64_t)numberOfInputPoints;
			}

			h_ptrA->at(tile) = AMatrix + previousTileScanRank*tileDimension;
			if(iteration == 0) {
				int diff = (tile%2 == 0) ? 1 : -1;
				h_ptrB->at(tile) = BMatrix + (tile + diff)*tileDimension;
				h_ptrC->at(tile) = CMatrix + tile*vectorWidth*tileDimension;
			}
			else {
				h_ptrB->at(tile) = BMatrix + tile*vectorWidth*tileDimension;
				h_ptrC->at(tile) = CMatrix + tile*tileDimension;
			}

			previousTileScanRank += tileRank;
		}

		d_problemSizes->reset(problemCount);
 		d_problemSizes->copy_from_host(h_problemSizes->data());

		lda->copy_from_host(h_lda->data());
		ldb->copy_from_host(h_ldb->data());
		ldc->copy_from_host(h_ldc->data());

		ptr_A->copy_from_host(h_ptrA->data());
		ptr_B->copy_from_host(h_ptrB->data());
		ptr_C->copy_from_host(h_ptrC->data());

		free(h_tileScanRanks);
}

__global__ void transposeMatrix(int tileDimension, HMatrixLevel matrixLevel, H2Opus_Real *d_VTransposed) {
	unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

	int scanRankVal;
	if(blockIdx.z == 0) {
		scanRankVal = 0;
	}
	else {
		scanRankVal = matrixLevel.tileScanRanks[blockIdx.z - 1];
	}
	int rank = matrixLevel.tileScanRanks[blockIdx.z] - scanRankVal;
	if(col < rank) {
		d_VTransposed[scanRankVal*tileDimension + row*rank + col] = matrixLevel.V[scanRankVal*tileDimension + col*tileDimension + row];
	}
}

void VxVector(unsigned int numberOfInputPoints, unsigned int level, int numLevels,
	int problemCount, unsigned int bucketSize, unsigned int vectorWidth,
	HMatrixLevel matrixLevel,
	std::vector<cutlass::gemm::GemmCoord> *h_problemSizes, cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> *d_problemSizes,
	std::vector<int64_t> *h_lda, std::vector<int64_t> *h_ldb, std::vector<int64_t> *h_ldc,
	cutlass::DeviceAllocation<int64_t> *lda, cutlass::DeviceAllocation<int64_t> *ldb, cutlass::DeviceAllocation<int64_t> *ldc,
	H2Opus_Real *inputVectors, H2Opus_Real *outputVectors,
	std::vector<H2Opus_Real*> *h_ptrA, std::vector<H2Opus_Real*> *h_ptrB, std::vector<H2Opus_Real*> *h_ptrC,
	cutlass::DeviceAllocation<H2Opus_Real *> *ptr_A, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_B, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_C) {

		using ElementInput = H2Opus_Real;
		using ElementOutput = H2Opus_Real;
		using ElementAccumulator = H2Opus_Real;

		using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
			ElementInput, 
			cutlass::layout::ColumnMajor,
			cutlass::ComplexTransform::kNone,
			1,
			ElementInput,
			cutlass::layout::ColumnMajor,
			cutlass::ComplexTransform::kNone,
			1,
			ElementOutput, cutlass::layout::ColumnMajor,
			ElementAccumulator, 
			cutlass::arch::OpClassTensorOp, 
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<64, 64, 16>,
			cutlass::gemm::GemmShape<32, 32, 16>,
			cutlass::gemm::GemmShape<8, 8, 4>,
			cutlass::epilogue::thread::LinearCombination<
				ElementOutput, 1,
				ElementAccumulator, ElementAccumulator>,
			cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
			4>::GemmKernel;

		using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

		// transpose Vs
		int tileDimension = (1<<(numLevels - (level + 1)))*bucketSize;
		printf("bucket size: %d\n", bucketSize);
		printf("numleveles: %d\n", numLevels);
		int matrixSize;
		cudaMemcpy(&matrixSize, &matrixLevel.tileScanRanks[problemCount - 1], sizeof(int), cudaMemcpyDeviceToHost);
		printf("tileDimension: %d    matrix size: %d   level: %d\n", tileDimension, matrixSize, level);
		H2Opus_Real *d_VTransposed;
		cudaMalloc((void**) &d_VTransposed, matrixSize*tileDimension*sizeof(H2Opus_Real));

		dim3 numThreadsPerBlock(32, 32);
		dim3 numBlocks(((tileDimension/2) + 31)/32, tileDimension/32, problemCount);
		transposeMatrix <<< numBlocks, numThreadsPerBlock >>> (tileDimension, matrixLevel, d_VTransposed);
		cudaDeviceSynchronize();
		preprocessGroupedGEMM(numberOfInputPoints, level, numLevels,
			problemCount, bucketSize, vectorWidth,
			matrixLevel,
			h_problemSizes, d_problemSizes,
			h_lda, h_ldb, h_ldc,
			lda, ldb, ldc,
			d_VTransposed, inputVectors, outputVectors,
			h_ptrA, h_ptrB, h_ptrC,
			ptr_A, ptr_B, ptr_C,
			0);

		int threadblockCount = Gemm::sufficient(h_problemSizes->data(), problemCount);
		if (!threadblockCount) {
			printf("Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");
		}

		double alpha = 1, beta = 0;
		typename Gemm::EpilogueOutputOp::Params epilogue(alpha, beta);
		typename Gemm::Arguments args(
			d_problemSizes->get(),
			problemCount,
			threadblockCount,
			epilogue,
			ptr_A->get(),
			ptr_B->get(),
			ptr_C->get(),
			ptr_C->get(),
			lda->get(),
			ldb->get(),
			ldc->get(),
			ldc->get(),
			h_problemSizes->data() // ptr to where data in vector starts
		);

		Gemm gemm;
		size_t workspace_size = gemm.get_workspace_size(args);
		cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
		gemm.initialize(args, workspace.get());
		gemm.run();

		cudaFree(d_VTransposed);
}

void UxResult(unsigned int numberOfInputPoints, unsigned int level, int numLevels,
	int problemCount, unsigned int bucketSize, unsigned int vectorWidth,
	HMatrixLevel matrixLevel,
	std::vector<cutlass::gemm::GemmCoord> *h_problemSizes, cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> *d_problemSizes,
	std::vector<int64_t> *h_lda, std::vector<int64_t> *h_ldb, std::vector<int64_t> *h_ldc,
	cutlass::DeviceAllocation<int64_t> *lda, cutlass::DeviceAllocation<int64_t> *ldb, cutlass::DeviceAllocation<int64_t> *ldc,
	H2Opus_Real *inputVectors, H2Opus_Real *outputVectors,
	std::vector<H2Opus_Real*> *h_ptrA, std::vector<H2Opus_Real*> *h_ptrB, std::vector<H2Opus_Real*> *h_ptrC,
	cutlass::DeviceAllocation<H2Opus_Real *> *ptr_A, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_B, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_C) {

		using ElementInput = H2Opus_Real;
		using ElementOutput = H2Opus_Real;
		using ElementAccumulator = H2Opus_Real;

		using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
			ElementInput, 
			cutlass::layout::ColumnMajor, 
			cutlass::ComplexTransform::kNone,
			1,
			ElementInput,
			cutlass::layout::ColumnMajor, 
			cutlass::ComplexTransform::kNone,
			1,
			ElementOutput, cutlass::layout::ColumnMajor,
			ElementAccumulator, 
			cutlass::arch::OpClassTensorOp, 
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<64, 64, 16>,
			cutlass::gemm::GemmShape<32, 32, 16>,
			cutlass::gemm::GemmShape<8, 8, 4>,
			cutlass::epilogue::thread::LinearCombination<
				ElementOutput, 1,
				ElementAccumulator, ElementAccumulator>,
			cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
			4>::GemmKernel;

		using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
		preprocessGroupedGEMM(numberOfInputPoints, level, numLevels,
			problemCount, bucketSize, vectorWidth,
			matrixLevel,
			h_problemSizes, d_problemSizes,
			h_lda, h_ldb, h_ldc,
			lda, ldb, ldc,
			matrixLevel.U, inputVectors, outputVectors,
			h_ptrA, h_ptrB, h_ptrC,
			ptr_A, ptr_B, ptr_C,
			1);
		int threadblockCount = Gemm::sufficient(h_problemSizes->data(), problemCount);
		if (!threadblockCount) {
			printf("Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");
		}

		typename Gemm::EpilogueOutputOp::Params epilogue(1.0f, 1.0f);
		typename Gemm::Arguments args(
			d_problemSizes->get(),
			problemCount,
			threadblockCount,
			epilogue,
			ptr_A->get(),
			ptr_B->get(),
			ptr_C->get(),
			ptr_C->get(),
			lda->get(),
			ldb->get(),
			ldc->get(),
			ldc->get(),
			h_problemSizes->data() // ptr to where data in vector starts
		);

		Gemm gemm;
		size_t workspace_size = gemm.get_workspace_size(args);
		cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
		gemm.initialize(args, workspace.get());
		gemm.run();
}

// __global__ void printOutputMatrix(unsigned int numberOfInputPoints, unsigned int  vectorWidth, H2Opus_Real *resultVectors) {
// 	for(unsigned int i = 0; i < vectorWidth; ++i) {
// 		for(unsigned int j = 0; j < numberOfInputPoints; ++j) {
// 			printf("%lf ", resultVectors[i*numberOfInputPoints + j]);
// 		}
// 		printf("\n");
// 	}
// 	printf("\n");
// }

cudaError_t cutlassHierarchicalXVec(
    unsigned int numberOfInputPoints, unsigned int  bucketSize, 
    unsigned int  numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix,
    H2Opus_Real *inputVectors, H2Opus_Real *bufferVectors, H2Opus_Real *resultVectors) {

		using ElementInput = H2Opus_Real;
		using ElementOutput = H2Opus_Real;
		using ElementAccumulator = H2Opus_Real;

		std::vector<cutlass::gemm::GemmCoord> h_problemSizes;
		cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> d_problemSizes;

		std::vector<int64_t> h_lda;
		std::vector<int64_t> h_ldb;
		std::vector<int64_t> h_ldc;
		cutlass::DeviceAllocation<int64_t> d_lda;
		cutlass::DeviceAllocation<int64_t> d_ldb;
		cutlass::DeviceAllocation<int64_t> d_ldc;
		// d_lda.reset(numSegments);
		// d_ldb.reset(numSegments);
		// d_ldc.reset(numSegments);

		std::vector<ElementInput*> h_ptrA;
		std::vector<ElementInput*> h_ptrB;
		std::vector<ElementAccumulator*> h_ptrC;
		cutlass::DeviceAllocation<ElementInput *> ptr_A;
		cutlass::DeviceAllocation<ElementInput *> ptr_B;
		cutlass::DeviceAllocation<ElementAccumulator *> ptr_C;
		// ptr_A.reset(numSegments);
		// ptr_B.reset(numSegments);
		// ptr_C.reset(numSegments);

      	// loop over levels
		for(unsigned int level = hierarchicalMatrix.numLevels - 1; level > 0; --level) {
			// preprocess each level
			int problemCount = hierarchicalMatrix.levels[level - 1].numTiles;

			h_lda.resize(problemCount);
			h_ldb.resize(problemCount);
			h_ldc.resize(problemCount);

			h_ptrA.resize(problemCount);
			h_ptrB.resize(problemCount);
			h_ptrC.resize(problemCount);

			d_lda.reset(problemCount);
			d_ldb.reset(problemCount);
			d_ldc.reset(problemCount);

			ptr_A.reset(problemCount);
			ptr_B.reset(problemCount);
			ptr_C.reset(problemCount);

			printf("problem count: %d   level: %d\n", problemCount, level);
			VxVector(numberOfInputPoints, level, hierarchicalMatrix.numLevels,
				problemCount, bucketSize, vectorWidth,
				hierarchicalMatrix.levels[level - 1],
				&h_problemSizes, &d_problemSizes,
				&h_lda, &h_ldb, &h_ldc,
				&d_lda, &d_ldb, &d_ldc,
				inputVectors, bufferVectors,
				&h_ptrA, &h_ptrB, &h_ptrC,
				&ptr_A, &ptr_B, &ptr_C);
			cudaDeviceSynchronize();
			h_problemSizes.clear();
			d_lda.reset(problemCount);
			d_ldb.reset(problemCount);
			d_ldc.reset(problemCount);

			ptr_A.reset(problemCount);
			ptr_B.reset(problemCount);
			ptr_C.reset(problemCount);

			UxResult(numberOfInputPoints, level, hierarchicalMatrix.numLevels,
				problemCount, bucketSize, vectorWidth,
				hierarchicalMatrix.levels[level - 1],
				&h_problemSizes, &d_problemSizes,
				&h_lda, &h_ldb, &h_ldc,
				&d_lda, &d_ldb, &d_ldc,
				bufferVectors, resultVectors,
				&h_ptrA, &h_ptrB, &h_ptrC,
				&ptr_A, &ptr_B, &ptr_C);
			cudaDeviceSynchronize();
			h_problemSizes.clear();
		}

		// printOutputMatrix <<< 1, 1 >>> (numberOfInputPoints, vectorWidth, resultVectors);

		cudaError_t result;
		return result;
}