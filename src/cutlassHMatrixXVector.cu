
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
	std::vector<int64_t> *lda_host, std::vector<int64_t> *ldb_host, std::vector<int64_t> *ldc_host,
	cutlass::DeviceAllocation<int64_t> *lda, cutlass::DeviceAllocation<int64_t> *ldb, cutlass::DeviceAllocation<int64_t> *ldc,
	H2Opus_Real *AMatrix, H2Opus_Real *BMatrix, H2Opus_Real *CMatrix,
	std::vector<H2Opus_Real*> *ptr_A_host, std::vector<H2Opus_Real*> *ptr_B_host, std::vector<H2Opus_Real*> *ptr_C_host,
	cutlass::DeviceAllocation<H2Opus_Real *> *ptr_A, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_B, cutlass::DeviceAllocation<H2Opus_Real *> *ptr_C,
	unsigned int iteration) {

		int *h_tileScanRanks = (int*)malloc(problemCount*sizeof(int));
		cudaMemcpy(h_tileScanRanks, matrixLevel.tileScanRanks, problemCount*sizeof(int), cudaMemcpyDeviceToHost);

		int previousTileScanRank = 0;
		for (unsigned int tile = 0; tile < problemCount; ++tile) {
			int tileRank = h_tileScanRanks[tile] - previousTileScanRank;

			unsigned int tileDimension = 1<<(numLevels - (level + 1))*bucketSize;
			if(iteration == 0) {
				cutlass::gemm::GemmCoord problem(tileRank, vectorWidth, tileDimension);
				h_problemSizes->push_back(problem);
			}
			else {
				cutlass::gemm::GemmCoord problem(tileDimension, vectorWidth, tileRank);
				h_problemSizes->push_back(problem);
			}

			if(iteration == 0) {
				lda_host->at(tile) = (int64_t)tileRank;
				ldb_host->at(tile) = (int64_t)numberOfInputPoints;
			}
			else {
				lda_host->at(tile) = (int64_t)numberOfInputPoints;
				ldb_host->at(tile) = (int64_t)tileRank;
			}
			ldc_host->at(tile) = (int64_t)numberOfInputPoints;

			ptr_A_host->at(tile) = AMatrix + previousTileScanRank*tileDimension;
			ptr_B_host->at(tile) = BMatrix + tile*tileDimension;
			ptr_C_host->at(tile) = CMatrix + tile*tileDimension;

			previousTileScanRank += tileRank;
		}

		d_problemSizes->reset(problemCount);
 		d_problemSizes->copy_from_host(h_problemSizes->data());
		h_problemSizes->clear();

		lda->copy_from_host(lda_host->data());
		ldb->copy_from_host(ldb_host->data());
		ldc->copy_from_host(ldc_host->data());

		ptr_A->copy_from_host(ptr_A_host->data());
		ptr_B->copy_from_host(ptr_B_host->data());
		ptr_C->copy_from_host(ptr_C_host->data());

		free(h_tileScanRanks);
}

cudaError_t cutlass_grouped_dgemm() {
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

		using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
			ElementInput, 
			cutlass::layout::ColumnMajor, 
			cutlass::ComplexTransform::kNone,
			1,
			ElementInput,
			cutlass::layout::RowMajor, 
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

		typename Gemm::EpilogueOutputOp::Params epilogue_1(1.0f, 0.0f);
		typename Gemm::EpilogueOutputOp::Params epilogue_2(1.0f, 1.0f);

		std::vector<cutlass::gemm::GemmCoord> h_problemSizes;
		cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> d_problemSizes;

		std::vector<int64_t> lda_host(numSegments);
		std::vector<int64_t> ldb_host(numSegments);
		std::vector<int64_t> ldc_host(numSegments);

		cutlass::DeviceAllocation<int64_t> lda;
		cutlass::DeviceAllocation<int64_t> ldb;
		cutlass::DeviceAllocation<int64_t> ldc;
		lda.reset(numSegments);
		ldb.reset(numSegments);
		ldc.reset(numSegments);

		std::vector<ElementInput*> ptr_A_host(numSegments);
		std::vector<ElementInput*> ptr_B_host(numSegments);
		std::vector<ElementAccumulator*> ptr_C_host(numSegments);

		cutlass::DeviceAllocation<ElementInput *> ptr_A;
		cutlass::DeviceAllocation<ElementInput *> ptr_B;
		cutlass::DeviceAllocation<ElementAccumulator *> ptr_C;
		ptr_A.reset(numSegments);
		ptr_B.reset(numSegments);
		ptr_C.reset(numSegments);

      	// loop over levels
		for(unsigned int level = hierarchicalMatrix.numLevels - 2; level > 0; --level) {
			// preprocess each level
			int problemCount = hierarchicalMatrix.levels[level - 1].numTiles;

			// V*inputVectors
			preprocessGroupedGEMM(numberOfInputPoints, level, hierarchicalMatrix.numLevels,
				problemCount, bucketSize, vectorWidth,
				hierarchicalMatrix.levels[level - 1],
				&h_problemSizes, &d_problemSizes,
				&lda_host, &ldb_host, &ldc_host,
				&lda, &ldb, &ldc,
				hierarchicalMatrix.levels[level - 1].V, inputVectors, bufferVectors,
				&ptr_A_host, &ptr_B_host, &ptr_C_host,
				&ptr_A, &ptr_B, &ptr_C,
				0);

			int threadblockCount = Gemm::sufficient(h_problemSizes.data(), problemCount);
			if (!threadblockCount) {
				printf("Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");
			}

			typename Gemm::Arguments args_1(
				d_problemSizes.get(),
				problemCount,
				threadblockCount,
				epilogue_1,
				ptr_A.get(),
				ptr_B.get(),
				ptr_C.get(),
				ptr_C.get(),
				lda.get(),
				ldb.get(),
				ldc.get(),
				ldc.get(),
				h_problemSizes.data() // ptr to where data in vector starts
			);

			Gemm gemm_1;
			size_t workspace_size = gemm_1.get_workspace_size(args_1);
			cutlass::DeviceAllocation<uint8_t> workspace_1(workspace_size);
			gemm_1.initialize(args_1, workspace_1.get());
			gemm_1.run();

			// U*results
			preprocessGroupedGEMM(numberOfInputPoints, level, hierarchicalMatrix.numLevels,
				problemCount, bucketSize, vectorWidth,
				hierarchicalMatrix.levels[level - 1],
				&h_problemSizes, &d_problemSizes,
				&lda_host, &ldb_host, &ldc_host,
				&lda, &ldb, &ldc,
				hierarchicalMatrix.levels[level - 1].U, bufferVectors, resultVectors,
				&ptr_A_host, &ptr_B_host, &ptr_C_host,
				&ptr_A, &ptr_B, &ptr_C,
				1);

			typename Gemm::Arguments args_2(
				d_problemSizes.get(),
				problemCount,
				threadblockCount,
				epilogue_1,
				ptr_A.get(),
				ptr_B.get(),
				ptr_C.get(),
				ptr_C.get(),
				lda.get(),
				ldb.get(),
				ldc.get(),
				ldc.get(),
				h_problemSizes.data() // ptr to where data in vector starts
			);

			Gemm gemm_2;
			workspace_size = gemm_2.get_workspace_size(args_2);
			cutlass::DeviceAllocation<uint8_t> workspace_2(workspace_size);
			gemm_2.initialize(args_2, workspace_2.get());
			gemm_2.run();
		}

		// printOutputMatrix <<< 1, 1 >>> (numberOfInputPoints, vectorWidth, resultVectors);

		cudaError_t result;
		return result;
}