
#include "cutlassHMatrixXVectors.cuh"
#include "HMatrix.cuh"
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/gemm/device/gemm_batched.h>

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

cudaError_t cutlass_grouped_dgemm() {
}

cudaError_t cutlassHierarchicalXVec(
    unsigned int numberOfInputPoints, unsigned int  bucketSize, 
    unsigned int  numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix,
    H2Opus_Real *inputVectors, H2Opus_Real *resultVectors) {

		std::vector<cutlass::gemm::GemmCoord> problemSizes;

		using ElementA = double;
		using ElementB = double;
		using ElementOutput = double;
		using ElementAccumulator = double;
		using LayoutA = cutlass::layout::ColumnMajor;
		using LayoutB = cutlass::layout::ColumnMajor;
		using LayoutC = cutlass::layout::ColumnMajor;

		using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
			ElementA, LayoutA,
			cutlass::ComplexTransform::kNone, 8,
			ElementB, LayoutB,
			cutlass::ComplexTransform::kNone, 8,
			ElementOutput, LayoutC,
			ElementAccumulator,
			cutlass::arch::OpClassTensorOp,
			cutlass::arch::Sm80,
			cutlass::gemm::GemmShape<128, 128, 32>,
			cutlass::gemm::GemmShape<64, 64, 32>,
			cutlass::gemm::GemmShape<16, 8, 16>,
			cutlass::epilogue::thread::LinearCombination<
				ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
				ElementAccumulator, ElementAccumulator>,
			cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 
			4>::GemmKernel;

    	using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

		typename Gemm::EpilogueOutputOp::Params epilogue_1(1.0f, 0.0f);
		typename Gemm::EpilogueOutputOp::Params epilogue_2(1.0f, 1.0f);

      	// loop over levels
		for(unsigned int level = WAStruct.numLevels - 2; level > 0; --level) {

			int problemCount = hierarchicalMatrix.levels[level - 1].numTiles;
			problemSizes.clear();
			problemSizes.reserve(problemCount);
			int previousTileScanRank = 0;
			for (unsigned int tile = 0; tile < problemCount; ++tile) {
				int tileRank = hierarchicalMatrix.levels[level - 1].tileScanRanks[tiles] - previousTileScanRank;
				previousTileScanRank += tileRank;
				unsigned int tileDimension = 1 << (hierarchicalMatrix..numLevels - (level + 1))*bucketSize;
				cutlass::gemm::GemmCoord problem(tileRank, vectorWidth, tileDimension);
      			problemSizes.push_back(problem);
			}

			// TODO: preprocess each level
			int threadblock_count = Gemm::sufficient(problemSizes.data(), problemCount);
			if (!threadblock_count) {
				printf("Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");
				return result;
			}

			typename Gemm::Arguments args(
				problem_sizes_device.get(),
				problemCount,
				threadblock_count,
				epilogue_1,
				ptr_A.get(), // double ptr to where A matrices start
				ptr_B.get(),
				ptr_C.get(),
				ptr_D.get(),
				lda.get(), // ptr to lda of each matrix
				ldb.get(),
				ldc.get(),
				ldd.get(),
				problemSizes.data() // ptr to where data in vector starts
			);

			Gemm gemm;
			size_t workspace_size = gemm.get_workspace_size(args);
			cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

			result.status = gemm.initialize(args, workspace.get());

			gemm.run();
		}
}