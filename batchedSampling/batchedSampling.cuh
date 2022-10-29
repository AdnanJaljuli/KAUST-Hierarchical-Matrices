#ifndef __BATCHED_SAMPLING__
#define __BATCHED_SAMPLING__

#include <stdio.h>

static __host__ __device__ int getMOIndexfromXY(unsigned int x, unsigned int y){
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

static __host__ __device__ int IndextoMOIndex(int numSegments, int n){
    unsigned int i = n%numSegments;
    unsigned int j = n/numSegments;
    return getMOIndexfromXY(j, i);
}

template<class T>
__global__ void batchedSampling(
    int tile_size, int batch_unit_size, T** U_ptrs, T** V_ptrs, int** scan_ranks,
    T** B_batch, T** A_batch, int* samples_batch, int max_rows, int max_cols, int max_samples, int transpose) {
		if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
			printf("samples batch\n");
			for(unsigned int i = 0; i < 2; ++i) {
				printf("%d ", samples_batch[i]);
			}
			printf("\n");
		}
		
		T outputValue = 0;
		T sum = 0;

		if(transpose == 0) {
			__shared__ T shmemArray1[32][16];
			__shared__ T shmemArray2[32][16];
			unsigned int batch = blockIdx.y;
			unsigned int blockInBatch = blockIdx.x;
			
			for(unsigned int tile = 0; tile < batch_unit_size; ++tile) {
				int MOIndex = IndextoMOIndex(batch_unit_size, tile*batch_unit_size + blockInBatch);
				int rank, scanRankVal;
				if(blockInBatch == 0 && tile == 0) {
					rank = scan_ranks[batch][0];
					scanRankVal = 0;
				}
				else {
					rank = scan_ranks[batch][MOIndex] - scan_ranks[batch][MOIndex - 1];
					scanRankVal = scan_ranks[batch][MOIndex - 1];
				}

				// load V and Omega into shared memory
				if(threadIdx.x < rank) {
					shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[batch][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
				}
				if(rank*2 <= max_samples) {
					if(threadIdx.x >= max_samples - rank) {
						shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - max_samples)*tile_size + threadIdx.y];
					}
				}
				if(threadIdx.x < samples_batch[batch]) {
					shmemArray2[threadIdx.y][threadIdx.x] = B_batch[batch][threadIdx.x*batch_unit_size*tile_size + tile*tile_size + threadIdx.y];
				}
				__syncthreads();

				if(threadIdx.y < rank && threadIdx.x < samples_batch[batch]) {
					// TODO: use warp shuffling: allocate multiple threadblocks per output element and let each do a part of the multiplication and then use shuffling
					sum = 0;
					for(unsigned int j = 0; j < tile_size; ++j) {
						sum += shmemArray1[j][threadIdx.y]*shmemArray2[j][threadIdx.x];
					}
				}
				__syncthreads();

				if(threadIdx.y < rank && threadIdx.x < samples_batch[batch]) {
					shmemArray2[threadIdx.y][threadIdx.x] = sum;
				}
				if(2*rank > max_samples) {
					if(threadIdx.x >= max_samples - rank) {
						shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - max_samples)*tile_size + threadIdx.y];
					}
				}
				__syncthreads();

				// multiply U by the result and add it to output
				sum = 0;
				if(threadIdx.x < samples_batch[batch]) {
					for(unsigned int j = 0; j < rank; ++j) {
						sum += shmemArray1[threadIdx.y][max_samples - rank + j]*shmemArray2[j][threadIdx.x];
					}
					outputValue += sum;
				}
				__syncthreads();
			}
			if(threadIdx.x < samples_batch[batch]) {
				A_batch[batch][threadIdx.x*batch_unit_size*tile_size + blockInBatch*tile_size + threadIdx.y] = outputValue;
			}
		}
		else {
			__shared__ T shmemArray1[32][32];
			__shared__ T shmemArray2[32][32];
			volatile unsigned int colInBatch = blockIdx.x*blockDim.x + threadIdx.x;
			volatile unsigned int rowInBatch = blockIdx.y*blockDim.y + threadIdx.y;

			for(unsigned int tile = 0; tile < batch_unit_size; ++tile) {
				// multiply V by the sampling vector and store it in buffer
				volatile int MOIndex = IndextoMOIndex(batch_unit_size, blockIdx.y*batch_unit_size + tile);
				volatile int rank, scanRankVal;
				if(colInBatch < samples_batch[blockIdx.z]) {
					if(blockIdx.y == 0 && tile == 0) {
						rank = scan_ranks[blockIdx.z][0];
						scanRankVal = 0;
					}
					else {
						rank = scan_ranks[blockIdx.z][MOIndex] - scan_ranks[blockIdx.z][MOIndex - 1];
						scanRankVal = scan_ranks[blockIdx.z][MOIndex - 1];
					}
				}

				// load U transpose and Omega into shared memory
				if(colInBatch < samples_batch[blockIdx.z]) {
					if(threadIdx.x < rank) {
						shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[blockIdx.z][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
					}
					// else if(threadIdx.x >= blockDim.x - rank) {
					// 	shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[blockIdx.z][scanRankVal*tile_size + (threadIdx.x + rank - blockDim.x)*tile_size + threadIdx.y];
					// }
					shmemArray2[threadIdx.y][threadIdx.x] = B_batch[blockIdx.z][colInBatch*batch_unit_size*tile_size + tile*tile_size + threadIdx.y];
				}
				__syncthreads();

				if(colInBatch < samples_batch[blockIdx.z]) {
					if(threadIdx.y < rank) {
						sum = 0;
						for(unsigned int j = 0; j < tile_size; ++j) {
							sum += shmemArray1[j][threadIdx.y]*shmemArray2[j][threadIdx.x];
						}
					}
				}
				__syncthreads();

				if(colInBatch < samples_batch[blockIdx.z]) {
					if(threadIdx.x < rank) {
						shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[blockIdx.z][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
					}
					if(threadIdx.y < rank) {
						shmemArray2[threadIdx.y][threadIdx.x] = sum;
					}
				}
				__syncthreads();

				// multiply V by the result and add it to output
				sum = 0;
				if(colInBatch < samples_batch[blockIdx.z]) {
					for(unsigned int j = 0; j < rank; ++j) {
						sum += shmemArray1[threadIdx.y][j]*shmemArray2[j][threadIdx.x];
					}
					outputValue += sum;
				}
				__syncthreads();
			}
			if(colInBatch < samples_batch[blockIdx.z]) {
				A_batch[blockIdx.z][colInBatch*batch_unit_size*tile_size + rowInBatch] = outputValue;
			}
		}
}


#if 0
// TODO: have one array for scanRanks, or an array of pointers similar to U and V?
template <typename T>
static __global__ void batchedSampling(int tile_size, int batch_unit_size, T** U_ptrs, T** V_ptrs, int** scan_ranks, int batchSize, T** samplingVectors, int samplingVectorWidth, T** output, int transpose) {
    __shared__ T shmemArray1[32][16];
    __shared__ T shmemArray2[32][16];
    unsigned int batch = blockIdx.y;
    unsigned int blockInBatch = blockIdx.x;
    T outputValue = 0;
    T sum = 0;

    if(transpose == 0) {
        for(unsigned int tile = 0; tile < batch_unit_size; ++tile) {
            // multiply V by the sampling vector and store it in buffer
            int MOIndex = IndextoMOIndex(batch_unit_size, tile*batch_unit_size + blockInBatch);
            int rank, scanRankVal;
            if(blockInBatch == 0 && tile == 0) {
                rank = scan_ranks[batch][0];
                scanRankVal = 0;
            }
            else {
                rank = scan_ranks[batch][MOIndex] - scan_ranks[batch][MOIndex - 1];
                scanRankVal = scan_ranks[batch][MOIndex - 1];
            }

            // load V and Omega into shared memory
            if(threadIdx.x < rank) {
                shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[batch][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
            }
            if(rank*2 <= samplingVectorWidth) {
                if(threadIdx.x >= samplingVectorWidth - rank) {
                    shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - samplingVectorWidth)*tile_size + threadIdx.y];
                }
            }
            shmemArray2[threadIdx.y][threadIdx.x] = samplingVectors[batch][threadIdx.x*batch_unit_size*tile_size + tile*tile_size + threadIdx.y];
            __syncthreads();

            if(threadIdx.y < rank) {
                // TODO: use warp shuffling: allocate multiple threadblocks per output element and let each do a part of the multiplication and then use shuffling
                sum = 0;
                for(unsigned int j = 0; j < tile_size; ++j) {
                    sum += shmemArray1[j][threadIdx.y]*shmemArray2[j][threadIdx.x];
                }
            }
            __syncthreads();

            if(threadIdx.y < rank) {
                shmemArray2[threadIdx.y][threadIdx.x] = sum;
            }
            if(2*rank > samplingVectorWidth) {
                if(threadIdx.x >= samplingVectorWidth - rank) {
                    shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - samplingVectorWidth)*tile_size + threadIdx.y];
                }
            }
            __syncthreads();

            // multiply U by the result and add it to output
            sum = 0;
            for(unsigned int j = 0; j < rank; ++j) {
                sum += shmemArray1[threadIdx.y][samplingVectorWidth - rank + j]*shmemArray2[j][threadIdx.x];
            }
            outputValue += sum;
            __syncthreads();
        }
    }
    else {
        for(unsigned int tile = 0; tile < batch_unit_size; ++tile) {
            // multiply V by the sampling vector and store it in buffer
            int MOIndex = IndextoMOIndex(batch_unit_size, blockInBatch*batch_unit_size + tile);
            int rank, scanRankVal;
            if(blockInBatch == 0 && tile == 0) {
                rank = scan_ranks[batch][0];
                scanRankVal = 0;
            }
            else {
                rank = scan_ranks[batch][MOIndex] - scan_ranks[batch][MOIndex - 1];
                scanRankVal = scan_ranks[batch][MOIndex - 1];
            }

            // load U transpose and Omega into shared memory
            if(threadIdx.x < rank) {
                shmemArray1[threadIdx.y][threadIdx.x] = U_ptrs[batch][scanRankVal*tile_size + threadIdx.x*tile_size + threadIdx.y];
            }
            if(rank*2 <= samplingVectorWidth) {
                if(threadIdx.x >= samplingVectorWidth - rank) {
                    shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - samplingVectorWidth)*tile_size + threadIdx.y];
                }
            }
            shmemArray2[threadIdx.y][threadIdx.x] = samplingVectors[batch][threadIdx.x*batch_unit_size*tile_size + tile*tile_size + threadIdx.y];
            __syncthreads();

            if(threadIdx.y < rank) {
                sum = 0;
                for(unsigned int j = 0; j < tile_size; ++j) {
                    sum += shmemArray1[j][threadIdx.y]*shmemArray2[j][threadIdx.x];
                }
            }
            __syncthreads();

            if(threadIdx.y < rank) {
                shmemArray2[threadIdx.y][threadIdx.x] = sum;
            }
            if(2*rank > samplingVectorWidth) {
                if(threadIdx.x >= samplingVectorWidth - rank) {
                    shmemArray1[threadIdx.y][threadIdx.x] = V_ptrs[batch][scanRankVal*tile_size + (threadIdx.x + rank - samplingVectorWidth)*tile_size + threadIdx.y];
                }
            }
            __syncthreads();

            // multiply V by the result and add it to output
            sum = 0;
            for(unsigned int j = 0; j < rank; ++j) {
                sum += shmemArray1[threadIdx.y][samplingVectorWidth - rank + j]*shmemArray2[j][threadIdx.x];
            }
            outputValue += sum;
            __syncthreads();
        }
    }
    output[batch][threadIdx.x*batch_unit_size*tile_size + blockInBatch*tile_size + threadIdx.y] = outputValue;
}
#endif

#endif