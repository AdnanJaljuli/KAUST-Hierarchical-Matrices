#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#define TILE_DIM 32
// TODO: multiply diagonal blocks
// TODO: go over for loops. might not be <TILE_DIM

__global__ void TLR_GEMM_kernel(TLR_Matrix matrix1,TLR_Matrix matrix2,float* matrix3,unsigned int N, unsigned int blockSize){
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    // allocate memory for two tiles and reuse the memory for different tiles.
    __shared__ float first_matrix[TILE_DIM][TILE_DIM];
    __shared__ float second_matrix[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    if(blockIdx.x != blockIdx.y){
        unsigned int matrix1_rank = matrix1.blockRanks[blockIdx.x*(matrix1.nBlocks-1) + blockIdx.y];
        unsigned int matrix2_rank = matrix2.blockRanks[blockIdx.x*(matrix2.nBlocks-1) + blockIdx.y];

        for(unsigned int tile = 0; tile<(N/TILE_DIM); ++tile){
            // loads Av_s into shared memory
            if(threadIdx.x < matrix1_rank){
                first_matrix[threadIdx.x][threadIdx.y] = matrix1.v[matrix1.lowRankBlockPointers[blockIdx.y*matrix1.nBlocks + blockIdx.x] + threadIdx.y*TILE_DIM + threadIdx.x];
            }
            // loads Bu_s into shared memory
            if(threadIdx.y < matrix2_rank){
                second_matrix[threadIdx.x][threadIdx.y] = matrix2.u[matrix2.lowRankBlockPointers[blockIdx.y*matrix2.nBlocks + blockIdx.x] + threadIdx.y*matrix2_rank + threadIdx.x];
            }
            __syncthreads();

            float temp = 0.0f;
            // multiplies Av_s*Bu_s
            // stores result in place of Av_s
            for(unsigned int i = 0; i < TILE_DIM; ++i){
                if(threadIdx.x < matrix2_rank && threadIdx.y < matrix1_rank){
                    temp += first_matrix[threadIdx.y][i]*second_matrix[i][threadIdx.x];
                }
            }
            __syncthreads();
            if(threadIdx.x < matrix2_rank && threadIdx.y < matrix1_rank){
                first_matrix[threadIdx.x][threadIdx.y] = temp;
            }
            __syncthreads();

            // loads Bv_s into shared memory. Replaces Bu_s
            if(threadIdx.x < matrix2_rank){
                second_matrix[threadIdx.x][threadIdx.y] = matrix2.v[matrix2.lowRankBlockPointers[blockIdx.y*matrix2.nBlocks + blockIdx.x] + threadIdx.y*TILE_DIM + threadIdx.x];
            }
            __syncthreads();

            temp = 0.0f;
            // multiplies (Av_s*Bu_s)*Bv_s and stores result in Av_s space
            for(unsigned int i = 0; i < TILE_DIM; ++i){
                if(threadIdx.y < matrix1_rank){
                    temp += first_matrix[threadIdx.y][i]*second_matrix[i][threadIdx.x];
                }
            }
            __syncthreads();
            if(threadIdx.y < matrix1_rank){
                first_matrix[threadIdx.x][threadIdx.y] = temp;
            }
            __syncthreads();

            // loads Au_s into shared memory
            if(threadIdx.y < matrix1_rank){
                second_matrix[threadIdx.x][threadIdx.y] = matrix1.u[matrix1.lowRankBlockPointers[blockIdx.y*matrix1.nBlocks + blockIdx.x] + threadIdx.y*matrix1_rank + threadIdx.x];
            }
            __syncthreads();
            // multiplies Au_s*(Av_s*Bu_s*Bv_s) and stores result in sum
            for(unsigned int i = 0; i < matrix1_rank; ++i){
                sum += first_matrix[threadIdx.y][i]*second_matrix[i][threadIdx.x];
            }
            __syncthreads();
        }
    }
    else {
        for(unsigned int tile = 0; tile<N/TILE_DIM; ++tile){
            first_matrix[threadIdx.y][threadIdx.x] = matrix1.diagonal[tile*TILE_DIM*TILE_DIM + threadIdx.x*TILE_DIM + threadIdx.y];
            second_matrix[threadIdx.y][threadIdx.x] = matrix2.diagonal[tile*TILE_DIM*TILE_DIM + threadIdx.x*TILE_DIM + threadIdx.y];
            __syncthreads();

            for(unsigned int i=0; i<TILE_DIM; ++i){
                sum += first_matrix[threadIdx.y][i]*second_matrix[i][threadIdx.x];
            }
            __syncthreads();
        }
    }
    matrix3[col*N + row] = sum;
}