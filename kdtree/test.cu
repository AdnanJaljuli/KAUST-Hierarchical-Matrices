#include "tlr_example.h"
// #include <cub/cub.cuh>
#include <iostream>

#define BUCKET_SIZE 8
#define printOutput 0

typedef double H2Opus_Real;

// TODO: fix size of memory allocated for offsets and currDimArray

// __global__ void fillArrays(unsigned int n, unsigned int dim, unsigned int num_segments, unsigned int segmentSize, int* offsets_sort, int* d_offsets_reduce){
//     printf("entered fillarrays\n");
//     unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
//     offsets_sort[i] = i*segmentSize;
//     for(unsigned int j=0; j<dim; ++j){
//         d_offsets_reduce[j*n + i] = i*segmentSize + segmentSize*(num_segments-1)*j;
//     }
//     if(threadIdx.x == 0 && blockIdx.x ==0){
//         printf("offsets_reduce\n");
//         for(unsigned int i=0; i<num_segments*dim + dim-1; ++i){
//             printf("%d\n", d_offsets_reduce);
//         }
//     }
//     // keys_in[i] = inputValues[currDimArray[i/segmentSize]*n + values_in[i]];
// }

__global__ void initializeArrays(unsigned int n, int* values_in, int* currDimArray){
    if(threadIdx.x==0){
      printf("entered initialize arrays\n");
    }
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    values_in[i] = i;
    currDimArray[i] = 1;
}

int main(){
    unsigned int n = 16;
    unsigned int dim = 3;

    printf("N = %d\n", (int)n);
    // Create point cloud
    PointCloud<H2Opus_Real> pt_cloud(dim, n);
    generateGrid<H2Opus_Real>(pt_cloud);
    printf("dimension: %d\n", pt_cloud.getDimension());
    printf("bucket size: %d\n", BUCKET_SIZE);

    #if printOutput
    printf("created point cloud\n");
    for(int i=0; i<n; ++i){
        for(int j=0; j<dim;++j){
            printf("%f ", pt_cloud.pts[j][i]);
        }
        printf("\n");
    }
    printf("\n\n");
    #endif

    H2Opus_Real *ptr;
    ptr = (H2Opus_Real*)malloc(n*dim*sizeof(H2Opus_Real));
    for (unsigned int i = 0; i < dim; ++i){
        for(unsigned int j = 0; j < n; ++j){
            ptr[i*n+j] = pt_cloud.getDataPoint(j, i);
        }
    }

    H2Opus_Real *ptr_d;
    cudaMalloc((void**) &ptr_d, n*dim*sizeof(H2Opus_Real));
    cudaMemcpy(ptr_d, ptr, n*dim*sizeof(H2Opus_Real*), cudaMemcpyHostToDevice);

    unsigned int segmentSize = n;
    unsigned int  num_segments = 1;
    unsigned int num_segments_reduce = num_segments*dim + dim-1;

    // int *d_offsets_sort;         // e.g., [0, 3, 3, 7]
    // int *d_offsets_reduce;
    // H2Opus_Real *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    // H2Opus_Real *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    // int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
    int *currDimArray_d;

    // cudaMalloc((void**) &d_offsets_sort, (n+1)*sizeof(int));
    // cudaMalloc((void**) &d_offsets_reduce, (n+1)*dim*sizeof(int));
    // cudaMalloc((void**) &d_keys_in,  n*sizeof(H2Opus_Real));
    // cudaMalloc((void**) &d_keys_out,  n*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_values_in, n*sizeof(int));
    // cudaMalloc((void**) &d_values_out,  n*sizeof(int));
    cudaMalloc((void**) &currDimArray_d, (n+1)*sizeof(int));
    printf("after cudaMallocs\n");

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;
    printf("numBlocks: %u\n", numBlocks);

    cudaDeviceSynchronize();
    initializeArrays<<<numBlocks, numThreadsPerBlock>>>(n, d_values_in, currDimArray_d);
    cudaDeviceSynchronize();

    // void *d_temp_storage = NULL;
    // size_t temp_storage_bytes = 0;

    // int *d_temp;
    // cudaMalloc((void**) &d_temp, n*sizeof(int));

    // while(segmentSize > BUCKET_SIZE) {
        // printf("entered while loop\n");
        // TODO: find values of numBlocks and numThreadsPerBlock using a function
        // numThreadsPerBlock = 1024;
        // numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;

        // fillArrays<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segmentSize, d_offsets_sort, d_offsets_reduce);
        // cudaDeviceSynchronize();

        // // TODO: find dimension of max span
        // d_temp_storage = NULL;
        // temp_storage_bytes = 0;

        // cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out,
        //     num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);

        // cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out,
        //     num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);
        // cudaDeviceSynchronize();

        // d_temp_storage = NULL;
        // temp_storage_bytes = 0;

        // cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        // d_keys_in, d_keys_out, d_values_in, d_values_out,
        // n, num_segments, d_offsets_sort, d_offsets_sort + 1);

        // cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        // d_keys_in, d_keys_out, d_values_in, d_values_out,
        // n, num_segments, d_offsets_sort, d_offsets_sort + 1);
        // cudaDeviceSynchronize();

        // cudaFree(d_temp_storage);

        // d_temp = d_values_in;
        // d_values_in = d_values_out;
        // d_values_out = d_temp;

    //     num_segments*=2;
    //     num_segments_reduce*=2;
    //     segmentSize/=2;
    // }

    // int *index_map = (int*)malloc(n*sizeof(int));
    // cudaMemcpy(index_map, d_values_in, n*sizeof(int), cudaMemcpyDeviceToHost);

    // #if printOutput
    // for(int i=0; i<n; ++i){
    //     for(int j=0; j<dim; ++j){
    //         printf("%f ", pt_cloud.pts[j][index_map[i]]);
    //     }
    //     printf("\n");
    // }
    // #endif

    // free(ptr);
    // cudaFree(ptr_d);
    // cudaFree(d_offsets_sort);
    // cudaFree(d_keys_in);
    // cudaFree(d_keys_out);
    // cudaFree(d_values_in);
    // cudaFree(d_values_out);
    // cudaFree(d_temp);
}