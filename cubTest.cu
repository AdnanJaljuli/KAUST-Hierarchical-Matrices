// #include <cub/cub.cuh>
// #include <iostream>
// #include <stdio.h>      /* printf, scanf, puts, NULL */
// #include <stdlib.h>     /* srand, rand */
// #include <time.h>       /* time */

// int main(){
//     int  num_items=7;          // e.g., 7
//     int  num_segments=2;       // e.g., 3
    
//     int offsets[3] = {1, 3, 6};
//     int  *d_offsets;         // e.g., [0, 3, 3, 7]

//     int keys_in[7] = {8, 6, 7, 5, 3, 0, 9};
//     int  *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]

//     int keys_out[7];
//     int  *d_keys_out;        // e.g., [-, -, -, -, -, -, -]

//     int values_in[7] = {0, 1, 2, 3, 4, 5, 6};
//     int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]

//     int values_out[7];
//     int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
    
//     void     *d_temp_storage = NULL;
//     size_t   temp_storage_bytes = 0;

//     cudaMalloc((void**) &d_offsets, 4*sizeof(int));
//     cudaMalloc((void**) &d_keys_in,  num_items*sizeof(int));
//     cudaMalloc((void**) &d_keys_out,  num_items*sizeof(int));
//     cudaMalloc((void**) &d_values_in, num_items*sizeof(int));
//     cudaMalloc((void**) &d_values_out,  num_items*sizeof(int));

//     cudaMemcpy(d_offsets, offsets, 3*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_keys_in, keys_in, num_items*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_keys_out, keys_out, num_items*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_values_in, values_in, num_items*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_values_out, values_out, num_items*sizeof(int), cudaMemcpyHostToDevice);

//     cudaDeviceSynchronize();
//     for(int i=0; i<num_items; ++i){
//         printf("%d ", keys_in[i]);
//     }
//     printf("\n");

//     for(int i=0; i<num_items; ++i){
//         printf("%d ", values_in[i]);
//     }
//     printf("\n");
//     printf("\n");

//     cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
//         d_keys_in, d_keys_out, d_values_in, d_values_out,
//         num_items, num_segments, d_offsets, d_offsets + 1);

//     // Allocate temporary storage
//     cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     printf("temp_storage_array: %zu\n", temp_storage_bytes);
//     // Run sorting operation
//     cudaDeviceSynchronize();
    
//     cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
//         d_keys_in, d_keys_out, d_values_in, d_values_out,
//         num_items, num_segments, d_offsets, d_offsets + 1);
//     // d_keys_out            <-- [6, 7, 8, 0, 3, 5, 9]
//     // d_values_out          <-- [1, 2, 0, 5, 4, 3, 6]

//     cudaDeviceSynchronize();


//     cudaMemcpy(keys_out, d_keys_out, num_items*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaMemcpy(values_out, d_values_out, num_items*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();

//     for(int i=0; i<num_items; ++i){
//         printf("%d ", keys_out[i]);
//     }
//     printf("\n");

//     for(int i=0; i<num_items; ++i){
//         printf("%d ", values_out[i]);
//     }
//     printf("\n");
//     printf("\n");
// }

#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

int main(){
    // Declare, allocate, and initialize device-accessible pointers for input and output
    int  num_items=7;      // e.g., 7
    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
    int  *d_out;         // e.g., [-]
    
    int *in = (int*)malloc(7*sizeof(int));
    int *out = (int*)malloc(sizeof(int));
    
    for(int i=0; i<7; ++i){
        in[i] = i;
    }
    cudaMalloc((void**) &d_in, 7*sizeof(int));
    cudaMalloc((void**) &d_out, sizeof(int));
    cudaMemcpy(d_in, in, 7*sizeof(int), cudaMemcpyHostToDevice);
    
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", out);
}