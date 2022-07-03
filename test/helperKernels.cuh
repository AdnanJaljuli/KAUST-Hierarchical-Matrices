
__global__ void fillARAArrays(int batchCount, int max_rows, int max_cols, int* d_rows_batch, int* d_cols_batch, H2Opus_Real** d_M_ptrs, H2Opus_Real* d_M, int* d_ldm_batch, int* d_lda_batch, int* d_ldb_batch){
    for(unsigned int i=0; i<batchCount; ++i){
        d_rows_batch[i] = max_rows;
        d_cols_batch[i] = max_cols;
        d_ldm_batch[i] = max_rows;
        d_lda_batch[i] = max_rows;
        d_ldb_batch[i] = max_rows;
    }
}

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
void generateArrayOfPointersT(T* original_array, T** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{
    // printf("generate array of pointers\n");
  thrust::device_ptr<T*> dev_data(array_of_arrays);

  thrust::transform(
    thrust::cuda::par.on(stream),
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(num_arrays),
    dev_data,
    UnaryAoAAssign<T>(original_array, stride)
    );
    // printf("ended generate array of pointers\n");
  cudaGetLastError();
}

// __global__ void printOutput(H2Opus_Real* d_A, H2Opus_Real* d_B, int* k, int batchCount){
//     printf("ks\n");
//     for(unsigned int i=0;i<batchCount; ++i){
//         printf("%d ", k[i]);
//     }
// }