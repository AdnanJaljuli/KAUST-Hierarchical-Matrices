#ifndef UTILITIES_CUH
#define UTILITIES_CUH

extern "C" int iDivUp(int, int);
extern "C" void gpuErrchk(cudaError_t);
extern "C" void cusolveSafeCall(cusolverStatus_t);

#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line);
        //fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
        //        file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        DEVICE_RESET
            // Make sure we call CUDA Device Reset before exiting
            exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif
#endif