#include "kblas.h"
#include "batch_rand.h"
#include "batch_ara.h"

template<class T>
struct DenseSampler_mod
{
	T *U_batch, *V_batch;
	int *ranks, *scan_ranks;
	int *ldm_batch, *rows_batch, *cols_batch;
	kblasHandle_t handle;
	int max_rows, max_cols;

	DenseSampler_mod(T *U_batch, T *V_batch, int* ranks, int* scan_ranks, int *rows_batch, int *cols_batch, int max_rows, int max_cols, kblasHandle_t handle) 
	{
		this->U_batch = U_batch;
		this->V_batch = V_batch;
		this->ranks = ranks;
		this->scan_ranks = scan_ranks;
		this->ldm_batch = ldm_batch;
		this->rows_batch = rows_batch;
		this->cols_batch = cols_batch;
		this->max_rows = max_rows;
		this->max_cols = max_cols;
		this->handle = handle;
	}

	// A = M * B or A = M' * B
	int sample(T** B_batch, int* ldb_batch, int* samples_batch, T** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose) 
	{
        sample_batch(U_batch, V_batch, ranks, scan_ranks, B_batch, ldb_batch, samples_batch, A_batch, lda_batch, max_samples, num_ops, transpose);
		return 1;
	}
};

template<class Real, class Sampler>
int kblas_ara_batch_template_mod(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, Sampler& sampler, 
	H2Opus_Real** A_batch, int* lda_batch, H2Opus_Real** B_batch, int* ldb_batch, int* ranks_batch, 
	Real tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	int rank = 0;
	const Real tolerance_scale = 1; //7.978845608028654; // (10 * sqrt(2 / pi))
	tol *= tolerance_scale;
	
	Real** Y_batch;
	
	double* diag_R, *max_diag;
	double* G_strided;
	double** G_batch_mp;
	int* block_ranks, *op_samples, *small_vectors;
	
	///////////////////////////////////////////////////////////////////////////
	// Workspace allocation
	///////////////////////////////////////////////////////////////////////////
	KBlasWorkspaceState external_ws, total_ws;
	kblas_ara_batch_wsquery<Real>(external_ws, bs, num_ops, 1);
	kblas_ara_batch_wsquery<Real>(total_ws, bs, num_ops, 0);
	
	KBlasWorkspaceState available_ws = handle->work_space.getAvailable();
	
	if(!total_ws.isSufficient(&available_ws))
		return KBLAS_InsufficientWorkspace;
	
	// Align workspace to sizeof(double) bytes
	external_ws.d_data_bytes += external_ws.d_data_bytes % sizeof(double);
	
	G_strided = (double*)((KBlasWorkspace::WS_Byte*)handle->work_space.d_data + external_ws.d_data_bytes);
	diag_R    = G_strided + num_ops * bs * bs;
	max_diag  = diag_R + num_ops * bs;
	
	external_ws.d_data_bytes += ( 
			num_ops * bs * bs + // G
			num_ops * bs + // diag_R
			num_ops        // max_diag
	) * sizeof(double);

	// Align to sizeof(int) bytes
	external_ws.d_data_bytes += external_ws.d_data_bytes % sizeof(int);

	block_ranks = (int*)((KBlasWorkspace::WS_Byte*)handle->work_space.d_data + external_ws.d_data_bytes);
	op_samples = block_ranks + num_ops;
	small_vectors = op_samples + num_ops;

	// Align to sizeof(Real*) bytes
	external_ws.d_ptrs_bytes += external_ws.d_ptrs_bytes % sizeof(Real*);
	Y_batch = (Real**)((KBlasWorkspace::WS_Byte*)handle->work_space.d_ptrs + external_ws.d_ptrs_bytes);	

	external_ws.d_ptrs_bytes += num_ops * sizeof(Real*); // Y_batch

	// Align to sizeof(double*) bytes
	external_ws.d_ptrs_bytes += external_ws.d_ptrs_bytes % sizeof(double*);
	G_batch_mp = (double**)((KBlasWorkspace::WS_Byte*)handle->work_space.d_ptrs + external_ws.d_ptrs_bytes);	

	///////////////////////////////////////////////////////////////////////////
	// Initializations
	///////////////////////////////////////////////////////////////////////////
	cudaStream_t kblas_stream = kblasGetStream(handle);
	// printDenseMatrixGPU(M_batch, ldm_batch, rows_batch, cols_batch, 0, 15, "M");
	
	// Initialize operation data
	fillArray(max_diag, num_ops, -1, kblas_stream);
	fillArray(small_vectors, num_ops, 0, kblas_stream);
	fillArray(ranks_batch, num_ops, 0, kblas_stream);
	
	// Copy over A to Y so we can advance Y 
	copyGPUArray(A_batch, Y_batch, num_ops, kblas_stream);
	
	// Generate array of pointers from strided data
	generateArrayOfPointers(G_strided, G_batch_mp, bs * bs, 0, num_ops, kblas_stream);

	///////////////////////////////////////////////////////////////////////////
	// Main Loop
	///////////////////////////////////////////////////////////////////////////
	while(rank < max_rank)
	{
		int samples = std::min(bs, max_rank - rank);
		
		// Set the op samples to 0 if the operation has converged
		int converged = kblas_ara_batch_set_samples(
			op_samples, small_vectors, 
			samples, r, num_ops, kblas_stream
		);
		
		if(converged == 1) break;
		
		// Generate random matrices Omega stored in B
		// Omega = randn(n, samples)
		Real** Omega = B_batch;
		check_error( kblas_rand_batch(
			handle, cols_batch, op_samples, Omega, ldb_batch, 
			max_cols, rand_state, num_ops
		) );
		
		// printDenseMatrixGPU(Omega, ldb_batch, cols_batch, op_samples, 0, 16, "Omega");
		
		// Take samples and store them in A
		// Y = M * Omega
		sampler.sample(
			Omega, ldb_batch, op_samples, 
			Y_batch, lda_batch, samples, num_ops, 0
		);

		// Set diag(R) = 1
		fillArray(diag_R, num_ops * bs, 1, kblas_stream);

		// Block CGS with one reorthogonalization step
		for(int i = 0; i < 2; i++)
		{
			// Project samples
			// Y = Y - Q * (Q' * Y) = Y - Q * Z
			// Store Z = Q' * Y in B
			Real** Z_batch = B_batch, **Q_batch = A_batch;
			check_error( kblas_gemm_batch(
				handle, KBLAS_Trans, KBLAS_NoTrans, ranks_batch, op_samples, rows_batch, 
				rank, samples, max_rows, 1, (const Real**)Q_batch, lda_batch,  
				(const Real**)Y_batch, lda_batch, 0, Z_batch, ldb_batch, num_ops
			) );
			check_error( kblas_gemm_batch(
				handle, KBLAS_NoTrans, KBLAS_NoTrans, rows_batch, op_samples, ranks_batch, 
				max_rows, samples, rank, -1, (const Real**)Q_batch, lda_batch,  
				(const Real**)Z_batch, ldb_batch, 1, Y_batch, lda_batch, num_ops
			) );

			// Pivoted panel orthogonalization using syrk+pstrf+trsm
			// Compute G = A'*A in mixed precision
			Real **R_batch = B_batch;
			check_error( kblas_ara_mp_syrk_batch_template(
				handle, rows_batch, op_samples, max_rows, samples, 
				(const Real**)Y_batch, lda_batch, G_batch_mp, op_samples, num_ops
			) );
			check_error( kblas_ara_fused_potrf_batch_template( 
				op_samples, G_batch_mp, op_samples, R_batch, ldb_batch, diag_R, bs, 
				block_ranks, num_ops, kblas_stream
			) );
			
			/*check_error( kblas_gemm_batch(
				handle, KBLAS_Trans, KBLAS_NoTrans, op_samples, op_samples, rows_batch, 
				samples, samples, max_rows, 1, (const Real**)Y_batch, lda_batch, 
				(const Real**)Y_batch, lda_batch, 0, R_batch, ldb_batch, num_ops
			) ); 
			check_error( kblas_ara_fused_potrf_batch( 
				op_samples, R_batch, ldb_batch, R_batch, ldb_batch, diag_R, bs, 
				block_ranks, num_ops, kblas_stream
			) );
			*/
			// Copy the ranks over to the samples in case the rank was less than the samples
			copyGPUArray(block_ranks, op_samples, num_ops, kblas_stream);

			// printDenseMatrixGPU(G_batch, ldb_batch, block_ranks, block_ranks, 0, 8, "R");

			check_error( kblas_ara_trsm_batch_template(
				handle, Y_batch, lda_batch, R_batch, ldb_batch, rows_batch, block_ranks, num_ops, max_rows, bs
			) );
		}

		// Count the number of vectors that have a small magnitude
		// also updates the rank, max diagonal and advances the Y_batch pointers
		check_error( kblas_ara_svec_count_batch_template(
			diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, lda_batch, 
			tol, r, small_vectors, relative, num_ops, kblas_stream
		) );
		
		// Advance the rank
		rank += samples;
	}

	// Finally, B = M' * A
	sampler.sample(
		A_batch, lda_batch, ranks_batch, 
		B_batch, ldb_batch, max_rank, num_ops, 1
	);

	// printDenseMatrixGPU(A_batch, lda_batch, rows_batch, ranks_batch, 0, 16, "A");
	// printDenseMatrixGPU(B_batch, ldb_batch, cols_batch, ranks_batch, 0, 16, "B");
	return KBLAS_Success;
}

void kblas_ara_batch_mod(kblasHandle_t handle, int* rows_batch, int* cols_batch, H2Opus_Real* U_batch, H2Opus_Real* V_batch, int* ranks, int* scan_ranks,
	H2Opus_Real** A_batch, int* lda_batch, H2Opus_Real** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops){

	    DenseSampler_mod<double> dense_sampler(U_batch, V_batch, ranks, scan_ranks, rows_batch, cols_batch, max_rows, max_cols, handle);

		kblas_ara_batch_template_mod<double, DenseSampler_mod<double> >(
			handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
			ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
		);
}