#ifndef FLASH_ATTENTION_V1_TILED_D_OPT_H
#define FLASH_ATTENTION_V1_TILED_D_OPT_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cassert>

using namespace nvcuda;

// Configuration constants
#ifndef BQ
#define BQ 16          // Query tile size (larger for better compute efficiency)
#endif

#ifndef BK
#define BK 16          // Key/Value tile size (matches BQ, enables Tensor Cores if used)
#endif

#ifndef D_TILE_QK
#define D_TILE_QK 32   // Head dimension tile size for Q@K^T (larger reduces d-loop iterations)
#endif

#ifndef D_TILE_V
#define D_TILE_V 32    // Head dimension tile size for S@V (matches D_TILE_QK for consistency)
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256  // Number of threads per block (good parallelism, low register pressure)
#endif

#ifndef D
#define D 128          // Head dimension (must match runtime value)
#endif

// Check if we can use Tensor Cores (WMMA)
// Requirements: BQ, BK >= 16 and divisible by 16, D_TILE >= 16
constexpr bool use_wmma_qk = (BQ >= 16) && (BK >= 16) && (D_TILE_QK >= 16) && 
                              (BQ % 16 == 0) && (BK % 16 == 0) && (D_TILE_QK % 16 == 0);
constexpr bool use_wmma_v = (BQ >= 16) && (BK >= 16) && (D_TILE_V >= 16) &&
                             (BQ % 16 == 0) && (BK % 16 == 0) && (D_TILE_V % 16 == 0);

// Precision configuration
#ifndef USE_FP64
#define USE_FP64 0      // Set to 1 for double precision, 0 for half precision
#endif

#if USE_FP64
#define DATA_TYPE double
#define FLOAT_TO_DATA(x) (x)
#define DATA_TO_FLOAT(x) (x)
#else
#define DATA_TYPE __half
#define FLOAT_TO_DATA(x) __float2half(x)
#define DATA_TO_FLOAT(x) __half2float(x)
#endif

// Helper function: 2D to 1D index conversion (row-major)
__device__ __host__ inline int idx2d(int i, int j, int cols) {
    return i * cols + j;
}

// Compute partial Q@K^T for one d-tile chunk using Tensor Cores
// Q_chunk: [bq, d_tile_qk], K_chunk: [bk, d_tile_qk]
// Accumulates into S: [bq, bk]
__device__ void mat_mul_chunk_accumulate(
    const DATA_TYPE* Q_chunk, const DATA_TYPE* K_chunk, DATA_TYPE* S,
    int bq, int bk, int d_tile_size
) {
    if constexpr (use_wmma_qk) {
        // Use Tensor Cores for 16x16x16 tiles
        // With BQ=16, BK=16, D_TILE=32, we have:
        // - 1 tile in M dimension (16/16 = 1)
        // - 1 tile in N dimension (16/16 = 1)
        // - 2 tiles in K dimension (32/16 = 2)
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        
        const int warp_id = threadIdx.x / 32;
        const int num_warps = blockDim.x / 32;
        
        // With 256 threads = 8 warps, and only 1×1=1 output tile, 
        // only the first warp does work
        if (warp_id == 0) {
            // Single 16x16 output tile
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
            
            // Load existing accumulator values from S
            wmma::load_matrix_sync(acc_frag, S, bk, wmma::mem_row_major);
            
            // Accumulate over K dimension
            for (int tile_k = 0; tile_k * WMMA_K < d_tile_size; tile_k++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                
                // Load Q chunk [16, 32] - take columns [tile_k*16 : (tile_k+1)*16]
                wmma::load_matrix_sync(a_frag, Q_chunk + tile_k * WMMA_K, d_tile_size);
                
                // Load K chunk [16, 32] - take columns [tile_k*16 : (tile_k+1)*16]
                // We want K^T, so we load K as col_major
                wmma::load_matrix_sync(b_frag, K_chunk + tile_k * WMMA_K, d_tile_size);
                
                // Compute: acc += A @ B^T
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            
            // Store result back to S [16, 16]
            wmma::store_matrix_sync(S, acc_frag, bk, wmma::mem_row_major);
        }
        __syncthreads();
    } else {
        // Fallback: standard implementation without Tensor Cores
        for (int idx = threadIdx.x; idx < bq * bk; idx += blockDim.x) {
            int i = idx / bk;
            int j = idx % bk;
            
            float sum = 0.0f;
            for (int kk = 0; kk < d_tile_size; kk++) {
                sum += DATA_TO_FLOAT(Q_chunk[idx2d(i, kk, d_tile_size)]) *
                       DATA_TO_FLOAT(K_chunk[idx2d(j, kk, d_tile_size)]);
            }
            S[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(S[idx]) + sum);
        }
        __syncthreads();
    }
}

// Fused: Subtract vector from each row and exponentiate
// out[i,j] = exp(S[i,j] - v[i])
__device__ void mat_sub_vec_exp(
    const DATA_TYPE* S, const float* v, DATA_TYPE* out,
    int m, int n
) {
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int i = idx / n;
        out[idx] = FLOAT_TO_DATA(expf(DATA_TO_FLOAT(S[idx]) - v[i]));
    }
}

// Fused: Row sum combined with multiply-add, updating l in place
// l[i] = l[i] * alpha[i] + sum_j S[i,j]
__device__ void row_sum_mul_add_inplace(
    const DATA_TYPE* S, float* l, const float* alpha,
    int m, int n
) {
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        float sum_val = 0.0f;
        for (int j = 0; j < n; j++) {
            sum_val += DATA_TO_FLOAT(S[idx2d(i, j, n)]);
        }
        l[i] = l[i] * alpha[i] + sum_val;
    }
}

// Compute S@V for one d-tile chunk and accumulate into register-based output
// Each thread updates its owned output elements
// S: [bq, bk], V_chunk: [bk, d_tile_v]
// O_tile_buf: temporary buffer to stage WMMA results (can reuse QK_chunk space)
//
// Design Note: We use shared memory staging (store_matrix_sync → O_tile_buf → thread reads)
// rather than direct fragment access because:
//   - Fragment element layout is architecture-specific and undocumented
//   - Direct access via fragment.x[i] is non-portable across GPU generations
//   - We reuse existing QK_chunk buffer, so no additional memory cost
//   - Shared memory round-trip is negligible compared to 4× Tensor Core speedup
__device__ void accumulate_output_chunk(
    float* O_reg, const DATA_TYPE* S, const DATA_TYPE* V_chunk,
    const float* alpha, int bq, int bk, int d_tile_size, int d_offset,
    int elems_per_thread, int thread_start_elem, DATA_TYPE* O_tile_buf
) {
    if constexpr (use_wmma_v) {
        // Verify that O_tile_buf is large enough to hold WMMA output [BQ, d_tile_size]
        // O_tile_buf should be the reused QK_chunk buffer with size (BQ+BK)*D_TILE_QK
        // Required: BQ * d_tile_size <= (BQ+BK) * D_TILE_QK
        static_assert((BQ + BK) * D_TILE_QK >= BQ * D_TILE_V, 
                     "QK_chunk buffer too small to stage WMMA S@V results");
        
        // Use Tensor Cores for S@V computation
        // With BQ=16, BK=16, D_TILE_V=32, we compute:
        // - 1 tile in M dimension (16/16 = 1)
        // - 2 tiles in N dimension (32/16 = 2)
        // - 1 tile in K dimension (16/16 = 1)
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        
        const int warp_id = threadIdx.x / 32;
        const int num_warps = blockDim.x / 32;
        
        // Compute how many 16-wide tiles we need in the N dimension
        int n_tiles = (d_tile_size + WMMA_N - 1) / WMMA_N;
        
        // Distribute N tiles across warps
        for (int tile_n = warp_id; tile_n < n_tiles; tile_n += num_warps) {
            if (tile_n * WMMA_N >= d_tile_size) continue;
            
            // Single M tile (BQ=16), single K tile (BK=16)
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
            wmma::fill_fragment(acc_frag, __float2half(0.0f));
            
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
            
            // Load S [16, 16] as matrix A
            wmma::load_matrix_sync(a_frag, S, bk);
            
            // Load V chunk [16, d_tile_size] - take columns [tile_n*16 : (tile_n+1)*16]
            wmma::load_matrix_sync(b_frag, V_chunk + tile_n * WMMA_N, d_tile_size);
            
            // Compute: O_tile = S @ V_chunk
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            
            // Store to temporary buffer [BQ, d_tile_size]
            // Use O_tile_buf which can be the reused QK_chunk space
            wmma::store_matrix_sync(O_tile_buf + tile_n * WMMA_N, acc_frag, 
                                   d_tile_size, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Now each thread reads from O_tile_buf and updates its registers
        for (int t = 0; t < elems_per_thread; t++) {
            int linear_idx = thread_start_elem + t;
            if (linear_idx >= bq * D) break;
            
            int row = linear_idx / D;
            int col = linear_idx % D;
            
            // Only process if this column is in the current d-tile
            if (col >= d_offset && col < d_offset + d_tile_size) {
                int col_local = col - d_offset;
                
                // Scale by alpha and add WMMA result
                O_reg[t] = O_reg[t] * alpha[row] + 
                          __half2float(O_tile_buf[idx2d(row, col_local, d_tile_size)]);
            }
        }
    } else {
        // Fallback: standard implementation without Tensor Cores
        for (int t = 0; t < elems_per_thread; t++) {
            int linear_idx = thread_start_elem + t;
            if (linear_idx >= bq * D) break;
            
            int row = linear_idx / D;
            int col = linear_idx % D;
            
            // Only process if this column is in the current d-tile
            if (col >= d_offset && col < d_offset + d_tile_size) {
                int col_local = col - d_offset;
                
                // Scale by alpha
                O_reg[t] *= alpha[row];
                
                // Accumulate S @ V_chunk contribution
                float sum = 0.0f;
                for (int k = 0; k < bk; k++) {
                    sum += DATA_TO_FLOAT(S[idx2d(row, k, bk)]) * 
                           DATA_TO_FLOAT(V_chunk[idx2d(k, col_local, d_tile_size)]);
                }
                O_reg[t] += sum;
            }
        }
    }
}

// Core tile processing with TRUE d-tiling (loads chunks from global memory)
__device__ void process_kv_tile(
    const DATA_TYPE* Q_global, const DATA_TYPE* K_global, const DATA_TYPE* V_global,
    int q_start, int k_start, int bq, int bk, int d,
    float* m, float* l, float* O_reg,
    DATA_TYPE* QK_chunk, DATA_TYPE* V_chunk, DATA_TYPE* S, float* alpha,
    int d_tile_qk, int d_tile_v, int base_offset,
    int elems_per_thread, int thread_start_elem
) {
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    // Initialize S to zero
    for (int idx = threadIdx.x; idx < bq * bk; idx += blockDim.x) {
        S[idx] = FLOAT_TO_DATA(0.0f);
    }
    __syncthreads();
    
    // 1) Compute S = Q @ K^T with d-tiling (load chunks from global memory)
    for (int d_start = 0; d_start < d; d_start += d_tile_qk) {
        int d_end = min(d_start + d_tile_qk, d);
        int d_size = d_end - d_start;
        
        // Load Q chunk [bq, d_tile_qk]
        for (int idx = threadIdx.x; idx < bq * d_size; idx += blockDim.x) {
            int i = idx / d_size;
            int j = idx % d_size;
            QK_chunk[idx2d(i, j, d_tile_qk)] = 
                Q_global[base_offset + idx2d(q_start + i, d_start + j, d)];
        }
        
        // Load K chunk [bk, d_tile_qk] - reuse space after Q
        DATA_TYPE* K_chunk = QK_chunk + BQ * D_TILE_QK;
        for (int idx = threadIdx.x; idx < bk * d_size; idx += blockDim.x) {
            int i = idx / d_size;
            int j = idx % d_size;
            K_chunk[idx2d(i, j, d_tile_qk)] = 
                K_global[base_offset + idx2d(k_start + i, d_start + j, d)];
        }
        __syncthreads();
        
        // Accumulate partial Q@K^T
        mat_mul_chunk_accumulate(QK_chunk, K_chunk, S, bq, bk, d_size);
    }
    
    // Apply scaling to S
    for (int idx = threadIdx.x; idx < bq * bk; idx += blockDim.x) {
        S[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(S[idx]) * inv_sqrt_d);
    }
    __syncthreads();
    
    // 2) Compute rescale factor and update m
    for (int i = threadIdx.x; i < bq; i += blockDim.x) {
        float new_max = m[i];
        for (int j = 0; j < bk; j++) {
            float s_val = DATA_TO_FLOAT(S[idx2d(i, j, bk)]);
            if (s_val > new_max) {
                new_max = s_val;
            }
        }
        alpha[i] = expf(m[i] - new_max);
        m[i] = new_max;
    }
    __syncthreads();
    
    // 3) Shift scores and exponentiate
    mat_sub_vec_exp(S, m, S, bq, bk);
    __syncthreads();
    
    // 4) Update running denominator
    row_sum_mul_add_inplace(S, l, alpha, bq, bk);
    __syncthreads();
    
    // 5) Compute O += S @ V with d-tiling on V (accumulate into registers)
    for (int d_start = 0; d_start < d; d_start += d_tile_v) {
        int d_end = min(d_start + d_tile_v, d);
        int d_size = d_end - d_start;
        
        // Load V chunk [bk, d_tile_v]
        for (int idx = threadIdx.x; idx < bk * d_size; idx += blockDim.x) {
            int i = idx / d_size;
            int j = idx % d_size;
            V_chunk[idx2d(i, j, d_tile_v)] = 
                V_global[base_offset + idx2d(k_start + i, d_start + j, d)];
        }
        __syncthreads();
        
        // Accumulate into register-based output
        // Reuse QK_chunk as temporary buffer for WMMA results (no longer needed here)
        // QK_chunk size: (BQ+BK)*D_TILE_QK elements
        // Required size: BQ*d_size elements (at most BQ*D_TILE_V)
        assert((BQ + BK) * D_TILE_QK >= BQ * d_size && 
               "QK_chunk buffer insufficient for WMMA output staging");
        accumulate_output_chunk(O_reg, S, V_chunk, alpha, bq, bk, d_size, d_start,
                               elems_per_thread, thread_start_elem, QK_chunk);
        __syncthreads();
    }
}

// Main flash attention kernel
__global__ void flash_attention_kernel_opt(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* O,
    int B, int H, int L, int d_runtime, int d_tile_qk_runtime, int d_tile_v_runtime
) {
    // Verify runtime parameters match compile-time constants
    assert(d_runtime == D && "Runtime d must match compile-time D constant");
    
    // blockIdx.y encodes (batch, head) pair
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / H;
    const int head_idx = bh_idx % H;
    
    // Each block processes one query tile for one (batch, head) pair
    int q_start = blockIdx.x * BQ;
    if (q_start >= L) return;
    
    int q_end = min(q_start + BQ, L);
    int q_len = q_end - q_start;
    
    // Compute base offset for this (batch, head) pair: [B, H, L, d]
    const int base_offset = (batch_idx * H * L * d_runtime) + (head_idx * L * d_runtime);
    
    // Shared memory allocation - TRUE d-tiling (only tile-sized chunks)
    extern __shared__ char shared_mem_raw[];
    DATA_TYPE* shared_mem_data = reinterpret_cast<DATA_TYPE*>(shared_mem_raw);
    
    // QK chunks: space for Q and K d-tiles
    DATA_TYPE* QK_chunk = shared_mem_data;                        // [BQ * D_TILE_QK + BK * D_TILE_QK]
    // V chunk: space for V d-tile
    DATA_TYPE* V_chunk = QK_chunk + (BQ + BK) * D_TILE_QK;       // [BK * D_TILE_V]
    // S: attention scores
    DATA_TYPE* S = V_chunk + BK * D_TILE_V;                      // [BQ * BK]
    
    float* float_buf = reinterpret_cast<float*>(S + BQ * BK);
    float* m = float_buf;                                         // [BQ]
    float* l = m + BQ;                                            // [BQ]
    float* alpha = l + BQ;                                        // [BQ]
    
    // Register-based O_acc: each thread owns elems_per_thread output elements
    constexpr int total_output = BQ * D;
    constexpr int elems_per_thread = (total_output + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    float O_reg[elems_per_thread];
    int thread_start_elem = threadIdx.x * elems_per_thread;
    
    // Initialize streaming state
    for (int i = threadIdx.x; i < BQ; i += blockDim.x) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    
    // Initialize register-based output
    for (int t = 0; t < elems_per_thread; t++) {
        O_reg[t] = 0.0f;
    }
    __syncthreads();
    
    // Loop over K/V tiles
    for (int k_start = 0; k_start < L; k_start += BK) {
        int k_end = min(k_start + BK, L);
        int k_len = k_end - k_start;
        
        // Process this KV tile with TRUE d-tiling
        process_kv_tile(Q, K, V, q_start, k_start, q_len, k_len, d_runtime,
                       m, l, O_reg, QK_chunk, V_chunk, S, alpha,
                       d_tile_qk_runtime, d_tile_v_runtime, base_offset,
                       elems_per_thread, thread_start_elem);
    }
    
    // Finalize and store to output from registers
    for (int t = 0; t < elems_per_thread; t++) {
        int linear_idx = thread_start_elem + t;
        if (linear_idx < q_len * D) {
            int row = linear_idx / D;
            int col = linear_idx % D;
            O[base_offset + idx2d(q_start + row, col, D)] = 
                FLOAT_TO_DATA(O_reg[t] / l[row]);
        }
    }
}

// Host function to launch flash attention
void flash_attention_v1_opt(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* O,
    int B, int H, int L, int d_runtime, int d_tile_qk_runtime, int d_tile_v_runtime
) {
    // Safety checks
    static_assert((BQ & (BQ - 1)) == 0, "BQ must be a power of 2");
    static_assert((BK & (BK - 1)) == 0, "BK must be a power of 2");
    static_assert((THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0, "THREADS_PER_BLOCK must be a power of 2");
    static_assert(BQ > 0 && BK > 0 && THREADS_PER_BLOCK > 0, "Constants must be positive");
    
    // Runtime checks
    assert(B > 0 && H > 0 && L > 0 && d_runtime > 0 && "All dimensions must be positive");
    assert(d_runtime == D && "Runtime d must match compile-time D constant");
    assert(d_tile_qk_runtime > 0 && d_tile_qk_runtime <= d_runtime && "d_tile_qk must be valid");
    assert(d_tile_v_runtime > 0 && d_tile_v_runtime <= d_runtime && "d_tile_v must be valid");
    
    // Calculate shared memory size with TRUE d-tiling
    size_t shared_mem_size = (
        (BQ + BK) * D_TILE_QK +  // QK_chunk (Q and K d-tiles)
        BK * D_TILE_V +          // V_chunk
        BQ * BK                  // S
    ) * sizeof(DATA_TYPE) + (
        BQ +          // m
        BQ +          // l
        BQ            // alpha
    ) * sizeof(float);
    
    // Check shared memory limit
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    assert(shared_mem_size <= prop.sharedMemPerBlock && 
           "Shared memory requirement exceeds device limit");
    
    // Launch kernel with 2D grid: x = query tiles, y = batch × heads
    dim3 grid_dim((L + BQ - 1) / BQ, B * H);
    
    flash_attention_kernel_opt<<<grid_dim, THREADS_PER_BLOCK, shared_mem_size>>>(
        Q, K, V, O, B, H, L, d_runtime, d_tile_qk_runtime, d_tile_v_runtime
    );
    
    cudaDeviceSynchronize();
}

#endif // FLASH_ATTENTION_V1_TILED_D_OPT_H
