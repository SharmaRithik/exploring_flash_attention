#ifndef FLASH_ATTENTION_V2_H
#define FLASH_ATTENTION_V2_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cassert>

using namespace nvcuda;

// Configuration constants
// 
// Performance vs OpenMP-parallelized C++ baseline (d=128):
//
// WITH TENSOR CORES (this _opt version):
// Large batch (B=32, H=8, L=1024): ~135× speedup (2.9× faster than non-TC version)
//   - Forward kernel: 52.8 ms (262K blocks, 1 GB workspace)
//   - Tensor cores accelerate Q@K^T and S@V matrix multiplications
//
// WITHOUT TENSOR CORES (regular v2.h):
// Large batch (B=32, H=8):
//   L=1024: ~47× speedup (262K forward blocks, 1 GB workspace)
//   - Workspace memory overhead and reduction kernel become bottlenecks
//
// Small batch (B=2, H=2):
//   L=1024: ~219× speedup (4K forward blocks, 4 MB workspace)
//   L=512:  ~144× speedup
//   L=256:  ~101× speedup
//   L=128:  ~83× speedup
//   - Smaller batch sizes scale better due to less reduction overhead
//
// Default config (balanced): BQ=16, BK=16, D_TILE=32, shared memory: 3.5 KB
// Low-memory config: BQ=8, BK=8, D_TILE=16, shared memory: 992 bytes
//
// Tensor Core requirements: BQ, BK >= 16 and divisible by 16, D_TILE >= 16
//
// Key insight: GPU advantage scales with sequence length and inversely with batch size.
// V2's split-KV parallelization creates 16× more blocks than V1 (with KV_TILES_PER_BLOCK=4).
// Tensor Cores provide ~2.9× additional speedup for matrix multiplications.
//
#ifndef BQ
#define BQ 16          // Query tile size (balanced config)
#endif

#ifndef BK
#define BK 16          // Key/Value tile size (balanced config)
#endif

#ifndef D_TILE_QK
#define D_TILE_QK 32   // Head dimension tile size for Q@K^T (balanced config)
#endif

#ifndef D_TILE_V
#define D_TILE_V 32    // Head dimension tile size for S@V (balanced config)
#endif

#ifndef KV_TILES_PER_BLOCK
#define KV_TILES_PER_BLOCK 64  // V2 parallelism parameter (higher = fewer blocks, less overhead)
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256  // Can try 128 or 512 depending on tile sizes
#endif

#ifndef D
#define D 128          // Head dimension
#endif

// Check if we can use Tensor Cores (WMMA)
// Requirements: BQ, BK >= 16 and divisible by 16, D_TILE >= 16
constexpr bool use_wmma_qk = (BQ >= 16) && (BK >= 16) && (D_TILE_QK >= 16) && 
                              (BQ % 16 == 0) && (BK % 16 == 0) && (D_TILE_QK % 16 == 0);
constexpr bool use_wmma_v = (BQ >= 16) && (BK >= 16) && (D_TILE_V >= 16) &&
                             (BQ % 16 == 0) && (BK % 16 == 0) && (D_TILE_V % 16 == 0);

// Precision configuration
#ifndef USE_FP64
#define USE_FP64 0
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
__device__ void mat_mul_qk_accumulate(
    const DATA_TYPE* Q_chunk, const DATA_TYPE* K_chunk, DATA_TYPE* S,
    int bq, int bk, int d_tile_size
) {
    if constexpr (use_wmma_qk) {
        // Use Tensor Cores for 16x16x16 tiles
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;
        
        const int warp_id = threadIdx.x / 32;
        
        // With BQ=16, BK=16, only 1×1=1 output tile
        // Only the first warp does work
        if (warp_id == 0) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
            
            // Load existing accumulator values from S
            wmma::load_matrix_sync(acc_frag, S, bk, wmma::mem_row_major);
            
            // Accumulate over K dimension (d_tile_qk / 16 iterations)
            for (int tile_k = 0; tile_k * WMMA_K < d_tile_size; tile_k++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                
                // Load Q chunk [16, d_tile_qk] - take columns [tile_k*16 : (tile_k+1)*16]
                wmma::load_matrix_sync(a_frag, Q_chunk + tile_k * WMMA_K, d_tile_size);
                
                // Load K chunk [16, d_tile_qk] - take columns [tile_k*16 : (tile_k+1)*16]
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
            for (int k = 0; k < d_tile_size; k++) {
                sum += DATA_TO_FLOAT(Q_chunk[idx2d(i, k, d_tile_size)]) *
                       DATA_TO_FLOAT(K_chunk[idx2d(j, k, d_tile_size)]);
            }
            S[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(S[idx]) + sum);
        }
        __syncthreads();
    }
}

// Compute S@V for one d-tile chunk and accumulate into register-based output
// Each thread updates its owned output elements
// S: [bq, bk], V_chunk: [bk, d_tile_v]
// O_tile_buf: temporary buffer to stage WMMA results (can reuse QK_chunk space)
__device__ void accumulate_output_sv(
    float* O_reg, const DATA_TYPE* S, const DATA_TYPE* V_chunk,
    int bq, int bk, int d, int d_tile_size, int d_offset,
    int elems_per_thread, int thread_start_elem, DATA_TYPE* O_tile_buf
) {
    if constexpr (use_wmma_v) {
        // Verify that O_tile_buf is large enough to hold WMMA output [BQ, d_tile_size]
        static_assert((BQ + BK) * D_TILE_QK >= BQ * D_TILE_V, 
                     "QK_chunk buffer too small to stage WMMA S@V results");
        
        // Use Tensor Cores for S@V computation
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
            wmma::store_matrix_sync(O_tile_buf + tile_n * WMMA_N, acc_frag, 
                                   d_tile_size, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Now each thread reads from O_tile_buf and updates its registers
        for (int t = 0; t < elems_per_thread; t++) {
            int linear_idx = thread_start_elem + t;
            if (linear_idx >= bq * d) break;
            
            int row = linear_idx / d;
            int col = linear_idx % d;
            
            // Only process if this column is in the current d-tile
            if (col >= d_offset && col < d_offset + d_tile_size) {
                int col_local = col - d_offset;
                
                // Add WMMA result
                O_reg[t] += __half2float(O_tile_buf[idx2d(row, col_local, d_tile_size)]);
            }
        }
    } else {
        // Fallback: standard implementation without Tensor Cores
        for (int t = 0; t < elems_per_thread; t++) {
            int linear_idx = thread_start_elem + t;
            if (linear_idx >= bq * d) break;
            
            int row = linear_idx / d;
            int col = linear_idx % d;
            
            // Only process if this column is in the current d-tile
            if (col >= d_offset && col < d_offset + d_tile_size) {
                int col_local = col - d_offset;
                
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

// D-tiled process_kv_tile with register-based O_acc (like V1)
// Each thread owns a fixed subset of output elements in registers
__device__ void process_kv_tile_dtiled(
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
        
        // Load K chunk [bk, d_tile_qk]
        DATA_TYPE* K_chunk = QK_chunk + BQ * D_TILE_QK;
        for (int idx = threadIdx.x; idx < bk * d_size; idx += blockDim.x) {
            int i = idx / d_size;
            int j = idx % d_size;
            K_chunk[idx2d(i, j, d_tile_qk)] = 
                K_global[base_offset + idx2d(k_start + i, d_start + j, d)];
        }
        __syncthreads();
        
        // Accumulate partial Q@K^T
        mat_mul_qk_accumulate(QK_chunk, K_chunk, S, bq, bk, d_size);
    }
    
    // Scale by 1/sqrt(d)
    for (int idx = threadIdx.x; idx < bq * bk; idx += blockDim.x) {
        S[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(S[idx]) * inv_sqrt_d);
    }
    __syncthreads();
    
    // 2) Compute rescale factor BEFORE updating m
    for (int i = threadIdx.x; i < bq; i += blockDim.x) {
        float new_max = m[i];
        for (int j = 0; j < bk; j++) {
            if (DATA_TO_FLOAT(S[idx2d(i, j, bk)]) > new_max) {
                new_max = DATA_TO_FLOAT(S[idx2d(i, j, bk)]);
            }
        }
        alpha[i] = expf(m[i] - new_max);
        m[i] = new_max;
    }
    __syncthreads();
    
    // 3) Shift scores and exponentiate
    for (int idx = threadIdx.x; idx < bq * bk; idx += blockDim.x) {
        int i = idx / bk;
        S[idx] = FLOAT_TO_DATA(expf(DATA_TO_FLOAT(S[idx]) - m[i]));
    }
    __syncthreads();
    
    // 4) Update running denominator
    for (int i = threadIdx.x; i < bq; i += blockDim.x) {
        float sum_val = 0.0f;
        for (int j = 0; j < bk; j++) {
            sum_val += DATA_TO_FLOAT(S[idx2d(i, j, bk)]);
        }
        l[i] = l[i] * alpha[i] + sum_val;
    }
    __syncthreads();
    
    // 5) Update O_reg with d-tiling for S @ V
    // Each thread only updates its owned elements
    
    // First, scale all O_reg elements by alpha (once before processing V d-tiles)
    for (int t = 0; t < elems_per_thread; t++) {
        int linear_idx = thread_start_elem + t;
        if (linear_idx >= bq * d) break;
        
        int row = linear_idx / d;
        O_reg[t] *= alpha[row];
    }
    
    // Then accumulate S @ V with d-tiling
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
        // Reuse QK_chunk as temporary buffer for WMMA results
        accumulate_output_sv(O_reg, S, V_chunk, bq, bk, d, d_size, d_start,
                            elems_per_thread, thread_start_elem, QK_chunk);
        __syncthreads();
    }
}

// KERNEL 1: Partial Attention with TRUE d-tiling and register-based O_acc
__global__ void partial_attention_kernel(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* workspace_O, float* workspace_m, float* workspace_l,
    int B, int H, int L, int d,
    int kv_tiles_per_block, int num_q_tiles, int num_kv_tiles,
    int d_tile_qk, int d_tile_v
) {
    // Grid: (num_q_tiles, num_kv_blocks, batch×heads)
    const int q_tile_idx = blockIdx.x;
    const int kv_block_idx = blockIdx.y;
    const int bh_idx = blockIdx.z;
    
    const int batch_idx = bh_idx / H;
    const int head_idx = bh_idx % H;
    
    // Compute query range
    int q_start = q_tile_idx * BQ;
    if (q_start >= L) return;
    int q_end = min(q_start + BQ, L);
    int q_len = q_end - q_start;
    
    // Compute KV range for this block
    int kv_block_start = kv_block_idx * kv_tiles_per_block;
    int kv_block_end = min(kv_block_start + kv_tiles_per_block, num_kv_tiles);
    
    // Compute base offset for this (batch, head) pair
    const int base_offset = (batch_idx * H * L * d) + (head_idx * L * d);
    
    // Shared memory layout with TRUE d-tiling (like V1)
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
    // Note: O_acc now in REGISTERS, not shared memory!
    
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
    
    // Process assigned KV tiles with TRUE d-tiling
    for (int kv_tile_idx = kv_block_start; kv_tile_idx < kv_block_end; kv_tile_idx++) {
        int k_start = kv_tile_idx * BK;
        if (k_start >= L) break;
        int k_end = min(k_start + BK, L);
        int k_len = k_end - k_start;
        
        // Process this KV tile with d-tiling (loads chunks from global memory)
        process_kv_tile_dtiled(Q, K, V, q_start, k_start, q_len, k_len, d,
                              m, l, O_reg, QK_chunk, V_chunk, S, alpha,
                              d_tile_qk, d_tile_v, base_offset,
                              elems_per_thread, thread_start_elem);
    }
    
    // Write partial results to workspace from registers
    const int num_kv_blocks = (num_kv_tiles + kv_tiles_per_block - 1) / kv_tiles_per_block;
    const int workspace_idx = (bh_idx * num_q_tiles * num_kv_blocks) + 
                             (q_tile_idx * num_kv_blocks) + kv_block_idx;
    const int workspace_O_offset = workspace_idx * BQ * D;
    const int workspace_ml_offset = workspace_idx * BQ;
    
    // Each thread writes its owned elements from registers
    for (int t = 0; t < elems_per_thread; t++) {
        int linear_idx = thread_start_elem + t;
        if (linear_idx < q_len * d) {
            int i = linear_idx / d;
            int j = linear_idx % d;
            workspace_O[workspace_O_offset + idx2d(i, j, d)] = FLOAT_TO_DATA(O_reg[t]);
        }
    }
    
    for (int i = threadIdx.x; i < q_len; i += blockDim.x) {
        workspace_m[workspace_ml_offset + i] = m[i];
        workspace_l[workspace_ml_offset + i] = l[i];
    }
}

// KERNEL 2: Reduction (parallelized over output elements)
// 
// Current implementation: Simple parallelization over rows and output elements
// Uses shared memory for m_global, l_global, and scales
//
// Potential optimizations (compared to official Dao-AILab implementation):
// 1. Use 128 threads instead of 256 for better occupancy
// 2. Add warp-level reductions (warp shuffle) for finding max/sum instead of simple loops
// 3. Vectorized memory access using GmemTiledCopy for coalesced loads/stores
// 4. Bank conflict avoidance: use stride kBlockM+1 for shared memory
// 5. Transpose LSE data in shared memory for better coalescing
// 6. Each thread processes multiple LSE values (kNLsePerThread) for better work distribution
//
__global__ void reduction_kernel(
    const DATA_TYPE* workspace_O, const float* workspace_m, const float* workspace_l,
    DATA_TYPE* O,
    int B, int H, int L, int d,
    int kv_tiles_per_block, int num_q_tiles, int num_kv_tiles
) {
    // Grid: (num_q_tiles, batch×heads)
    const int q_tile_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    
    const int batch_idx = bh_idx / H;
    const int head_idx = bh_idx % H;
    
    // Compute query range
    int q_start = q_tile_idx * BQ;
    if (q_start >= L) return;
    int q_end = min(q_start + BQ, L);
    int q_len = q_end - q_start;
    
    const int num_kv_blocks = (num_kv_tiles + kv_tiles_per_block - 1) / kv_tiles_per_block;
    
    // Compute base offset for output
    const int base_offset = (batch_idx * H * L * d) + (head_idx * L * d);
    
    // Use shared memory for m_global, l_global, and scales
    extern __shared__ float shared_buf[];
    float* m_global = shared_buf;                           // [BQ]
    float* l_global = m_global + BQ;                       // [BQ]
    float* scales = l_global + BQ;                         // [num_kv_blocks * BQ]
    
    // Step 1: Each thread computes m_global for some rows
    for (int i = threadIdx.x; i < q_len; i += blockDim.x) {
        float m_max = -INFINITY;
        for (int kb = 0; kb < num_kv_blocks; kb++) {
            int workspace_idx = (bh_idx * num_q_tiles * num_kv_blocks) + 
                               (q_tile_idx * num_kv_blocks) + kb;
            int workspace_ml_offset = workspace_idx * BQ;
            float m_partial = workspace_m[workspace_ml_offset + i];
            if (m_partial > m_max) {
                m_max = m_partial;
            }
        }
        m_global[i] = m_max;
    }
    __syncthreads();
    
    // Step 2: Each thread computes scales and l_global for some rows
    for (int i = threadIdx.x; i < q_len; i += blockDim.x) {
        float l_sum = 0.0f;
        for (int kb = 0; kb < num_kv_blocks; kb++) {
            int workspace_idx = (bh_idx * num_q_tiles * num_kv_blocks) + 
                               (q_tile_idx * num_kv_blocks) + kb;
            int workspace_ml_offset = workspace_idx * BQ;
            float m_partial = workspace_m[workspace_ml_offset + i];
            float l_partial = workspace_l[workspace_ml_offset + i];
            float scale = expf(m_partial - m_global[i]);
            scales[kb * BQ + i] = scale;
            l_sum += l_partial * scale;
        }
        l_global[i] = l_sum;
    }
    __syncthreads();
    
    // Step 3: Each thread computes some output elements
    for (int idx = threadIdx.x; idx < q_len * d; idx += blockDim.x) {
        int i = idx / d;
        int j = idx % d;
        
        float numerator = 0.0f;
        for (int kb = 0; kb < num_kv_blocks; kb++) {
            int workspace_idx = (bh_idx * num_q_tiles * num_kv_blocks) + 
                               (q_tile_idx * num_kv_blocks) + kb;
            int workspace_O_offset = workspace_idx * BQ * D;
            float O_partial = DATA_TO_FLOAT(workspace_O[workspace_O_offset + idx2d(i, j, d)]);
            numerator += O_partial * scales[kb * BQ + i];
        }
        
        O[base_offset + idx2d(q_start + i, j, d)] = FLOAT_TO_DATA(numerator / l_global[i]);
    }
}

// Host function to launch flash attention V2
void flash_attention_v2(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* O,
    int B, int H, int L, int d, int d_tile_qk, int d_tile_v,
    int kv_tiles_per_block
) {
    // Runtime checks
    assert(B > 0 && H > 0 && L > 0 && d > 0 && "All dimensions must be positive");
    assert(d == D && "Runtime d must match compile-time D constant");
    assert(kv_tiles_per_block > 0 && "kv_tiles_per_block must be positive");
    
    const int num_q_tiles = (L + BQ - 1) / BQ;
    const int num_kv_tiles = (L + BK - 1) / BK;
    const int num_kv_blocks = (num_kv_tiles + kv_tiles_per_block - 1) / kv_tiles_per_block;
    
    // Allocate workspace memory
    const size_t workspace_O_size = B * H * num_q_tiles * num_kv_blocks * BQ * D * sizeof(DATA_TYPE);
    const size_t workspace_ml_size = B * H * num_q_tiles * num_kv_blocks * BQ * sizeof(float);
    
    DATA_TYPE* workspace_O;
    float* workspace_m;
    float* workspace_l;
    
    cudaMalloc(&workspace_O, workspace_O_size);
    cudaMalloc(&workspace_m, workspace_ml_size);
    cudaMalloc(&workspace_l, workspace_ml_size);
    
    // PHASE 1: Launch forward kernels with TRUE d-tiling and register-based O_acc
    dim3 forward_grid(num_q_tiles, num_kv_blocks, B * H);
    
    // Calculate shared memory for forward kernel (d-tiled like V1, O_acc in registers)
    size_t forward_shared_mem = (
        (BQ + BK) * D_TILE_QK +      // QK_chunk (Q and K d-tiles)
        BK * D_TILE_V +              // V_chunk (V d-tile)
        BQ * BK                      // S (attention scores)
    ) * sizeof(DATA_TYPE) + (
        BQ +                         // m
        BQ +                         // l
        BQ                           // alpha
        // O_acc is now in registers, not shared memory!
    ) * sizeof(float);
    
    partial_attention_kernel<<<forward_grid, THREADS_PER_BLOCK, forward_shared_mem>>>(
        Q, K, V, workspace_O, workspace_m, workspace_l,
        B, H, L, d, kv_tiles_per_block, num_q_tiles, num_kv_tiles,
        d_tile_qk, d_tile_v
    );
    
    cudaDeviceSynchronize();
    
    // PHASE 2: Launch reduction kernels (parallelized)
    dim3 reduction_grid(num_q_tiles, B * H);
    
    // Calculate shared memory for reduction kernel
    size_t reduction_shared_mem = (
        BQ +                         // m_global
        BQ +                         // l_global
        num_kv_blocks * BQ          // scales
    ) * sizeof(float);
    
    reduction_kernel<<<reduction_grid, THREADS_PER_BLOCK, reduction_shared_mem>>>(
        workspace_O, workspace_m, workspace_l, O,
        B, H, L, d, kv_tiles_per_block, num_q_tiles, num_kv_tiles
    );
    
    cudaDeviceSynchronize();
    
    // Free workspace memory
    cudaFree(workspace_O);
    cudaFree(workspace_m);
    cudaFree(workspace_l);
}

#endif // FLASH_ATTENTION_V2_H
