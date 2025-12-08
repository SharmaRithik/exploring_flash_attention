#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Use common standard.h
#include "../../common/standard.h"

// Include V2 kernel
#ifdef USE_OPT_VERSION
#include "flash_attention_v2_opt.h"
#else
#include "flash_attention_v2.h"
#endif

// Utility: Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Helper to convert DATA_TYPE to float for display/comparison
inline float data_to_float(const DATA_TYPE& val) {
    return DATA_TO_FLOAT(val);
}

inline DATA_TYPE float_to_data(float val) {
    return FLOAT_TO_DATA(val);
}

// Initialize random data
void initialize_random(DATA_TYPE* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = float_to_data(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
}

// Compare two arrays and compute errors
void compare_arrays(const DATA_TYPE* a, const DATA_TYPE* b, int size, 
                    float& max_abs_diff, float& max_rel_diff, float& max_rel_diff_all) {
    max_abs_diff = 0.0f;
    max_rel_diff = 0.0f;
    max_rel_diff_all = 0.0f;
#if USE_FP64
    const float eps = 1e-10f;
#else
    const float eps = 1e-3f;
#endif
    
    for (int i = 0; i < size; i++) {
        float val_a = data_to_float(a[i]);
        float val_b = data_to_float(b[i]);
        float abs_diff = fabsf(val_a - val_b);
        
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        
        // Relative error for all values
        float denom_all = fabsf(val_b) + 1e-8f;
        float rel_diff_all = abs_diff / denom_all;
        if (rel_diff_all > max_rel_diff_all) {
            max_rel_diff_all = rel_diff_all;
        }
        
        // Relative error only for significant values
        if (fabsf(val_b) > eps) {
            float rel_diff = abs_diff / fabsf(val_b);
            if (rel_diff > max_rel_diff) {
                max_rel_diff = rel_diff;
            }
        }
    }
}

int main(int argc, char** argv) {
    srand(42);  // Fixed seed for reproducibility
    
    // Problem dimensions
    const int B = 32;      // Batch size
    const int H = 8;       // Number of heads
    const int L = 1024;    // Sequence length
    const int d = D;       // Head dimension (from compile-time constant)
    
    std::cout << "=== Flash Attention V2 CUDA Implementation ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size (B): " << B << std::endl;
    std::cout << "  Num heads (H): " << H << std::endl;
    std::cout << "  Sequence length (L): " << L << std::endl;
    std::cout << "  Head dimension (d): " << d << std::endl;
    std::cout << "  BQ: " << BQ << std::endl;
    std::cout << "  BK: " << BK << std::endl;
    std::cout << "  D_TILE_QK: " << D_TILE_QK << std::endl;
    std::cout << "  D_TILE_V: " << D_TILE_V << std::endl;
    std::cout << "  KV_TILES_PER_BLOCK: " << KV_TILES_PER_BLOCK << std::endl;
    std::cout << "  THREADS_PER_BLOCK: " << THREADS_PER_BLOCK << std::endl;
    
    // Calculate grid dimensions
    const int num_q_tiles = (L + BQ - 1) / BQ;
    const int num_kv_tiles = (L + BK - 1) / BK;
    const int num_kv_blocks = (num_kv_tiles + KV_TILES_PER_BLOCK - 1) / KV_TILES_PER_BLOCK;
    
    std::cout << "\nGrid dimensions:" << std::endl;
    std::cout << "  Forward kernel:  (" << num_q_tiles << " q_tiles, " 
              << num_kv_blocks << " kv_blocks, " << B*H << " batch×heads) = " 
              << num_q_tiles * num_kv_blocks * B * H << " blocks" << std::endl;
    std::cout << "  Reduction kernel: (" << num_q_tiles << " q_tiles, " 
              << B*H << " batch×heads) = " << num_q_tiles * B * H << " blocks" << std::endl;
    
    // Calculate workspace memory
    const size_t workspace_entries = B * H * num_q_tiles * num_kv_blocks;
    const size_t workspace_size_per_entry = BQ * d * sizeof(DATA_TYPE) + 2 * BQ * sizeof(float);
    const size_t total_workspace_mb = (workspace_entries * workspace_size_per_entry) / (1024 * 1024);
    
    std::cout << "\nWorkspace memory:" << std::endl;
    std::cout << "  Total entries: " << workspace_entries << std::endl;
    std::cout << "  Size per entry: " << workspace_size_per_entry << " bytes" << std::endl;
    std::cout << "  Total workspace: " << total_workspace_mb << " MB" << std::endl;
    
    const int total_size = B * H * L * d;
    
    // Allocate host memory
    DATA_TYPE* h_Q = new DATA_TYPE[total_size];
    DATA_TYPE* h_K = new DATA_TYPE[total_size];
    DATA_TYPE* h_V = new DATA_TYPE[total_size];
    DATA_TYPE* h_O_cuda = new DATA_TYPE[total_size];
    DATA_TYPE* h_O_reference = new DATA_TYPE[total_size];
    
    // Initialize with random data
    initialize_random(h_Q, total_size);
    initialize_random(h_K, total_size);
    initialize_random(h_V, total_size);
    
    // Allocate device memory
    DATA_TYPE *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_K, total_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_V, total_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_O, total_size * sizeof(DATA_TYPE)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, total_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, total_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, total_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    
    // Warmup run
    std::cout << "\n--- Warmup Run ---" << std::endl;
    flash_attention_v2(d_Q, d_K, d_V, d_O, B, H, L, d, D_TILE_QK, D_TILE_V, KV_TILES_PER_BLOCK);
    CUDA_CHECK(cudaGetLastError());
    
    // Benchmark
    std::cout << "\n--- Benchmark (10 iterations) ---" << std::endl;
    const int num_iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        flash_attention_v2(d_Q, d_K, d_V, d_O, B, H, L, d, D_TILE_QK, D_TILE_V, KV_TILES_PER_BLOCK);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / num_iterations;
    
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_O_cuda, d_O, total_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    
    // Compute reference on CPU
    std::cout << "\n--- Computing CPU Reference ---" << std::endl;
    
    // Warmup CPU (let turbo boost stabilize and threads warm up)
    std::cout << "CPU warmup..." << std::endl;
    standard_attention_cpu(h_Q, h_K, h_V, h_O_reference, B, H, L, d);
    
    // Benchmark CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    standard_attention_cpu(h_Q, h_K, h_V, h_O_reference, B, H, L, d);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_elapsed = cpu_end - cpu_start;
    double avg_cpu_time = cpu_elapsed.count();
    
    std::cout << "CPU time: " << avg_cpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << (avg_cpu_time / avg_time) << "x" << std::endl;
    
    // Compare results
    std::cout << "\n--- Accuracy Check ---" << std::endl;
    float max_abs_diff, max_rel_diff, max_rel_diff_all;
    compare_arrays(h_O_cuda, h_O_reference, total_size, max_abs_diff, max_rel_diff, max_rel_diff_all);
    
    std::cout << "Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "Max relative difference (significant values): " << max_rel_diff << std::endl;
    std::cout << "Max relative difference (all values): " << max_rel_diff_all << std::endl;
    
    // Determine pass/fail
    bool passed = (max_abs_diff < 0.1f) && (max_rel_diff < 0.1f);
    if (passed) {
        std::cout << "\n✓ Test PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Test FAILED" << std::endl;
    }
    
    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_cuda;
    delete[] h_O_reference;
    
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    
    return passed ? 0 : 1;
}
