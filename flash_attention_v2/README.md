# Flash Attention V2 Implementation

This directory contains both a **Python simulation** and a **CUDA implementation** of Flash Attention V2, focusing on the split-KV parallelization architecture that enables significantly better GPU utilization than V1.

## Key Insight: Split-KV Parallelization

V2's main innovation is **parallelizing over both query tiles AND key-value tiles**:

**V1 Architecture:**
- Grid: `(num_q_tiles, batch×heads)` 
- Each block processes 1 Q tile against ALL KV tiles sequentially
- For L=256, BQ=8: **32 blocks total**

**V2 Architecture:**
- Grid: `(num_q_tiles, num_kv_blocks, batch×heads)`
- Each block processes 1 Q tile against a SUBSET of KV tiles
- For L=256, BQ=8, KV_TILES_PER_BLOCK=4: **256 blocks = 8× more parallelism**

This requires a **two-kernel design**:
1. **Forward kernel**: Computes partial results in parallel
2. **Reduction kernel**: Combines partial results with stable softmax rescaling

## Trade-offs and Design Decisions

**V2 requires workspace memory** to store partial results:
- Overhead: ~10-20% additional memory
- Trade-off: **workspace + reduction cost << parallelism gains**

**Tunable parallelism** via `KV_TILES_PER_BLOCK`:
- Smaller values (1-2): Maximum parallelism, more workspace
- Larger values (8+): Less parallelism, approaches V1 behavior
- Default (4): Good balance

## Python Simulation

The Python implementation (`numpy_gpu_like.py`) demonstrates the algorithm structure:

```bash
python numpy_gpu_like.py
```

**Key functions:**
- `partial_attention_kernel()`: Simulates forward pass GPU blocks
- `reduction_kernel()`: Simulates reduction pass GPU blocks  
- `process_kv_tile()`: Core streaming softmax (unchanged from V1)

This code is structured to directly translate to CUDA with minimal changes.

## CUDA Implementation

The CUDA implementation (`CUDA/`) provides an educational, performant GPU version.

### Running CUDA Tests

```bash
cd CUDA
make
./flash_attention_v2
```

### Key Features

**1. D-Tiling for Memory Efficiency**

Loads small chunks instead of full [BQ, D] tiles:
- Q@K^T: Process d-dimension in chunks of d_tile_qk=16
- S@V: Process d-dimension in chunks of d_tile_v=16
- **Memory savings: 90.7%** (10.47 KB → 0.97 KB shared memory)

**2. Register-Based Output Accumulator**

Each thread owns output elements in registers, not shared memory:
```cuda
constexpr int elems_per_thread = 4;  // (BQ*D) / THREADS_PER_BLOCK
float O_reg[elems_per_thread];
```

**Benefits:** Minimal shared memory (992 bytes), works on low-memory GPUs

### Performance Characteristics

Performance scales with problem size and batch configuration:

**With realistic production config (B=32, H=8, L=1024, D=128):**
- **Speedup: ~47× vs parallel C++ reference** (OpenMP)
- **Forward blocks: 262,144** (massive split-KV parallelism)
- **Workspace: 1 GB** (acceptable for production)
- **GPU time: 153 ms** vs CPU time: 7.2 seconds

**With small batch (B=2, H=2, L=1024, D=128):**
- **Speedup: ~219× vs parallel C++ reference**
- **Forward blocks: 4,096**
- **Workspace: 4 MB**

**Key insight:** Large batch sizes reduce speedup ratio because:
1. CPU OpenMP parallelization scales well with batch size
2. V2's workspace memory grows: B×H×num_q_tiles×num_kv_blocks
3. Reduction kernel overhead increases with batch size
4. Memory bandwidth can become saturated

**Performance vs sequence length (B=2, H=2):**
- L=128: ~83× speedup
- L=256: ~101× speedup
- L=512: ~144× speedup
- L=1024: ~219× speedup

The GPU advantage scales well with sequence length as parallelism increases.

### Implementation Trade-offs

Our CUDA version prioritizes **educational clarity** and **flexibility**:

| Approach | Speedup | Shared Memory | Best For |
|----------|---------|---------------|----------|
| **Register O_acc** (ours) | ~32× | 992 bytes | Learning, low-mem GPUs |
| Shared O_acc | ~134× | 5,088 bytes | Maximum performance |
| Production (CuTe) | ~200×+ | Variable | Peak performance |

**Why registers can be slower:**
- Less memory coalescing than shared memory
- Simpler work distribution vs sophisticated partitioning
- No tensor core optimizations

**When to prefer registers:**
- Devices with limited shared memory
- Educational clarity
- Flexibility for experimentation

See `CUDA/README.md` for detailed implementation insights.

## Configuration

Both implementations share these parameters:

```python
BQ = 8                    # Query tile size
BK = 8                    # Key/Value tile size
D_TILE_QK = 16           # D-tile for Q@K^T
D_TILE_V = 16            # D-tile for S@V
KV_TILES_PER_BLOCK = 4   # Controls parallelism (lower = more blocks)
```

## Numerical Stability

The reduction kernel maintains stability by tracking max values:

```python
# Find global max
m_global[i] = max(m_1[i], m_2[i], ..., m_K[i])

# Rescale before combining
scale_k[i] = exp(m_k[i] - m_global[i])
O_final[i] = sum_k(O_k[i] × scale_k[i]) / sum_k(l_k[i] × scale_k[i])
```

This prevents overflow/underflow when combining partial softmax results.

## V1 vs V2 Comparison

| Aspect | V1 | V2 |
|--------|----|----|
| **Kernels** | 1 (monolithic) | 2 (split + reduce) |
| **Parallelism** | Query tiles only | Query AND KV tiles |
| **Blocks (L=256)** | 32 | 256 (8× more) |
| **Workspace** | None | ~10-20% overhead |
| **GPU utilization** | Good | Excellent |
| **Best for** | Short sequences | Long sequences |

## References

- [Flash Attention V2 Paper](https://arxiv.org/abs/2307.08691) - Dao (2023)
- [Official CUDA Implementation](https://github.com/Dao-AILab/flash-attention)
- [Flash Attention V1](../flash_attention_v1/) - Our implementation
