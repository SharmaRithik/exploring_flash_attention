# Makefile for Flash Attention V2 CUDA Implementation

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -Xcompiler -fopenmp
LDFLAGS = -Xcompiler -fopenmp
# Adjust -arch flag based on your GPU:
# sm_70 for V100, sm_80 for A100, sm_86 for RTX 3090, sm_89 for RTX 4090

# Tuning parameters (can be overridden)
# Using balanced config as default
BQ ?= 16
BK ?= 16
D_TILE_QK ?= 32
D_TILE_V ?= 32
D ?= 128
KV_TILES_PER_BLOCK ?= 64  # Higher values reduce kernel launch overhead
THREADS_PER_BLOCK ?= 256

# Build flags
DEFINES = -DBQ=$(BQ) -DBK=$(BK) -DD_TILE_QK=$(D_TILE_QK) -DD_TILE_V=$(D_TILE_V) -DD=$(D) -DKV_TILES_PER_BLOCK=$(KV_TILES_PER_BLOCK) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK)

TARGET = flash_attention_v2
TARGET_OPT = flash_attention_v2_opt
SOURCES = driver.cu
HEADERS = flash_attention_v2.h ../../common/standard.h
HEADERS_OPT = flash_attention_v2_opt.h ../../common/standard.h

all: $(TARGET)

# Build V2 version
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(DEFINES) $(SOURCES) -o $(TARGET) $(LDFLAGS)

# Build V2 optimized version
$(TARGET_OPT): $(SOURCES) $(HEADERS_OPT)
	$(NVCC) $(NVCC_FLAGS) $(DEFINES) -DUSE_OPT_VERSION $(SOURCES) -o $(TARGET_OPT) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_OPT)

run: $(TARGET)
	./$(TARGET)

run_opt: $(TARGET_OPT)
	./$(TARGET_OPT)

# Print configuration
config:
	@echo "Build Configuration:"
	@echo "  BQ = $(BQ)"
	@echo "  BK = $(BK)"
	@echo "  D_TILE_QK = $(D_TILE_QK)"
	@echo "  D_TILE_V = $(D_TILE_V)"
	@echo "  D = $(D)"
	@echo "  KV_TILES_PER_BLOCK = $(KV_TILES_PER_BLOCK)"
	@echo "  THREADS_PER_BLOCK = $(THREADS_PER_BLOCK)"

# Convenience targets for different configurations
fast: clean
	$(MAKE) BQ=16 BK=16 D_TILE_QK=32 D_TILE_V=32 KV_TILES_PER_BLOCK=2

parallel: clean
	$(MAKE) BQ=8 BK=8 D_TILE_QK=16 D_TILE_V=16 KV_TILES_PER_BLOCK=8

balanced: clean
	$(MAKE) BQ=8 BK=8 D_TILE_QK=16 D_TILE_V=16 KV_TILES_PER_BLOCK=4

.PHONY: all clean run run_opt config fast parallel balanced
