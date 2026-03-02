# Wheeler Memory — GPU kernel build
#
# Usage:
#   make              Build libwheeler_ca.so (v1, 64×64 fixed grid)
#   make v2           Build libwheeler_ca_v2.so (v2, variable grid size)
#   make all          Build both v1 and v2
#   make clean        Remove build artifacts
#   make test         Verify .so exports ca_evolve_* symbols (v1)
#   make test-v2      Verify ca_evolve_*_v2 symbols (v2)
#   make verify       Run bench_gpu.py --verify-only (CPU vs GPU correctness)

# Auto-detect installed GPU arch; fall back to gfx1201 (RDNA4 / RX 9070)
GPU_ARCH ?= $(shell rocminfo 2>/dev/null | grep -o 'gfx[0-9]*' | head -1)
ifeq ($(GPU_ARCH),)
    GPU_ARCH := gfx1201
endif

HIPCC  ?= hipcc
CFLAGS  = --offload-arch=$(GPU_ARCH) -O3 -shared -fPIC

TARGET_V1 = libwheeler_ca.so
TARGET_V2 = libwheeler_ca_v2.so
SRC_V1    = ca_kernel.hip
SRC_V2    = ca_kernel_v2.hip

.PHONY: all v2 clean test test-v2 verify

all: $(TARGET_V1) $(TARGET_V2)
	@echo "Built $(TARGET_V1) and $(TARGET_V2) for $(GPU_ARCH)"

$(TARGET_V1): $(SRC_V1)
	$(HIPCC) $(CFLAGS) -o $@ $<
	@echo "Built $(TARGET_V1) for $(GPU_ARCH)"

v2: $(TARGET_V2)

$(TARGET_V2): $(SRC_V2)
	$(HIPCC) $(CFLAGS) -o $@ $<
	@echo "Built $(TARGET_V2) for $(GPU_ARCH)"

clean:
	rm -f $(TARGET_V1) $(TARGET_V2)

test: $(TARGET_V1)
	@echo "=== Library: $(TARGET_V1) ==="
	@ls -lh $(TARGET_V1)
	@echo "=== Exported symbols ==="
	@nm -D $(TARGET_V1) | grep "ca_evolve" || (echo "ERROR: symbols missing"; exit 1)
	@echo "OK"

test-v2: $(TARGET_V2)
	@echo "=== Library: $(TARGET_V2) ==="
	@ls -lh $(TARGET_V2)
	@echo "=== Exported symbols ==="
	@nm -D $(TARGET_V2) | grep "ca_evolve" || (echo "ERROR: symbols missing"; exit 1)
	@echo "OK"

verify: $(TARGET_V1)
	@echo "=== CPU vs GPU correctness ==="
	cd ../.. && python scripts/bench_gpu.py --verify-only
