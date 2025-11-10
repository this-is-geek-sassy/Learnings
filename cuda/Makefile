# Makefile to compile all .cu files in the current directory into binaries
# Usage:
#   make            -> builds all .cu files using default compute capability (61 -> sm_61)
#   make CC=70      -> builds with compute capability 70 (sm_70)
#   make NVCC=/path/to/nvcc CC=75 -> specify nvcc and compute capability

NVCC ?= nvcc
# Always enable -O3 optimization as requested
CFLAGS := -O3
# Accept CC as a number like 61, 70, 75. If CC is not provided, default to 61.
CC ?= 61
SM := sm_$(CC)

# Find all .cu files in this directory and subdirectories (recursive)
# SRCS will contain paths like ./subdir/file.cu
ALL_CU := $(shell find . -type f -name '*.cu')

# Only build binaries for .cu files that contain a 'main(' definition.
# This avoids trying to link helper/source-only .cu files that don't define main.
SRCS := $(shell find . -type f -name '*.cu' -print0 | xargs -0 grep -Il 'int main' || true)

# If none found, fall back to compiling all cu files
ifeq ($(strip $(SRCS)),)
SRCS := $(ALL_CU)
endif

# Targets: for each src ./path/foo.cu create a binary ./path/foo
BINS := $(SRCS:.cu=)

.PHONY: all clean help prebuild postbuild

# Temporary file to record failed targets (kept in repo root)
BUILD_FAILS := .build_failures

all: prebuild $(BINS)
	@$(MAKE) postbuild

prebuild:
	@rm -f $(BUILD_FAILS)

postbuild:
	@if [ -f $(BUILD_FAILS) ]; then \
		echo "Some builds failed:"; cat $(BUILD_FAILS); \
		exit 0; \
	else \
		echo "All builds succeeded."; \
	fi

# Rule to build each binary from its .cu source
%: %.cu
	@echo "Compiling $< -> $@ (arch=$(SM))"
	@mkdir -p $(dir $@)
	@$(NVCC) $(CFLAGS) $< -o $@ -arch=$(SM) 2>&1 || { \
		echo "$@" >> $(BUILD_FAILS); \
		echo "(continuing)"; \
		true; \
	}

clean:
	rm -f $(BINS) *.o *~
	@echo "Cleaned binaries"

help:
	@echo "Makefile usage"
	@echo "  make           # build all .cu files in this directory (default CC=$(CC) -> arch=$(SM))"
	@echo "  make CC=70     # build with compute capability 70 (arch=sm_70)"
	@echo "  make NVCC=/path/to/nvcc CC=75"
	@echo "  make clean"
