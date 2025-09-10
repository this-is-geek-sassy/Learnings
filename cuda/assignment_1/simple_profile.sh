#!/bin/bash

# Simple memory profiling script that bypasses interactive input

echo "CUDA Memory Access Profiling Demo"
echo "================================="

# First, let's try with nvprof on a simple existing program
echo "1. Using nvprof for basic memory analysis..."

# Profile the memory profiler program (non-interactive)
if [ -f "./memory_profiler" ]; then
    echo "Profiling the memory_profiler program:"
    nvprof --metrics gld_transactions,gst_transactions,shared_load_transactions,shared_store_transactions ./memory_profiler
    
    echo ""
    echo "Memory efficiency metrics:"
    nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./memory_profiler
    
    echo ""
    echo "Memory throughput metrics:"
    nvprof --metrics gld_throughput,gst_throughput ./memory_profiler
else
    echo "memory_profiler not found. Let's compile it first."
    nvcc memory_profiler.cu -o memory_profiler -arch=sm_61
    echo "Now running profiling..."
    nvprof --metrics gld_transactions,gst_transactions,shared_load_transactions,shared_store_transactions ./memory_profiler
fi

echo ""
echo "2. Alternative: Using ncu (Nsight Compute)..."
echo "For more detailed analysis, you can use:"
echo "ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld ./memory_profiler"
echo ""
echo "Profiling complete!"
