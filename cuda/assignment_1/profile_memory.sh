#!/bin/bash

# Script to profile memory access using NVIDIA profiling tools

echo "CUDA Memory Access Profiling Script"
echo "==================================="

# Check if nvprof is available
if command -v nvprof &> /dev/null; then
    echo "Using nvprof for profiling..."
    
    # Compile the program
    nvcc main_2.cu -o main_2_profiled -arch=sm_61
    
    # Profile global memory transactions
    echo "Profiling global memory transactions..."
    echo "2" | nvprof --metrics gld_transactions,gst_transactions,shared_load_transactions,shared_store_transactions ./main_2_profiled 16 public_test_cases/matrix1.csv public_test_cases/matrix2.csv
    
    echo ""
    echo "Profiling memory efficiency..."
    echo "2" | nvprof --metrics gld_efficiency,gst_efficiency,shared_efficiency ./main_2_profiled 16 public_test_cases/matrix1.csv public_test_cases/matrix2.csv
    
    echo ""
    echo "Profiling memory throughput..."
    echo "2" | nvprof --metrics gld_throughput,gst_throughput ./main_2_profiled 16 public_test_cases/matrix1.csv public_test_cases/matrix2.csv

elif command -v ncu &> /dev/null; then
    echo "Using Nsight Compute (ncu) for profiling..."
    
    # Compile the program
    nvcc main_2.cu -o main_2_profiled -arch=sm_61
    
    # Profile with Nsight Compute
    echo "2" | ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld,l1tex__t_bytes_pipe_lsu_mem_global_op_st,l1tex__t_bytes_pipe_lsu_mem_shared_op_ld,l1tex__t_bytes_pipe_lsu_mem_shared_op_st ./main_2_profiled 16 public_test_cases/matrix1.csv public_test_cases/matrix2.csv

else
    echo "Neither nvprof nor ncu found. Please install NVIDIA profiling tools."
    exit 1
fi

echo ""
echo "Profiling complete!"
