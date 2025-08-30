#!/bin/bash

mkdir target/results
# Build all CUDA programs in this directory
echo "Compiling all CUDA programs..."

nvcc main_1_1.cu -o ./target/main_1_1 -arch=sm_61
nvcc main_1_2.cu -o ./target/main_1_2 -arch=sm_61
nvcc main_2.cu -o ./target/main_2 -arch=sm_61
nvcc main_3.cu -o ./target/main_3 -arch=sm_61
nvcc main_4.cu -o ./target/main_4 -arch=sm_61
# nvcc program_2.cu -o program_2 -arch=sm_61
# nvcc program_4.cu -o program_4 -arch=sm_61
# nvcc temp.cu -o temp -arch=sm_61
