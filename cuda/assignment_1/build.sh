#!/bin/bash

# Build all CUDA programs in this directory
echo "Compiling all CUDA programs..."

nvcc main_1_1.cu -o main_1_1 -arch=sm_61
nvcc main_1_2.cu -o main_1_2 -arch=sm_61
nvcc program_2.cu -o program_2 -arch=sm_61
nvcc program_4.cu -o program_4 -arch=sm_61
nvcc program3.cu -o program3 -arch=sm_61
#nvcc temp.cu -o temp -arch=sm_61
