#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

// Global variables to store memory access counts
__device__ unsigned long long global_mem_reads = 0;
__device__ unsigned long long global_mem_writes = 0;
__device__ unsigned long long shared_mem_reads = 0;
__device__ unsigned long long shared_mem_writes = 0;

// Instrumented memory access functions
__device__ int read_global_memory(int* ptr, int index) {
    atomicAdd(&global_mem_reads, 1ULL);
    return ptr[index];
}

__device__ void write_global_memory(int* ptr, int index, int value) {
    atomicAdd(&global_mem_writes, 1ULL);
    ptr[index] = value;
}

__device__ int read_shared_memory(int* ptr, int index) {
    atomicAdd(&shared_mem_reads, 1ULL);
    return ptr[index];
}

__device__ void write_shared_memory(int* ptr, int index, int value) {
    atomicAdd(&shared_mem_writes, 1ULL);
    ptr[index] = value;
}

// Instrumented version of the tiled matrix multiplication kernel
__global__ void instrumented_gpu_tile_mat_mul(int *d_matrix_1, int *d_matrix_2, int *d_matrix_res, int m, int n, int k, int tile_D) {
    extern __shared__ int p_tile[];
    
    int *v_tile_1 = p_tile;
    int *v_tile_2 = p_tile + tile_D*tile_D;

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    
    int sum = 0;
    for (size_t t = 0; t < ceil((double)n/tile_D); t++) {

        // Load A tile with instrumentation
        if (i < m && (t*tile_D + threadIdx.y) < n) {
            int value = read_global_memory(d_matrix_1, i*n + (t*tile_D + threadIdx.y));
            write_shared_memory(v_tile_1, threadIdx.x*tile_D + threadIdx.y, value);
        } else {
            write_shared_memory(v_tile_1, threadIdx.x*tile_D + threadIdx.y, 0);
        }

        // Load B tile with instrumentation
        if (j < k && (t*tile_D + threadIdx.x) < n) {
            int value = read_global_memory(d_matrix_2, (t*tile_D + threadIdx.x)*k + j);
            write_shared_memory(v_tile_2, threadIdx.x*tile_D + threadIdx.y, value);
        } else {
            write_shared_memory(v_tile_2, threadIdx.x*tile_D + threadIdx.y, 0);
        }

        __syncthreads();

        // Compute with instrumented shared memory reads
        for (int l = 0; l < tile_D; l++) {
            int a_val = read_shared_memory(v_tile_1, threadIdx.x * tile_D + l);
            int b_val = read_shared_memory(v_tile_2, l * tile_D + threadIdx.y);
            sum += a_val * b_val;
        }

        __syncthreads();
    }
    
    if (i < m && j < k) {
        write_global_memory(d_matrix_res, i*k + j, sum);
    }
}

// Function to get memory access statistics
void get_memory_access_stats(unsigned long long* global_reads, unsigned long long* global_writes,
                           unsigned long long* shared_reads, unsigned long long* shared_writes) {
    // Copy counters from device to host
    cudaMemcpyFromSymbol(global_reads, global_mem_reads, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(global_writes, global_mem_writes, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(shared_reads, shared_mem_reads, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(shared_writes, shared_mem_writes, sizeof(unsigned long long));
}

// Function to reset counters
void reset_memory_counters() {
    unsigned long long zero = 0;
    cudaMemcpyToSymbol(global_mem_reads, &zero, sizeof(unsigned long long));
    cudaMemcpyToSymbol(global_mem_writes, &zero, sizeof(unsigned long long));
    cudaMemcpyToSymbol(shared_mem_reads, &zero, sizeof(unsigned long long));
    cudaMemcpyToSymbol(shared_mem_writes, &zero, sizeof(unsigned long long));
}

int main() {
    // Example usage
    int m = 1024, n = 1024, k = 1024;
    int tile_size = 16;
    
    // Allocate matrices (simplified for example)
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(int));
    cudaMalloc(&d_B, n * k * sizeof(int));
    cudaMalloc(&d_C, m * k * sizeof(int));
    
    // Reset counters before kernel launch
    reset_memory_counters();
    
    // Launch configuration
    dim3 block(tile_size, tile_size);
    dim3 grid((m + tile_size - 1) / tile_size, (k + tile_size - 1) / tile_size);
    
    // Launch instrumented kernel
    instrumented_gpu_tile_mat_mul<<<grid, block, 2 * tile_size * tile_size * sizeof(int)>>>
                                 (d_A, d_B, d_C, m, n, k, tile_size);
    
    cudaDeviceSynchronize();
    
    // Get memory access statistics
    unsigned long long global_reads, global_writes, shared_reads, shared_writes;
    get_memory_access_stats(&global_reads, &global_writes, &shared_reads, &shared_writes);
    
    cout << "Memory Access Statistics:" << endl;
    cout << "Global Memory Reads: " << global_reads << endl;
    cout << "Global Memory Writes: " << global_writes << endl;
    cout << "Shared Memory Reads: " << shared_reads << endl;
    cout << "Shared Memory Writes: " << shared_writes << endl;
    cout << "Total Global Memory Accesses: " << (global_reads + global_writes) << endl;
    cout << "Total Shared Memory Accesses: " << (shared_reads + shared_writes) << endl;
    
    // Calculate theoretical values for comparison
    long long total_threads = (long long)grid.x * grid.y * block.x * block.y;
    long long tiles_per_thread = (n + tile_size - 1) / tile_size;
    
    // Calculate total theoretical memory accesses
    long long expected_global_reads_per_thread = 2 * tiles_per_thread;
    long long expected_shared_reads_per_thread = tile_size * tiles_per_thread;
    long long expected_shared_writes_per_thread = 2 * tiles_per_thread; // Loading A and B tiles
    long long expected_global_writes_per_thread = 1; // Writing result once per thread
    
    long long total_expected_global_reads = expected_global_reads_per_thread * total_threads;
    long long total_expected_global_writes = expected_global_writes_per_thread * total_threads;
    long long total_expected_shared_reads = expected_shared_reads_per_thread * total_threads;
    long long total_expected_shared_writes = expected_shared_writes_per_thread * total_threads;
    
    cout << "\nTheoretical Analysis:" << endl;
    cout << "Total threads: " << total_threads << endl;
    cout << "Tiles per thread: " << tiles_per_thread << endl;
    cout << "Expected global reads per thread: " << expected_global_reads_per_thread << endl;
    cout << "Expected shared reads per thread: " << expected_shared_reads_per_thread << endl;
    cout << "\nTotal Expected Memory Accesses:" << endl;
    cout << "Total Global Memory Reads: " << total_expected_global_reads << endl;
    cout << "Total Global Memory Writes: " << total_expected_global_writes << endl;
    cout << "Total Shared Memory Reads: " << total_expected_shared_reads << endl;
    cout << "Total Shared Memory Writes: " << total_expected_shared_writes << endl;
    cout << "Total Global Memory Accesses: " << (total_expected_global_reads + total_expected_global_writes) << endl;
    cout << "Total Shared Memory Accesses: " << (total_expected_shared_reads + total_expected_shared_writes) << endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
