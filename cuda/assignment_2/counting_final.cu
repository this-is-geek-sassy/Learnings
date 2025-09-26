#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda.h>

using namespace std;

// ================================================================
// Template file I/O (kept from main.cu)
// ================================================================
int write_file(string filename, char *data, int size) {
    int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -1;
    int written = 0;
    while (written < size) {
        int ret = write(fd, data + written, size - written);
        if (ret < 0) {
            close(fd);
            return -1;
        }
        written += ret;
    }
    close(fd);
    return 0;
}

char *read_file(string &filename, int *out_size) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f)
        return NULL;

    struct stat st;
    if (stat(filename.c_str(), &st) != 0) {
        fclose(f);
        return NULL;
    }
    *out_size = st.st_size;

    char *buf = (char *)malloc(*out_size);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    int read_sz = fread(buf, 1, *out_size, f);
    fclose(f);
    if (read_sz != *out_size) {
        free(buf);
        return NULL;
    }
    return buf;
}

// ================================================================
// User kernel (from raddix.cu)
// ================================================================
__global__ void alt_gpu_counting_sort(int *d_input, int *d_output, int n, int exp, int base)
{
    extern __shared__ int count[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = threadIdx.x; i < base; i += blockDim.x)
        count[i] = 0;
    __syncthreads();

    if (tid < n) {
        int digit = (d_input[tid] / exp) % base;
        atomicAdd(&count[digit], 1);
    }
    __syncthreads();

    // prefix sum
    for (int i = 1; i < base; i++) {
        count[i] += count[i - 1];
    }
    __syncthreads();

    if (tid < n) {
        int digit = (d_input[tid] / exp) % base;
        int pos = atomicSub(&count[digit], 1) - 1;
        d_output[pos] = d_input[tid];
    }
}

// ================================================================
// GPU Radix Sort driver using kernel
// ================================================================
void gpu_radix_sort(int *h_data, int n) {
    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    int base = 10;
    int max_val = 0;
    for (int i = 0; i < n; i++)
        max_val = max(max_val, h_data[i]);

    for (int exp = 1; max_val / exp > 0; exp *= base) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        alt_gpu_counting_sort<<<blocks, threads, base * sizeof(int)>>>(d_input, d_output, n, exp, base);
        cudaDeviceSynchronize();

        // swap pointers for next pass
        int *tmp = d_input;
        d_input = d_output;
        d_output = tmp;
    }

    cudaMemcpy(h_data, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

// ================================================================
// Main program (template + user kernel)
// ================================================================
int main(int argc, char *argv[]) {
    if (argc < 4 || string(argv[2]) != "-o") {
        string command;
        for (int i = 0; i < argc; i++) {
            command += argv[i];
            command += " ";
        }
        cerr << "Usage: ./main <path_to_the_input_file>.csv -o "
                "<output_file_name>.csv . But command run was: "
                << command << "\n";
        return 1;
    }

    string input_filename = argv[1];
    string output_filename = argv[2];

    int in_size;
    char *in_buf = read_file(input_filename, &in_size);
    if (!in_buf) {
        cerr << "Failed to read input file" << endl;
        return 1;
    }

    int n = in_size / sizeof(int);
    int *h_data = (int *)in_buf;

    gpu_radix_sort(h_data, n);

    if (write_file(output_filename, (char *)h_data, n * sizeof(int)) < 0) {
        cerr << "Failed to write output file" << endl;
        free(in_buf);
        return 1;
    }

    free(in_buf);
    return 0;
}
