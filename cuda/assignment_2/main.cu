#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

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

    char *buffer = (char *)malloc(*out_size);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    size_t read_bytes = fread(buffer, 1, *out_size, f);
    fclose(f);

    if (read_bytes != *out_size) {
        free(buffer);
        return NULL;
    }

    return buffer;
}

__global__ void gpu_parse(char *contents, uint32_t *sequences,
                          uint32_t *offsets, uint32_t *lengths,
                          uint32_t *num_seq, int filesize, int num_warps,
                          int L) {
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int chars_per_warp = (filesize + num_warps - 1) / num_warps;
    int offset = warp_id * chars_per_warp;
    int next_offset = (warp_id + 1) * chars_per_warp;
    while (__all_sync(
        0xffffffff, offset != 0 && contents[offset - 1 + threadIdx.x] != '\n' &&
                        offset - 2 + threadIdx.x < filesize))
        offset += 32;
    if (threadIdx.x == 0)
        while (offset != 0 && contents[offset - 1] != '\n' &&
               offset - 2 < filesize)
            offset++;
    offset = __shfl_sync(0xffffffff, offset, 0);
    if (offset >= next_offset)
        return;
    if (threadIdx.x == 0) {
        while (offset < next_offset && offset < filesize) {
            int i = -1;
            int start_offset;
            int sequence_number = atomicInc(num_seq, 30000);
            uint32_t num = 0;
            while (offset < filesize && contents[offset] != '\n') {
                // assuming only valid characters are 0-9 , \r \n and space
                if (contents[offset] >= '0') {
                    num *= 10;
                    num += (uint32_t)(contents[offset] - '0');
                } else if (contents[offset] == ',') {
                    if (i >= 0) {
                        sequences[L * sequence_number + i] = num;
                    } else {
                        start_offset = offset;
                    }
                    i++;
                    num = 0;
                }
                offset++;
            }
            sequences[L * sequence_number + i] = num;
            i++;
            offsets[sequence_number] = start_offset;
            lengths[sequence_number] = i;
            offset++;
        }
    }
}

__global__ void gpu_write(char *contents, uint32_t *sequences,
                          uint32_t *offsets, uint32_t *lengths,
                          uint32_t *num_seq, int num_warps, int L) {
    extern __shared__ uint32_t char_lengths_shared[];
    uint32_t *char_lengths = &char_lengths_shared[64 * threadIdx.y];
    int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int sequences_per_warp = ((*num_seq + num_warps - 1) / num_warps);
    int first_sequence = sequences_per_warp * warp_id;
    int total_seq = *num_seq;
    for (int sequence_number = first_sequence;
         sequence_number < sequences_per_warp * (warp_id + 1) &&
         sequence_number < total_seq;
         sequence_number++) {
        int i = lengths[sequence_number];
        int start_offset = offsets[sequence_number];
        for (int pref_offset = 0; pref_offset < i; pref_offset += 64) {
            char num1[16];
            char num2[16];
            int num1_len = 0, num2_len = 0;
            int num1_ind = pref_offset + threadIdx.x * 2;

            // num1
            if (num1_ind < i) {
                int rem = sequences[L * sequence_number + num1_ind];
                if (rem == 0) {
                    num1[0] = '0';
                    num1_len = 1;
                } else {
                    while (rem) {
                        int digit = rem % 10;
                        rem /= 10;
                        num1[num1_len] = (char)('0' + digit);
                        num1_len++;
                    }
                }
                char_lengths[num1_ind - pref_offset] = num1_len;
            }
            // num2
            if (num1_ind + 1 < i) {
                int rem = sequences[L * sequence_number + num1_ind + 1];
                if (rem == 0) {
                    num2[0] = '0';
                    num2_len = 1;
                } else {
                    while (rem) {
                        int digit = rem % 10;
                        rem /= 10;
                        num2[num2_len] = (char)('0' + digit);
                        num2_len++;
                    }
                }
                char_lengths[num1_ind + 1 - pref_offset] = num2_len;
            }
            int max_gap = i - pref_offset;
            if (max_gap > 64)
                max_gap = 64;
            for (int gap = 1; gap < max_gap; gap *= 2) {
                int sub_index = threadIdx.x % gap;
                int pref_sub_offset = (2 * (threadIdx.x / gap) + 1) * gap;
                if (pref_sub_offset + sub_index < max_gap) {
                    char_lengths[pref_sub_offset + sub_index] +=
                        char_lengths[pref_sub_offset - 1];
                }
                __syncwarp();
            }
            if (num1_ind < i) {
                int num_1_offset =
                    start_offset +
                    (threadIdx.x != 0 ? char_lengths[num1_ind - 1 - pref_offset]
                                      : 0) +
                    threadIdx.x * 2;
                contents[num_1_offset] = ',';
                num_1_offset++;
                while (num1_len) {
                    num1_len--;
                    contents[num_1_offset] = num1[num1_len];
                    num_1_offset++;
                }
            }
            if (num1_ind + 1 < i) {
                int num_2_offset = start_offset +
                                   char_lengths[num1_ind - pref_offset] +
                                   threadIdx.x * 2 + 1;
                contents[num_2_offset] = ',';
                num_2_offset++;
                while (num2_len) {
                    num2_len--;
                    contents[num_2_offset] = num2[num2_len];
                    num_2_offset++;
                }
            }

            if (pref_offset + 63 < i)
                start_offset += char_lengths[63] + 64;
        }
    }
}
__global__ void gpu_sort(uint32_t *matrix_alike, uint32_t *row_offsets, uint32_t *row_sizes, uint32_t *num_seq, int L, uint32_t *sorted_matrix, unsigned int max_value) {
    // sort the sequences
    int row_idx = blockIdx.x;
    int row_start = row_offsets[row_idx];
    int row_size = row_sizes[row_idx];
    unsigned int indiv_elem = row_start + threadIdx.x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("DEBUG KERNEL LAUNCHED total_flat_size=%d max_value=%u row_start=%d row_size=%d\n",
    //         total_flat_size, max_value, row_start, row_size);
    // }
    // __syncthreads();

    extern __shared__ unsigned int shared_mem[];
    unsigned int *count = shared_mem;
    // count array initialization
    count[threadIdx.x] = 0;
    __syncthreads();

    // trying to count frequancies
    if (threadIdx.x < row_size) {
        unsigned int indiv_elem = row_start + 1 + threadIdx.x; // +1 to skip row size
        atomicAdd(&count[matrix_alike[indiv_elem]], 1);
    }

    // trying prefix sum now
    // Serial prefix sum (more reliable for counting sort)
    if (threadIdx.x == 0) {
        // Convert counts to cumulative counts
        for (int i = 1; i <= max_value; i++) {
            count[i] += count[i-1];
        }
        
        // Convert to exclusive prefix sum (shift right)
        for (int i = max_value; i > 0; i--) {
            count[i] = count[i-1];
        }
        count[0] = 0;
    }
    __syncthreads();

    //DEBUG
    // printf("HELLO!\n");
    // if (row_idx == 0 && threadIdx.x == 0) {
    //     printf("HELLO!\n");
    //     for (int i=0; i<256; i++) {
    //         printf("%d ", count[i]);
    //     }
    //     printf("\n");
    // }

    // trying to place stored value now
    if (threadIdx.x < row_size) {
        unsigned int data_idx = row_start + 1 + threadIdx.x; // +1 to skip row size
        unsigned int value = matrix_alike[data_idx];
        
        // Get position and increment for next element with same value
        int pos = atomicAdd(&count[value], 1); // atomicAdd, not atomicSub!
        sorted_matrix[row_start + 1 + pos] = value;
    }
    
    // Copy the row size to output
    if (threadIdx.x == 0) {
        sorted_matrix[row_start] = matrix_alike[row_start];
    }
}

int main(int argc, char **argv) {
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
    string input_filepath = argv[1];
    string output_filename = argv[3];
    int L = 128;

    int filesize;
    char *contents = read_file(input_filepath, &filesize);

    const int NUM_WARPS = 8192;
    const int WARPS_PER_BLOCK = 4;
    dim3 block(32, WARPS_PER_BLOCK);

    char *contents_gpu;
    cudaMalloc(&contents_gpu, filesize);

    // Stores all sequences
    uint32_t *sequences;

    // Stores the offsets in the file from which to start writing each sequnce.
    // These offsets need not be in ascending order
    uint32_t *offsets;

    // Stores the length of each sequence
    uint32_t *lengths;

    // a single integer storing total number of integers
    uint32_t *num_seq;

    cudaMalloc(&sequences, (30000 * L + 30000 + 30000 + 1) * sizeof(uint32_t));
    offsets = sequences + 30000 * L;
    lengths = offsets + 30000;
    num_seq = lengths + 30000;
    // making sure num_seq is zero to begin with
    cudaMemset(num_seq, 0, 1);

    cudaMemcpy(contents_gpu, contents, filesize, cudaMemcpyHostToDevice);

    // auto time_cudaMemcpy = chrono::high_resolution_clock::now();

    cudaEvent_t start, parse, sort, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&parse);
    cudaEventCreate(&sort);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gpu_parse<<<(NUM_WARPS + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, block>>>(
        contents_gpu, sequences, offsets, lengths, num_seq, filesize, NUM_WARPS,
        L);
    cudaEventRecord(parse);

    uint32_t num_seq_host = 0;
    cudaMemcpy(&num_seq_host, num_seq, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cout << "Number of rows (sequences): " << num_seq_host << endl;

    // Copy lengths from GPU
    uint32_t *lengths_host = new uint32_t[num_seq_host];
    cudaMemcpy(lengths_host, lengths, num_seq_host * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Now compute total size
    uint32_t total_size = 0;
    for (uint32_t i = 0; i < num_seq_host; i++) {
        total_size += lengths_host[i];
    }
    cout << "Total number of elements across all sequences = " << total_size << endl;

    uint32_t *sorted_sequences;
    cudaMalloc(&sorted_sequences, total_size * sizeof(uint32_t));
    


    gpu_sort<<<num_seq_host, 256, 256*sizeof(uint32_t)>>>(sequences, offsets, lengths, num_seq, L, sorted_sequences, 255); //complete

    cudaEventRecord(sort);
    gpu_write<<<(NUM_WARPS + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, block,
                WARPS_PER_BLOCK * 64 * sizeof(uint32_t)>>>(
        contents_gpu, sorted_sequences, offsets, lengths, num_seq, NUM_WARPS, L);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
        exit(1);
    }
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, parse, sort);
    printf("Sorting kernel execution time: %f us\n", ms * 1000);
    cudaEventElapsedTime(&ms, start, parse);
    printf("Parsing time: %f us\n", ms * 1000);
    cudaEventElapsedTime(&ms, sort, stop);
    printf("Writing time: %f us\n", ms * 1000);

    cudaMemcpy(contents, contents_gpu, filesize, cudaMemcpyDeviceToHost);

    write_file(output_filename, contents, filesize);
}
